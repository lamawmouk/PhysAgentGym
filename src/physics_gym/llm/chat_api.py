"""Minimal LLM clients. Anthropic for production, Mock for offline tests."""
import os
from dataclasses import dataclass, field
from typing import Callable


def make_system_message(content: str) -> dict:
    return {"role": "system", "content": content}


def make_user_message(content: str) -> dict:
    return {"role": "user", "content": content}


def make_assistant_message(content: str) -> dict:
    return {"role": "assistant", "content": content}


@dataclass
class AnthropicClient:
    model: str = "claude-opus-4-7"
    max_tokens: int = 4096
    temperature: float = 0.2
    api_key: str | None = None
    _client: object = field(init=False, default=None)

    def __post_init__(self):
        from anthropic import Anthropic
        self._client = Anthropic(
            api_key=self.api_key or os.environ.get("ANTHROPIC_API_KEY"),
            max_retries=8,  # 50 RPM tier-1 hits 429 often under concurrency
        )

    def __call__(self, messages: list[dict]) -> tuple[str, dict]:
        system = ""
        chat = []
        for m in messages:
            if m["role"] == "system":
                system = (system + "\n" + m["content"]).strip()
            else:
                chat.append({"role": m["role"], "content": m["content"]})
        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system or None,
            "messages": chat,
        }
        # Opus 4.7 and newer reasoning-class models deprecate temperature.
        if "opus-4-7" not in self.model and "opus-4.7" not in self.model:
            kwargs["temperature"] = self.temperature
        resp = self._client.messages.create(**kwargs)
        text = resp.content[0].text
        usage = {
            "input_tokens": resp.usage.input_tokens,
            "output_tokens": resp.usage.output_tokens,
        }
        return text, usage


@dataclass
class OpenAIClient:
    """OpenAI Chat Completions client. Supports GPT-5 / GPT-4.x / o-series.

    GPT-5 and o-series reasoning models reject `temperature` (similar to Anthropic
    Opus 4.7). We detect by model-name and omit it for those.
    """
    model: str = "gpt-5"
    max_tokens: int = 4096
    temperature: float = 0.2
    api_key: str | None = None
    reasoning_effort: str | None = None  # "low" | "medium" | "high" (reasoning models only)
    _client: object = field(init=False, default=None)

    def __post_init__(self):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError(
                "OpenAIClient requires the openai SDK. Install with: pip install openai"
            ) from e
        self._client = OpenAI(
            api_key=self.api_key or os.environ.get("OPENAI_API_KEY"),
            max_retries=8,
        )

    def __call__(self, messages: list[dict]) -> tuple[str, dict]:
        # OpenAI takes messages list directly (system included as a role).
        kwargs = {
            "model": self.model,
            "messages": messages,
            "max_completion_tokens": self.max_tokens,
        }
        # GPT-5 and o-series reasoning models deprecate temperature.
        is_reasoning = any(
            tok in self.model for tok in ("gpt-5", "o1", "o3", "o4")
        )
        if not is_reasoning:
            kwargs["temperature"] = self.temperature
        elif self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
            # Reasoning tokens count against max_completion_tokens; leave headroom.
            kwargs["max_completion_tokens"] = max(self.max_tokens, 16384)
        resp = self._client.chat.completions.create(**kwargs)
        text = resp.choices[0].message.content or ""
        usage = {
            "input_tokens": int(resp.usage.prompt_tokens),
            "output_tokens": int(resp.usage.completion_tokens),
        }
        return text, usage


@dataclass
class MockClient:
    """Returns canned responses; for offline loop validation."""
    responder: Callable[[list[dict]], str]

    def __call__(self, messages: list[dict]) -> tuple[str, dict]:
        return self.responder(messages), {"input_tokens": 0, "output_tokens": 0}


@dataclass
class HFTransformersClient:
    """Local HF transformers client. For evaluating a trained LoRA adapter.

    Loads base + optional PEFT adapter, applies the chat template, sample-generates.
    Slower than vLLM but no extra server infra; fine for eval-time rollouts on
    the few-hundred-task test sets.
    """
    base_model: str
    adapter_path: str | None = None
    max_new_tokens: int = 1024
    temperature: float = 0.2
    top_p: float = 0.95
    device_map: str = "auto"
    _model: object = field(init=False, default=None)
    _tokenizer: object = field(init=False, default=None)

    def __post_init__(self):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise RuntimeError(
                "HFTransformersClient requires transformers + torch. "
                "Install with: pip install 'physagentgym[training]'"
            ) from e

        self._tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            torch_dtype=dtype,
            device_map=self.device_map if torch.cuda.is_available() else None,
        )

        if self.adapter_path:
            try:
                from peft import PeftModel
            except ImportError as e:
                raise RuntimeError("adapter_path set but peft not installed") from e
            model = PeftModel.from_pretrained(model, self.adapter_path)
            model = model.merge_and_unload()

        model.eval()
        self._model = model

    def __call__(self, messages: list[dict]) -> tuple[str, dict]:
        import torch

        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        n_in = int(inputs.input_ids.shape[-1])

        with torch.no_grad():
            out_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=self.temperature > 0,
                pad_token_id=self._tokenizer.pad_token_id,
            )

        gen_ids = out_ids[0, n_in:]
        text = self._tokenizer.decode(gen_ids, skip_special_tokens=True)
        usage = {"input_tokens": n_in, "output_tokens": int(gen_ids.shape[-1])}
        return text, usage
