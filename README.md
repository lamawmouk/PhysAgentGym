# PhysAgentGym

An agentic code-execution gym in which an LLM reads a natural-language physics
scenario, writes Python (NumPy/SciPy) that simulates it, and emits a trajectory
scored by **physical-law verifiers** — closed-form, annotation-free predicates
(energy conservation, parabolic fit, circular-motion constancy, amplitude
decay, …) compiled from natural-language standards by a rule-based compiler on
top of the vendored [`physjudge`](src/physjudge/) library.

Companion code for the NeurIPS 2026 paper *"PhysAgentGym: Free
Physical-Law Verifiers for Training Small Code-Reasoning Agents."*
Project page: https://lamawmouk.github.io/PhysAgentGym/

```
scenario ──► agent writes Python ──► sandboxed exec ──► trajectory (t, pos)
                    ▲                                        │
                    └── per-standard verdicts ◄── verifier ◄─┘
                        (compiled from NL standards, no answer key)
```

## Install

```bash
git clone https://github.com/lamawmouk/PhysAgentGym.git && cd PhysAgentGym
pip install -e .                 # core: gym + verifier (numpy/scipy only)
pip install -e ".[llm]"          # + Anthropic/OpenAI API clients
pip install -e ".[training]"     # + QLoRA fine-tuning stack (GPU)
pip install -e ".[dev]"          # + pytest
```

`physjudge` is vendored in this repository — no external checkout needed.

One data file ships compressed to fit GitHub's size limit; decompress once
after cloning:

```bash
gunzip -k data/circular_motion/train_tasks.jsonl.gz
```

The composite subcategories (drag, kepler, driven, coupled) ship their test
splits; the train splits are regenerated deterministically:

```bash
python scripts/generate_harder_tasks.py   # writes data/{drag,kepler,driven,coupled}/
```

## Trained adapters

QLoRA adapters for Qwen2.5-Coder-{1.5B, 3B, 7B}-Instruct (37–80 MB each) are
distributed through the Hugging Face Hub rather than git:
TODO-HF-LINK. Download one into `checkpoints/` and pass it as `--adapter_path`
below.

## Quickstart

**Offline smoke test (no API key, no GPU):**

```bash
pytest tests/ -k mock
```

**Roll out a frontier agent** (needs `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`):

```bash
python scripts/rollout.py --task pe_ke --provider anthropic --n_tasks 10
python scripts/rollout.py --task damping --provider openai --model gpt-5 \
    --reasoning_effort high --n_tasks 10
```

**Roll out a local model** (zero-shot base, or with a trained adapter):

```bash
python scripts/rollout.py --task damping --provider hf \
    --model Qwen/Qwen2.5-Coder-7B-Instruct \
    --adapter_path checkpoints/qwen-coder-7b-physagent \
    --mode test --n_tasks 30 --seed 42
```

**Summarize results:**

```bash
python scripts/analyze_results.py results/damping_test_*.jsonl --by __source
```

## Train an adapter

QLoRA SFT on the verifier-filtered corpus (11–25 min on one L40S):

```bash
python scripts/train_sft.py --data data/sft/all_dense.jsonl \
    --base_model Qwen/Qwen2.5-Coder-7B-Instruct \
    --output_dir checkpoints/qwen-coder-7b-physagent --qlora --seed 42
```

To rebuild the SFT corpus from your own rollouts, see
`scripts/build_sft_dataset.py`.

## Repository layout

```
src/
  physics_gym/        the gym
    env/              PhysicsEnv + task classes (pe_ke, linear_motion, ...)
    agent/            ReAct-style agent: JSON actions, retry, prompting
    llm/              clients: Anthropic, OpenAI (reasoning_effort), HF+PEFT, Mock
    utils/compiler.py NL standard -> executable predicate (regex rules + fallback)
  physjudge/          vendored verifier library: Trajectory, checks, compilers
data/<subcategory>/   300 train / 30 test instances each, with golden trajectory
data/sft/             verifier-filtered SFT corpora (491 combined / 508 sparse)
scripts/
  rollout.py          run any model through the gym, JSONL logging, concurrency
  train_sft.py        QLoRA SFT on filtered rollouts (TRL SFTTrainer)
  generate_*.py       task generators
  build_sft_dataset.py, analyze_results.py
tests/                offline mock-agent loop + per-subcategory smoke tests
```

## Data format

Each line of `data/<sub>/{train,test}_tasks.jsonl` is one task:

```jsonc
{
  "task_id": 0,
  "category": "Energy Conservation",
  "subcategory": "PE-KE",
  "regime": "oblique_throw",
  "scenario": "A point mass is launched from height 2.46 m ...",   // NL prompt
  "initial_conditions": {"x0": 0.0, "y0": 2.46, "vx0": 11.49, ...},
  "physjudge_spec": { "key_standards": ["The object follows a parabolic arc ..."] },
  "golden": {"t": [...], "pos": [[x, y], ...], "fps": 60.0}  // withheld from agent & verifier
}
```

The agent must print one JSON trajectory `{"t": [...], "pos": [[x,y],...]}` on
the last stdout line; the verifier scores it against the compiled standards
only — the `golden` trajectory is used solely for benchmark validation.

## Extending the benchmark

**Add a task regime:** write a generator emitting the schema above with a
closed-form solver for `golden` (see `scripts/generate_damping_tasks.py`), and
a task class in `src/physics_gym/env/task/` (subclass, ~20 lines).

**Add a verifier primitive:** implement `fn(traj: Trajectory, ...) -> Verdict`
and register a regex rule in `src/physics_gym/utils/compiler.py` (see
`amplitude_decay` / `final_at_rest` for the pattern). Standards that match no
rule fall through to `physjudge`'s hybrid compiler.

## Tests

```bash
pytest            # offline: mock-agent loop + smoke tests for all 4 subcategories
```

## License & citation

Code: MIT. Benchmark data (`data/`): CC-BY-4.0.

```bibtex
@inproceedings{moukheiber2026physagentgym,
  title     = {PhysAgentGym: Free Physical-Law Verifiers for Training Small Code-Reasoning Agents},
  author    = {Moukheiber, Lama},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```
