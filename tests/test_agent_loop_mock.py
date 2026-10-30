"""Exercise the full agent <-> env loop with a MockClient. No network."""
import json


from physics_gym.agent.base import PhysicsAgent
from physics_gym.env.base import PhysicsEnv
from physics_gym.env.task.pe_ke import PEKETask
from physics_gym.llm.chat_api import MockClient
from physics_gym.utils.compiler import PhysicsGymCompiler


GOLDEN_CODE_TEMPLATE = (
    "import json\n"
    "import numpy as np\n"
    "g = 9.81\n"
    "ic = {ic}\n"
    "T = ic['duration_s']; fps = ic['fps']\n"
    "n = int(round(T * fps))\n"
    "t = np.linspace(0.0, T, n)\n"
    "x = ic['x0'] + ic['vx0'] * t\n"
    "y = ic['y0'] + ic['vy0'] * t - 0.5 * g * t * t\n"
    "pos = np.stack([x, y], axis=1)\n"
    "print(json.dumps({{'t': t.tolist(), 'pos': pos.tolist(), 'object_id': 'ball', 'fps': fps}}))\n"
)


def make_responder():
    """Stateful: first turn -> validate_code; second turn -> submit."""
    state = {"turn": 0, "ic": None}

    def respond(messages: list[dict]) -> str:
        if state["turn"] == 0:
            user_text = messages[1]["content"]
            ic = json.loads(user_text.split("Initial conditions (use these exactly):\n")[1]
                            .split("\n\nEmit your first action.")[0])
            state["ic"] = ic
            code = GOLDEN_CODE_TEMPLATE.format(ic=ic)
            state["turn"] = 1
            return json.dumps({"action": "validate_code", "params": {"code": code}})
        return json.dumps({"action": "submit", "params": {}})

    return respond


def main():
    env = PhysicsEnv(
        task_entrypoint=PEKETask,
        task_kwargs={"data_path": "data/pe_ke", "mode": "test"},
        max_steps=4,
        compiler=PhysicsGymCompiler(),
    )

    scores = []
    for tid in range(3):
        agent = PhysicsAgent(llm=MockClient(responder=make_responder()))
        obs, info = env.reset(task_id=tid)
        agent.reset()
        last_obs = obs
        for _ in range(4):
            action, params, _ = agent.act(last_obs)
            result, reward, terminated, _, _ = env.step(action, **params)
            last_obs = result
            if terminated:
                break
        scores.append(env.best_score)
        print(f"task {tid}: best_score={env.best_score:.3f} pass_rate={env.best_judge_result.pass_rate:.3f}")

    mean = sum(scores) / len(scores)
    print(f"mean = {mean:.3f}")
    assert mean > 0.85, f"agent loop with golden mock should score high; got {mean:.3f}"
    print("OK")


if __name__ == "__main__":
    main()


def test_run():
    main()
