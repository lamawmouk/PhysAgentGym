import json
from pathlib import Path
from .types import Scenario


V1_REGIMES: dict[str, list[str]] = {
    "1. Object Motion and Kinematics": [
        "Linear Motion", "Parabolic Motion", "Circular Motion",
        "Rotational Motion", "Combined Translational and Rotational Motion",
    ],
    "2. Interaction Dynamics": ["Collisions", "Gravity and Free Fall"],
    "3. Energy Conservation": [
        "Potential to Kinetic Energy",
        "Conservation of Mechanical Energy",
        "Energy Loss (Damping)",
    ],
    "5. Rigid Body Dynamics": ["Rotation and Torque", "Balance and Stability", "Center of Mass"],
    "10. Anti-Physics": [
        "Defying Gravity", "Energy Creation or Perpetual Motion",
        "Time Reversal and Alteration", "Infinite Duplication and Division",
    ],
}


def load_scenarios(path: str | Path) -> list[Scenario]:
    raw = json.loads(Path(path).read_text())
    out: list[Scenario] = []
    for cat, subs in raw.items():
        for sub, items in subs.items():
            for item in items:
                out.append(Scenario(
                    prompt_index=item["Prompt_index"],
                    prompt=item["Prompt"],
                    physics=item.get("Physics", ""),
                    basic_standards=item.get("Basic_Standards", {}),
                    key_standards=list(item.get("Key_Standards", [])),
                    category=cat,
                    subcategory=sub,
                    detailed=item.get("Detailed"),
                    detailed_long=item.get("Detailed_long"),
                ))
    return out


def filter_v1(scenarios: list[Scenario]) -> list[Scenario]:
    return [s for s in scenarios
            if s.category in V1_REGIMES
            and s.subcategory in V1_REGIMES[s.category]]
