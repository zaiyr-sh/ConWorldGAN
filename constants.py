import os
from typing import Dict, List, Tuple


PROJECT_PATH = os.path.abspath(os.path.dirname(__file__))


def _env_or_default(name: str, default: str) -> str:
    """Return a non-empty environment value or a portable project default."""

    return os.getenv(name) or default


# Developer-specific paths and W&B ownership can be set in the environment.
# W&B reads WANDB_API_KEY itself; credentials must never be added here.
WANDB_ENTITY = os.getenv("WANDB_ENTITY") or None
WANDB_PROJECT = _env_or_default("WANDB_PROJECT", "conworldgan")
DEFAULT_INPUT_DIR = _env_or_default(
    "CONWORLDGAN_INPUT_WORLDS_DIR", os.path.join(PROJECT_PATH, "minecraft_worlds")
)
DEFAULT_OUTPUT_DIR = _env_or_default(
    "CONWORLDGAN_OUTPUT_WORLDS_DIR", os.path.join(PROJECT_PATH, "minecraft_worlds")
)
DEFAULT_REPRESENTATION_DIR = _env_or_default(
    "CONWORLDGAN_REPR_DIR", os.path.join(PROJECT_PATH, "input", "minecraft")
)
DEFAULT_RUN_DIR = _env_or_default(
    "CONWORLDGAN_RUNS_DIR", os.path.join(PROJECT_PATH, "output")
)
DEFAULT_WINE_EXECUTABLE = _env_or_default(
    "CONWORLDGAN_WINE_BIN",
    "wine",
)
DEFAULT_MINEWAYS_EXECUTABLE = _env_or_default(
    "CONWORLDGAN_MINEWAYS_BIN",
    os.path.join(
        PROJECT_PATH,
        "minecraft",
        "mineways",
        "Mineways.app",
        "Contents",
        "Resources",
        "drive_c",
        "Program Files",
        "mineways",
        "Mineways.exe",
    ),
)
DEFAULT_MINEWAYS_SCRIPT_DIR = _env_or_default(
    "CONWORLDGAN_MINEWAYS_SCRIPT_DIR",
    os.path.join(PROJECT_PATH, "minecraft", "mineways"),
)

SUB_COORDS: Dict[str, List[float]] = dict(
    ruins=[0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
    simple_beach=[0.0, 0.5, 0.0, 1.0, 0.0, 1.0],
    desert=[0.25, 0.75, 0.0, 1.0, 0.25, 0.75],
    plains=[0.25, 0.75, 0.0, 1.0, 0.25, 0.75],
    swamp=[0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
    vanilla_village=[0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
    vanilla_mineshaft=[0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
)

# Region list and human labels + plurality for sentence templates
REGION_NAMES: List[str] = [
    "vanilla_village",
]
WORLD_LABELS_PLURAL: List[Tuple[str, bool]] = [
    ("village", False),
]

# BERT / NLP
BERT_MODEL_NAME = "bert-base-uncased"

# Define all 26 neighbor offsets with human-readable labels.
RELATIVE_OFFSETS_26 = [
    # UP layer (+y)
    (+1, +1, +1, "up-front-right"),
    (+1, +1,  0, "up-front"),
    (+1, +1, -1, "up-front-left"),
    (+1,  0, +1, "up-right"),
    (+1,  0,  0, "up"),
    (+1,  0, -1, "up-left"),
    (+1, -1, +1, "up-back-right"),
    (+1, -1,  0, "up-back"),
    (+1, -1, -1, "up-back-left"),

    # SAME layer (0y)
    ( 0, +1, +1, "front-right"),
    ( 0, +1,  0, "front"),
    ( 0, +1, -1, "front-left"),
    ( 0,  0, +1, "right"),
    # (0,0,0) is the center, we skip it
    ( 0,  0, -1, "left"),
    ( 0, -1, +1, "back-right"),
    ( 0, -1,  0, "back"),
    ( 0, -1, -1, "back-left"),

    # DOWN layer (-y)
    (-1, +1, +1, "down-front-right"),
    (-1, +1,  0, "down-front"),
    (-1, +1, -1, "down-front-left"),
    (-1,  0, +1, "down-right"),
    (-1,  0,  0, "down"),
    (-1,  0, -1, "down-left"),
    (-1, -1, +1, "down-back-right"),
    (-1, -1,  0, "down-back"),
    (-1, -1, -1, "down-back-left"),
]

REPR_TYPES = {"bert", "clip"}
