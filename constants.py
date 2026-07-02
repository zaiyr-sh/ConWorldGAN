import os
from typing import Dict, List, Tuple

WANDB_ENTITY = "sharsheyev-zaiyr-q7-tohoku-university"

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

# Paths
PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__)))

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

ORDER = [
    "up-front-right","up-front","up-front-left","up-right","up","up-left",
    "up-back-right","up-back","up-back-left",
    "front-right","front","front-left","right","left",
    "back-right","back","back-left",
    "down-front-right","down-front","down-front-left",
    "down-right","down","down-left",
    "down-back-right","down-back","down-back-left",
]

SHORT = {
    "up-front-right":"UFR", "up-front":"UF", "up-front-left":"UFL",
    "up-right":"UR", "up":"U", "up-left":"UL",
    "up-back-right":"UBR", "up-back":"UB", "up-back-left":"UBL",
    "front-right":"FR", "front":"F", "front-left":"FL",
    "right":"R", "left":"L", "back-right":"BR", "back":"B", "back-left":"BL",
    "down-front-right":"DFR", "down-front":"DFL", "down-front-left":"DFL",
    "down-right":"DR", "down":"D", "down-left":"DL",
    "down-back-right":"DBR", "down-back":"DB", "down-back-left":"DBL",
}

HOUSE_BLOCKS = {
    "chest",
    "white bed",
    "yellow bed",
    "oak planks",
    "oak stairs",
    "cobblestone",
    "cobblestone stairs",
    "glass pane",
    "oak door",
    "oak pressure plate",
    "green carpet",
    "wall torch",
    "ladder",
    "stripped oak log",
    "stripped oak wood",
    "oak log",
}

REPR_TYPES = {"bert", "clip"}
