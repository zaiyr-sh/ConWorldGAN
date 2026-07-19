import os
import random
import pickle
import numpy as np
import torch
from torch.nn.functional import grid_sample
import subprocess
from constants import (
    PROJECT_PATH,
    RELATIVE_OFFSETS_26,
    REPR_TYPES,
    WANDB_ENTITY,
    WANDB_PROJECT,
)
from typing import List, Dict, Tuple
from loguru import logger
import sys
import wandb
import shutil

def set_seed(seed=0):
    """Seed Python, NumPy, and PyTorch for reproducible experiments."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

def interpolate3D(data, shape, mode='bilinear', align_corners=False):
    d_1 = torch.linspace(-1, 1, shape[0])
    d_2 = torch.linspace(-1, 1, shape[1])
    d_3 = torch.linspace(-1, 1, shape[2])
    meshz, meshy, meshx = torch.meshgrid((d_1, d_2, d_3))
    grid = torch.stack((meshx, meshy, meshz), 3)
    grid = grid.unsqueeze(0).to(data.device)

    scaled = grid_sample(data, grid, mode=mode, align_corners=align_corners)
    return scaled

def _contains_tensor(obj):
    if isinstance(obj, torch.Tensor):
        return True
    if isinstance(obj, dict):
        return any(_contains_tensor(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return any(_contains_tensor(v) for v in obj)
    return False

def save_pkl(obj, name, prepath='output/'):
    os.makedirs(prepath, exist_ok=True)
    path_pkl = os.path.join(prepath, f"{name}.pkl")
    path_pt  = os.path.join(prepath, f"{name}.pt")
    if _contains_tensor(obj):
        torch.save(obj, path_pt)
    else:
        with open(path_pkl, "wb") as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_pt(name, prepath='output/'):
    return torch.load(os.path.join(prepath, f"{name}.pt"))

def get_subdir_path(name: str = "") -> str:
    return os.path.join(PROJECT_PATH, name) + "/"

def call_wine(wine_executable, on_success, on_failure=None):
    """Run an action only when the configured Wine executable is available."""

    try:
        subprocess.run(
            [wine_executable, "--version"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        on_success()
    except (OSError, subprocess.CalledProcessError):
        if on_failure is not None:
            on_failure()
        logger.warning("Wine could not be run: {}", wine_executable)

def collect_neighbors_for_voxel(
    wrld,                    # PyAnvil World(...)
    center_yzxl: Tuple[int,int,int],  # (j=y, k=z, l=x) in *world coordinates*
    bounds: Tuple[Tuple[int,int], Tuple[int,int], Tuple[int,int]],  # coords ((y0,y1),(z0,z1),(x0,x1))
    offsets = RELATIVE_OFFSETS_26,
    out_of_bounds_token: str = "__OUT_OF_BOUNDS__"
) -> List[Dict]:
    """
    Returns a list of dicts, one per neighbor around center (excluding the center).
    Each dict: { 'pos_label', 'y', 'z', 'x', 'block_name' }
    If neighbor is outside the chosen subregion, block_name = '__OUT_OF_BOUNDS__'.
    """
    j, k, l = center_yzxl
    (y0, y1), (z0, z1), (x0, x1) = bounds

    descs = []

    # Add main block
    block = wrld.get_block((j, k, l))
    b_name = block.get_state().name

    descs.append({
        "pos_label": "center",
        "y": j, "z": k, "x": l,
        "block_name": b_name
    })

    for dy, dz, dx, label in offsets:
        ny, nz, nx = j+dy, k+dz, l+dx

        # OOB check against the selected subregion
        if not (y0 <= ny < y1 and z0 <= nz < z1 and x0 <= nx < x1):
            descs.append({
                "pos_label": label,
                "y": ny, "z": nz, "x": nx,
                "block_name": out_of_bounds_token
            })
            continue

        # In-bounds: read neighbor block from world
        b = wrld.get_block((ny, nz, nx))
        b_name = b.get_state().name  # e.g., 'minecraft:stone'
        descs.append({
            "pos_label": label,
            "y": ny, "z": nz, "x": nx,
            "block_name": b_name
        })

    return descs

def init_logger():
    logger.remove()
    logger.add(sys.stdout, colorize=True,
               format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                      + "<level>{level}</level> | "
                      + "<light-black>{file.path}:{line}</light-black> | "
                      + "{message}")

def get_tags(opt):
    return [opt.input_name.split(".")[0], str(opt.scales), str(opt.repr_type), opt.input_area_name]

def init_wandb(opt):
    os.makedirs(opt.out, exist_ok=True)
    run = wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        tags=get_tags(opt),
        dir=opt.out,
    )
    opt.out_ = run.dir

def make_repr_tensor(opt) -> torch.Tensor:
    (y0, y1), (z0, z1), (x0, x1) = opt.coords
    H = y1 - y0
    D = z1 - z0
    W = x1 - x0
    return torch.zeros((1, opt.repr_dim, H, D, W), device=opt.device)

def make_index_tensor(opt) -> torch.Tensor:
    (y0, y1), (z0, z1), (x0, x1) = opt.coords
    H = y1 - y0
    D = z1 - z0
    W = x1 - x0
    return torch.zeros((H, D, W), dtype=torch.long)

def is_repr_mode(opt) -> bool:
    return opt.repr_type in REPR_TYPES

def to_one_hot(map: torch.Tensor, uniques: List[str]) -> torch.Tensor:
    # map: (H, D, W) long
    oh = torch.zeros((1, len(uniques)) + tuple(map.shape), dtype=torch.float32)
    for i in range(len(uniques)):
        oh[0, i] = (map == i)
    return oh

def archive_directory(source_folder, output_zip):
    """Create a ZIP archive from an experiment directory."""
    shutil.make_archive(output_zip, 'zip', source_folder)
