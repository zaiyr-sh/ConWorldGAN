"""
It computes:
  1) TPKL-Div for pattern sizes 5 and 10
  2) Average TPKL-Div over [5, 10]
  3) Normalized Levenshtein diversity
  4) Token entropy difference from the real map
  5) Rare Block Recall
  6) Rare Count Ratio
  7) House Fragmentation Index
  8) House Completeness Score
  9) Optional PyVista debug visualizations for house metrics

It supports two modes:

A) Fresh per-scale generation from a trained old World-GAN run:
   run_dir/
     generators.pth
     noise_maps.pth
     reals.pth
     noise_amplitudes.pth
     token_list.pth
     num_layer.pth

B) Existing final samples only:
   samples_dir/
     real_bdata.pt
     token_list.pt or token_list.pth
     torch_blockdata/*.pt

Notes for embedding runs:
  - If repr_type is block2vec / bert / bert_naive / neighbert, the model output is continuous.
  - To decode generated tensors to blocks, this script needs the same block embeddings used in training.
  - Pass --repr_pkl path/to/representations.pkl if automatic loading does not work.

Example, per-scale generation:
python diagnostics_original_worldgan.py \
  --run_dir output/wandb/run-XXXX/files \
  --out_dir output/wandb/run-XXXX/files/diagnostics_old \
  --generate_per_scale True \
  --num_samples 20 \
  --repr_type bert \
  --repr_pkl input/minecraft/village/natural_representations_small_32.pkl

Example, existing final samples only:
python diagnostics_original_worldgan.py \
  --samples_dir output/wandb/run-XXXX/files/random_samples \
  --out_dir output/wandb/run-XXXX/files/diagnostics_old_existing \
  --generate_per_scale False \
  --num_samples 20
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    from scipy import ndimage
except Exception:
    ndimage = None

try:
    from Levenshtein import distance as levenshtein_distance
except Exception:
    try:
        from rapidfuzz.distance.Levenshtein import distance as levenshtein_distance
    except Exception:
        levenshtein_distance = None

# These imports exist in the original World-GAN repo.
from generation.generate_noise import generate_spatial_noise
from utils import interpolate3D


# ---------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------


def torch_load(path: Path, map_location="cpu"):
    """Compatible with both old and new PyTorch defaults."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: List[dict], fieldnames: Optional[List[str]] = None) -> None:
    ensure_dir(path.parent)
    if fieldnames is None:
        fieldnames = []
        seen = set()
        for row in rows:
            for k in row.keys():
                if k not in seen:
                    seen.add(k)
                    fieldnames.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_json(path: Path, data: dict) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def load_token_list(run_dir: Optional[Path], samples_dir: Optional[Path]) -> List[str]:
    candidates = []
    if run_dir is not None:
        candidates += [run_dir / "token_list.pth", run_dir / "token_list.pt"]
    if samples_dir is not None:
        candidates += [samples_dir / "token_list.pth", samples_dir / "token_list.pt"]
    for p in candidates:
        if p.exists():
            return list(torch_load(p, map_location="cpu"))
    raise FileNotFoundError("Could not find token_list.pth or token_list.pt in run_dir/samples_dir")


def load_block2repr(args, tokens: Sequence[str], channel_dim: int):
    """
    Returns dict[token] -> torch.Tensor or None.

    If repr_type is None, no representation table is needed.
    If channel_dim == len(tokens), the run is likely one-hot/logit mode.
    If channel_dim != len(tokens), embeddings are required.
    """
    if args.repr_type in {"None", "none", "null", ""}:
        return None

    if args.repr_pkl:
        obj = load_pickle(Path(args.repr_pkl).expanduser())
        return {k: torch.as_tensor(v).float() for k, v in obj.items()}

    # Try common locations in the old repo.
    if args.input_area_name:
        area = args.input_area_name
        dim = args.repr_dim
        root = Path(args.repr_root).expanduser() if args.repr_root else Path("../input/minecraft")
        candidates = []
        if args.repr_type == "block2vec":
            candidates.append(root / area / "representations.pkl")
        elif args.repr_type == "bert":
            if dim is not None:
                candidates.append(root / area / f"natural_representations_small_{dim}.pkl")
            candidates.append(root / area / "natural_representations_small.pkl")
        elif args.repr_type == "bert_naive":
            if dim is not None:
                candidates.append(root / area / f"natural_representations_small_{dim}_naive.pkl")
        elif args.repr_type == "neighbert":
            candidates.append(root / area / "natural_representations_small_neighbors.pkl")

        for p in candidates:
            if p.exists():
                obj = load_pickle(p)
                return {k: torch.as_tensor(v).float() for k, v in obj.items()}

    if channel_dim == len(tokens):
        return None

    raise FileNotFoundError(
        "This looks like an embedding run because channels != len(token_list), "
        "but I could not load block2repr. Pass --repr_pkl path/to/representations.pkl "
        "or set --repr_root and --input_area_name."
    )


# ---------------------------------------------------------------------
# Loading old trained World-GAN pyramid
# ---------------------------------------------------------------------


def load_old_pyramid(run_dir: Path, device: torch.device):
    required = ["generators.pth", "noise_maps.pth", "reals.pth", "noise_amplitudes.pth"]
    missing = [x for x in required if not (run_dir / x).exists()]
    if missing:
        raise FileNotFoundError(f"Missing files in {run_dir}: {missing}")

    generators = torch_load(run_dir / "generators.pth", map_location=device)
    noise_maps = torch_load(run_dir / "noise_maps.pth", map_location=device)
    reals = torch_load(run_dir / "reals.pth", map_location=device)
    noise_amplitudes = torch_load(run_dir / "noise_amplitudes.pth", map_location=device)

    generators = [g.to(device).eval() for g in generators]
    reals = [r.to(device) for r in reals]
    noise_maps = [z.to(device) for z in noise_maps]
    return generators, noise_maps, reals, noise_amplitudes


def load_num_layer(run_dir: Path, default_num_layer: int) -> int:
    p = run_dir / "num_layer.pth"
    if p.exists():
        try:
            return int(torch_load(p, map_location="cpu"))
        except Exception:
            return default_num_layer
    return default_num_layer


# ---------------------------------------------------------------------
# Generation, adapted from old generate_samples.py
# ---------------------------------------------------------------------


def make_pad(num_layer: int) -> nn.Module:
    # Old branch uses ReplicationPad3d in generate_samples.py and train_single_scale.py.
    return nn.ReplicationPad3d(int(num_layer))


@torch.no_grad()
def generate_old_per_scale(
    generators: Sequence[torch.nn.Module],
    noise_maps: Sequence[torch.Tensor],
    reals: Sequence[torch.Tensor],
    noise_amplitudes: Sequence[float],
    num_samples: int,
    num_layer: int,
    device: torch.device,
    scale_v: float = 1.0,
    scale_h: float = 1.0,
    scale_d: float = 1.0,
    gen_start_scale: int = 0,
) -> Dict[int, List[torch.Tensor]]:
    """
    Returns generated continuous tensors per scale, not decoded yet.
    Tensor order is [B, C, Y, Z, X]. Old code comments call this y,z,x.
    """
    fake_by_scale: Dict[int, List[torch.Tensor]] = defaultdict(list)
    images_cur: List[torch.Tensor] = []
    m = make_pad(num_layer).to(device)
    n_pad = int(num_layer)
    channels = int(reals[0].shape[1])

    in_s = torch.zeros(reals[0].shape[0], channels, *reals[0].shape[2:], device=device)

    current_scale = 0
    for sc, (G, Z_opt, noise_amp) in enumerate(zip(generators, noise_maps, noise_amplitudes)):
        if current_scale >= len(generators):
            break
        elif sc < current_scale:
            continue

        # Shape calculation copied from old generate_samples.py.
        nz = []
        nz.append(int(round((Z_opt.shape[-3] - n_pad * 2) * scale_h)))
        nz.append(int(round((Z_opt.shape[-2] - n_pad * 2) * scale_d)))
        nz.append(int(round((Z_opt.shape[-1] - n_pad * 2) * scale_v)))
        nz = tuple(max(1, x) for x in nz)

        images_prev = images_cur
        images_cur = []

        for sample_id in range(num_samples):
            z_curr = generate_spatial_noise((1, channels, *nz), device=device)
            z_curr = m(z_curr)

            if (not images_prev) or current_scale == 0:
                I_prev = in_s
            else:
                I_prev = images_prev[sample_id]
                I_prev = interpolate3D(I_prev, nz, mode="bilinear", align_corners=True)
                I_prev = m(I_prev)

            if current_scale < gen_start_scale:
                z_curr = Z_opt

            z_in = float(noise_amp) * z_curr + I_prev
            I_curr = G(z_in.detach(), I_prev, temperature=1)

            fake_by_scale[current_scale].append(I_curr.detach().cpu())
            images_cur.append(I_curr.detach())

        current_scale += 1

    return fake_by_scale


# ---------------------------------------------------------------------
# Existing final samples mode
# ---------------------------------------------------------------------


def load_existing_final_samples(samples_dir: Path, num_samples: Optional[int] = None):
    real_candidates = [samples_dir / "real_bdata.pt", samples_dir / "real_bdata.pth"]
    real_path = next((p for p in real_candidates if p.exists()), None)
    if real_path is None:
        raise FileNotFoundError(f"Could not find real_bdata.pt in {samples_dir}")

    block_dir = samples_dir / "torch_blockdata"
    if not block_dir.exists():
        raise FileNotFoundError(f"Could not find torch_blockdata/ in {samples_dir}")

    real = torch_load(real_path, map_location="cpu").long()
    sample_paths = sorted(block_dir.glob("*.pt"), key=lambda p: p.name)
    if num_samples is not None:
        sample_paths = sample_paths[:num_samples]
    fakes = [torch_load(p, map_location="cpu").long() for p in sample_paths]
    return real, fakes


# ---------------------------------------------------------------------
# Decoding tensors to discrete block ids
# ---------------------------------------------------------------------


def decode_tensor_to_indices(x: torch.Tensor, tokens: Sequence[str], block2repr: Optional[dict]) -> torch.Tensor:
    """
    Input:
      - discrete:       [Y,Z,X]
      - logits/one-hot: [1,N,Y,Z,X]
      - embeddings:    [1,C,Y,Z,X]
    Output:
      - long [Y,Z,X]
    """
    if x.ndim == 3:
        return x.detach().cpu().long()
    if x.ndim != 5:
        raise ValueError(f"Expected 3D or 5D tensor, got {tuple(x.shape)}")

    x = x.detach().cpu()
    c = int(x.shape[1])

    if block2repr is None:
        return x.squeeze(0).argmax(dim=0).long()

    # nearest neighbor in embedding space
    repr_table = torch.stack([torch.as_tensor(block2repr[t]).float() for t in tokens], dim=0)  # [N,C]
    if repr_table.shape[1] != c:
        raise ValueError(
            f"Embedding dimension mismatch: model has C={c}, repr table has C={repr_table.shape[1]}. "
            "Use the same representation file as during training."
        )
    vox = x.squeeze(0).permute(1, 2, 3, 0).float()  # [Y,Z,X,C]
    # Compute in chunks to avoid huge memory spikes.
    flat = vox.reshape(-1, c)
    out_ids = []
    chunk = 200000
    for start in range(0, flat.shape[0], chunk):
        part = flat[start:start + chunk]
        d = torch.cdist(part, repr_table)
        out_ids.append(d.argmin(dim=1).cpu())
    ids = torch.cat(out_ids, dim=0).reshape(vox.shape[:3])
    return ids.long()


# ---------------------------------------------------------------------
# Semantic labels, standalone version for original World-GAN
# ---------------------------------------------------------------------


SEM_AIR = 0
SEM_GROUND = 1
SEM_LIQUID = 2
SEM_STRUCTURE = 3
SEM_FOLIAGE = 4
SEM_DECOR = 5

SEM_NAMES = {
    SEM_AIR: "AIR",
    SEM_GROUND: "GROUND",
    SEM_LIQUID: "LIQUID",
    SEM_STRUCTURE: "STRUCTURE",
    SEM_FOLIAGE: "FOLIAGE",
    SEM_DECOR: "DECOR",
}

GROUND_WORDS = [
    "grass_block", "dirt", "grass_path", "path", "farmland", "stone", "andesite",
    "diorite", "granite", "gravel", "sand", "clay", "ore", "netherrack", "podzol",
    "coarse_dirt", "mycelium", "snow_block", "ice",
]
LIQUID_WORDS = ["water", "lava"]
FOLIAGE_WORDS = [
    "leaves", "vine", "grass", "fern", "seagrass", "lily_pad", "poppy", "dandelion",
    "cornflower", "daisy", "orchid", "bluet", "bush", "wheat", "crop", "sapling",
    "cactus", "bamboo", "kelp", "mushroom",
]
STRUCTURE_WORDS = [
    "planks", "log", "wood", "stairs", "slab", "cobblestone", "wall", "terracotta",
    "bricks", "brick", "door", "trapdoor", "fence", "gate", "glass", "pane",
    "bars", "ladder", "hay_block",
]
DECOR_WORDS = [
    "torch", "bed", "chest", "furnace", "smoker", "blast_furnace", "brewing_stand",
    "composter", "bell", "grindstone", "carpet", "pressure_plate", "button", "lever",
    "lantern", "flower_pot",
]


def token_to_semantic_id(token: str) -> int:
    t = token.replace("minecraft:", "").lower()
    if t in {"air", "cave_air", "void_air"}:
        return SEM_AIR
    if any(w in t for w in LIQUID_WORDS):
        return SEM_LIQUID
    # Order matters: grass_block/path should be ground, tall_grass should be foliage.
    if any(w == t or w in t for w in GROUND_WORDS):
        if t not in {"grass", "tall_grass", "seagrass", "tall_seagrass"}:
            return SEM_GROUND
    if any(w in t for w in DECOR_WORDS):
        return SEM_DECOR
    if any(w in t for w in STRUCTURE_WORDS):
        return SEM_STRUCTURE
    if any(w in t for w in FOLIAGE_WORDS):
        return SEM_FOLIAGE
    return SEM_DECOR

# ---------------------------------------------------------------------
# Metric 1: block histogram
# ---------------------------------------------------------------------


def block_counts(index_grid: torch.Tensor, num_tokens: int) -> np.ndarray:
    return torch.bincount(index_grid.reshape(-1).long(), minlength=num_tokens).cpu().numpy().astype(np.int64)

# ---------------------------------------------------------------------
# Metric 3: rare block recall
# ---------------------------------------------------------------------


def rare_block_recall_summary(real_grid, generated_grids, tokens, scale: int, rare_max_count=10, rare_max_freq=0.005):
    num_tokens = len(tokens)
    real_counts = block_counts(real_grid, num_tokens)
    real_total = int(real_counts.sum())
    rare_ids = [
        i for i, c in enumerate(real_counts)
        if c > 0 and (c <= rare_max_count or (c / max(real_total, 1)) <= rare_max_freq)
    ]

    rows = []
    recalls = []
    count_ratios = []
    for sample_id, fake in enumerate(generated_grids):
        fake_counts = block_counts(fake, num_tokens)
        hit = [i for i in rare_ids if fake_counts[i] > 0]
        recall = len(hit) / max(len(rare_ids), 1)
        real_rare_total = int(real_counts[rare_ids].sum()) if rare_ids else 0
        fake_rare_total = int(fake_counts[rare_ids].sum()) if rare_ids else 0
        ratio = fake_rare_total / max(real_rare_total, 1)
        recalls.append(recall)
        count_ratios.append(ratio)
        for i in rare_ids:
            rows.append({
                "scale": scale,
                "sample_id": sample_id,
                "token_id": i,
                "token": tokens[i],
                "real_count": int(real_counts[i]),
                "fake_count": int(fake_counts[i]),
                "appeared_in_fake": int(fake_counts[i] > 0),
            })

    summary = {
        "scale": scale,
        "num_rare_tokens": len(rare_ids),
        "rare_max_count": rare_max_count,
        "rare_max_freq": rare_max_freq,
        "recall_mean": float(np.mean(recalls)) if recalls else 0.0,
        "recall_std": float(np.std(recalls)) if recalls else 0.0,
        "rare_count_ratio_mean": float(np.mean(count_ratios)) if count_ratios else 0.0,
        "rare_count_ratio_std": float(np.std(count_ratios)) if count_ratios else 0.0,
    }
    return rows, summary

# ---------------------------------------------------------------------
# Exact token groups for village house metrics
# ---------------------------------------------------------------------

# Blocks that belong to house-like objects.
# Used for House Fragmentation Index.
HOUSE_COMPONENT_TOKENS = {
    # main structure / walls
    "oak_log",
    "birch_log",
    "stripped_oak_log",
    "oak_planks",
    "cobblestone",
    "cobblestone_wall",
    "white_terracotta",
    "bricks",
    "smooth_stone",
    "iron_bars",

    # roof / shape blocks
    "oak_stairs",
    "cobblestone_stairs",
    "oak_slab",
    "smooth_stone_slab",
    "oak_trapdoor",

    # openings
    "oak_door",
    "glass_pane",
    "yellow_stained_glass_pane",
    "white_stained_glass_pane",
}

TREE_LOG_TOKENS = {
    "oak_log",
    "birch_log",
}

TREE_CONTEXT_TOKENS = {
    "oak_leaves",
    "birch_leaves",
    "vine",
}

# Structural shell blocks.
# We combine walls and roofs because in this village map
# the same blocks can be used for both walls and roofs.
STRUCTURAL_SHELL_TOKENS = {
    "oak_log",
    "birch_log",
    "stripped_oak_wood",
    "stripped_oak_log",
    "oak_planks",
    "cobblestone",
    "cobblestone_wall",
    "white_terracotta",
    "bricks",
    "smooth_stone",
    "iron_bars",

    "oak_stairs",
    "cobblestone_stairs",
    "oak_slab",
    "smooth_stone_slab",
    "oak_trapdoor",
}

WINDOW_TOKENS = {
    "glass_pane",
    "yellow_stained_glass_pane",
    "white_stained_glass_pane",
    "iron_bars",
}


DOOR_TOKENS = {
    "oak_door",
    "oak_trapdoor",
}


GROUND_SUPPORT_TOKENS = {
    "stone",
    "dirt",
    "andesite",
    "diorite",
    "gravel",
    "sand",
    "grass_block",
    "grass_path",
    "clay",
    "farmland",
    "smooth_stone",
    "cobblestone",
    "bricks",
}


def token_in_set(token: str, allowed: set[str]) -> bool:
    return clean_token_name(token) in allowed


def is_house_token(token: str) -> bool:
    return token_in_set(token, HOUSE_COMPONENT_TOKENS)


def is_structural_shell_token(token: str) -> bool:
    return token_in_set(token, STRUCTURAL_SHELL_TOKENS)


# Keep these aliases only if some old code still calls them.
def is_wall_token(token: str) -> bool:
    return is_structural_shell_token(token)


def is_roof_token(token: str) -> bool:
    return is_structural_shell_token(token)

def is_window_token(token: str) -> bool:
    return token_in_set(token, WINDOW_TOKENS)


def is_door_token(token: str) -> bool:
    return token_in_set(token, DOOR_TOKENS)


def is_ground_support_token(token: str) -> bool:
    return token_in_set(token, GROUND_SUPPORT_TOKENS)

def clean_token_name(token: str) -> str:
    return token.replace("minecraft:", "").lower()

def house_token_flags(tokens: Sequence[str]) -> np.ndarray:
    return np.array([is_house_token(t) for t in tokens], dtype=bool)

def token_set_mask_from_grid(
    index_grid: torch.Tensor,
    tokens: Sequence[str],
    allowed_tokens: set[str],
) -> np.ndarray:
    arr = index_grid.cpu().numpy().astype(np.int64)

    if arr.size == 0:
        return np.zeros(arr.shape, dtype=bool)

    max_id = int(arr.max())
    if max_id >= len(tokens):
        raise ValueError(
            f"Token id {max_id} is out of range for token list of length {len(tokens)}. "
            "This usually means decoding produced invalid token ids."
        )

    flags = np.array(
        [clean_token_name(t) in allowed_tokens for t in tokens],
        dtype=bool,
    )

    return flags[arr]


def tree_like_log_mask_from_grid(
    index_grid: torch.Tensor,
    tokens: Sequence[str],
    foliage_radius: int = 2,
) -> np.ndarray:
    """
    Detects log blocks that are likely part of trees, not houses.

    A log component is treated as tree-like if it is close to leaves/vines.
    This solves the ambiguity where oak_log/birch_log can be used both
    in trees and houses.
    """
    if ndimage is None:
        raise ImportError("scipy is required for tree-context filtering: pip install scipy")

    log_mask = token_set_mask_from_grid(index_grid, tokens, TREE_LOG_TOKENS)
    foliage_mask = token_set_mask_from_grid(index_grid, tokens, TREE_CONTEXT_TOKENS)

    if not log_mask.any() or not foliage_mask.any():
        return np.zeros_like(log_mask, dtype=bool)

    # Expand leaves/vines region so nearby trunks are detected.
    dilation_structure = np.ones(
        (
            2 * foliage_radius + 1,
            2 * foliage_radius + 1,
            2 * foliage_radius + 1,
        ),
        dtype=bool,
    )

    near_foliage = ndimage.binary_dilation(
        foliage_mask,
        structure=dilation_structure,
    )

    log_near_foliage = log_mask & near_foliage

    # Remove whole connected log components if any part touches/approaches foliage.
    structure = ndimage.generate_binary_structure(rank=3, connectivity=1)
    comp_grid, num_components = ndimage.label(log_mask, structure=structure)

    tree_log_mask = np.zeros_like(log_mask, dtype=bool)

    for comp_id in range(1, num_components + 1):
        comp = comp_grid == comp_id

        if (comp & log_near_foliage).any():
            tree_log_mask |= comp

    return tree_log_mask


def house_mask_from_grid(index_grid: torch.Tensor, tokens: Sequence[str]) -> np.ndarray:
    """
    Creates the house-like block mask.

    Important:
    - uses exact token names
    - removes tree-like oak/birch logs using foliage/vine context
    """
    base_house_mask = token_set_mask_from_grid(
        index_grid,
        tokens,
        HOUSE_COMPONENT_TOKENS,
    )

    tree_log_mask = tree_like_log_mask_from_grid(
        index_grid,
        tokens,
        foliage_radius=2,
    )

    return base_house_mask & ~tree_log_mask


def _component_bbox_stats(comp_grid: np.ndarray, comp_id: int, size: int) -> dict:
    positions = np.argwhere(comp_grid == comp_id)
    if positions.size == 0:
        return {
            "bbox_y": 0,
            "bbox_z": 0,
            "bbox_x": 0,
            "bbox_volume": 0,
            "bbox_fill_ratio": 0.0,
            "footprint_area": 0,
            "height": 0,
        }

    y0, z0, x0 = positions.min(axis=0)
    y1, z1, x1 = positions.max(axis=0) + 1

    bbox_y = int(y1 - y0)
    bbox_z = int(z1 - z0)
    bbox_x = int(x1 - x0)
    bbox_volume = max(1, bbox_y * bbox_z * bbox_x)

    # Top-down footprint in Z-X plane
    footprint = set((int(p[1]), int(p[2])) for p in positions)
    footprint_area = len(footprint)

    return {
        "bbox_y": bbox_y,
        "bbox_z": bbox_z,
        "bbox_x": bbox_x,
        "bbox_volume": int(bbox_volume),
        "bbox_fill_ratio": float(size / bbox_volume),
        "footprint_area": int(footprint_area),
        "height": bbox_y,
    }


def house_component_coherence_stats(
    index_grid: torch.Tensor,
    tokens: Sequence[str],
    min_house_component_size: int = 20,
    small_house_component_size: int = 5,
) -> Tuple[dict, List[dict]]:
    """
    Computes object-level connected-component metrics for house-like blocks.

    Tensor shape is expected as (Y, Z, X), where Y is vertical height.
    Connectivity is 6-neighborhood, face-connected only.
    """
    if ndimage is None:
        raise ImportError("scipy is required for connected components: pip install scipy")

    house_mask = house_mask_from_grid(index_grid, tokens)
    total_house_voxels = int(house_mask.sum())

    if total_house_voxels == 0:
        stats = {
            "total_house_voxels": 0,
            "num_house_components": 0,
            "meaningful_house_components": 0,
            "largest_house_component_voxels": 0,
            "largest_house_component_ratio": 0.0,
            "small_house_components": 0,
            "small_house_component_ratio": 0.0,
            "small_house_voxels": 0,
            "small_house_voxel_ratio": 0.0,
            "mean_house_component_size": 0.0,
            "median_house_component_size": 0.0,
            "largest_bbox_fill_ratio": 0.0,
            "mean_bbox_fill_ratio": 0.0,
            "largest_component_height": 0,
            "largest_component_footprint_area": 0,
            "house_fragmentation_index": 0.0,
            "house_component_coherence_score": 0.0,
        }
        return stats, []

    # 6-connectivity in 3D, only face-neighbors count as connected.
    structure = ndimage.generate_binary_structure(rank=3, connectivity=1)
    comp_grid, num_components = ndimage.label(house_mask, structure=structure)

    sizes = np.bincount(comp_grid.reshape(-1))[1:]  # skip background 0
    sizes = sizes.astype(np.int64)

    largest_idx = int(np.argmax(sizes)) + 1
    largest_size = int(sizes.max())

    small_mask = sizes < small_house_component_size
    meaningful_mask = sizes >= min_house_component_size

    small_components = int(small_mask.sum())
    meaningful_components = int(meaningful_mask.sum())
    small_voxels = int(sizes[small_mask].sum()) if len(sizes) else 0

    component_rows = []
    bbox_fill_values = []

    for comp_id, size in enumerate(sizes, start=1):
        size = int(size)
        bbox_stats = _component_bbox_stats(comp_grid, comp_id, size)
        bbox_fill_values.append(bbox_stats["bbox_fill_ratio"])

        component_rows.append({
            "component_id": comp_id,
            "component_size": size,
            "is_largest_component": int(comp_id == largest_idx),
            "is_small_component": int(size < small_house_component_size),
            "is_meaningful_component": int(size >= min_house_component_size),
            **bbox_stats,
        })

    largest_bbox = component_rows[largest_idx - 1]

    largest_ratio = largest_size / max(total_house_voxels, 1)
    small_component_ratio = small_components / max(int(num_components), 1)
    small_voxel_ratio = small_voxels / max(total_house_voxels, 1)

    # Lower means fewer house-like voxels are located in tiny broken fragments.
    # This is better for village maps because multiple separate houses are normal.
    fragmentation_index = small_voxels / max(total_house_voxels, 1)

    # Simple bounded score, higher is better.
    # It rewards one dominant component, few tiny fragments, and compact object shape.
    coherence_score = (
        0.45 * largest_ratio
        + 0.35 * (1.0 - small_voxel_ratio)
        + 0.20 * float(largest_bbox["bbox_fill_ratio"])
    )

    stats = {
        "total_house_voxels": total_house_voxels,
        "num_house_components": int(num_components),
        "meaningful_house_components": meaningful_components,
        "largest_house_component_voxels": largest_size,
        "largest_house_component_ratio": float(largest_ratio),
        "small_house_components": small_components,
        "small_house_component_ratio": float(small_component_ratio),
        "small_house_voxels": small_voxels,
        "small_house_voxel_ratio": float(small_voxel_ratio),
        "mean_house_component_size": float(sizes.mean()) if len(sizes) else 0.0,
        "median_house_component_size": float(np.median(sizes)) if len(sizes) else 0.0,
        "largest_bbox_fill_ratio": float(largest_bbox["bbox_fill_ratio"]),
        "mean_bbox_fill_ratio": float(np.mean(bbox_fill_values)) if bbox_fill_values else 0.0,
        "largest_component_height": int(largest_bbox["height"]),
        "largest_component_footprint_area": int(largest_bbox["footprint_area"]),
        "house_fragmentation_index": float(fragmentation_index),
        "house_component_coherence_score": float(coherence_score),
    }

    return stats, component_rows


def house_component_coherence_summary(
    real_grid: torch.Tensor,
    generated_grids: Sequence[torch.Tensor],
    tokens: Sequence[str],
    scale: int,
    min_house_component_size: int = 20,
    small_house_component_size: int = 5,
) -> Tuple[List[dict], List[dict], dict]:
    """
    Returns:
      1) per-sample summary rows
      2) per-component detail rows
      3) aggregate summary row comparing generated mean to real
    """
    sample_rows = []
    detail_rows = []

    real_stats, real_components = house_component_coherence_stats(
        real_grid,
        tokens,
        min_house_component_size=min_house_component_size,
        small_house_component_size=small_house_component_size,
    )

    sample_rows.append({
        "scale": scale,
        "sample_id": "real",
        **real_stats,
    })

    for comp in real_components:
        detail_rows.append({
            "scale": scale,
            "sample_id": "real",
            **comp,
        })

    for sample_id, fake in enumerate(generated_grids):
        fake_stats, fake_components = house_component_coherence_stats(
            fake,
            tokens,
            min_house_component_size=min_house_component_size,
            small_house_component_size=small_house_component_size,
        )

        sample_rows.append({
            "scale": scale,
            "sample_id": sample_id,
            **fake_stats,
        })

        for comp in fake_components:
            detail_rows.append({
                "scale": scale,
                "sample_id": sample_id,
                **comp,
            })

    metric_keys = [
        "total_house_voxels",
        "num_house_components",
        "meaningful_house_components",
        "largest_house_component_ratio",
        "small_house_component_ratio",
        "small_house_voxel_ratio",
        "mean_house_component_size",
        "median_house_component_size",
        "largest_bbox_fill_ratio",
        "mean_bbox_fill_ratio",
        "largest_component_height",
        "largest_component_footprint_area",
        "house_fragmentation_index",
        "house_component_coherence_score",
    ]

    gen_rows = [r for r in sample_rows if r["sample_id"] != "real"]

    aggregate = {
        "scale": scale,
        "num_generated_samples": len(gen_rows),
        "min_house_component_size": min_house_component_size,
        "small_house_component_size": small_house_component_size,
    }

    for key in metric_keys:
        real_val = float(real_stats[key])
        gen_vals = [float(r[key]) for r in gen_rows]

        aggregate[f"real_{key}"] = real_val
        aggregate[f"generated_{key}_mean"] = float(np.mean(gen_vals)) if gen_vals else float("nan")
        aggregate[f"generated_{key}_std"] = float(np.std(gen_vals)) if gen_vals else float("nan")
        aggregate[f"abs_diff_{key}"] = (
            float(abs(np.mean(gen_vals) - real_val)) if gen_vals else float("nan")
        )

    return sample_rows, detail_rows, aggregate

# ---------------------------------------------------------------------
# Metric 4.6: House Completeness
# ---------------------------------------------------------------------

def token_matches_any(token: str, keywords: Sequence[str]) -> bool:
    t = clean_token_name(token)
    return any(k in t for k in keywords)


def is_air_token(token: str) -> bool:
    return clean_token_name(token) in {"air", "cave_air", "void_air"}

def token_flags(tokens: Sequence[str], predicate) -> np.ndarray:
    return np.array([predicate(t) for t in tokens], dtype=bool)


def mask_from_token_predicate(index_grid: torch.Tensor, tokens: Sequence[str], predicate) -> np.ndarray:
    arr = index_grid.cpu().numpy().astype(np.int64)
    flags = token_flags(tokens, predicate)

    if arr.size == 0:
        return np.zeros(arr.shape, dtype=bool)

    max_id = int(arr.max())
    if max_id >= len(tokens):
        raise ValueError(
            f"Token id {max_id} is out of range for token list length {len(tokens)}"
        )

    return flags[arr]


def air_mask_from_grid(index_grid: torch.Tensor, tokens: Sequence[str]) -> np.ndarray:
    return mask_from_token_predicate(index_grid, tokens, is_air_token)


def shell_mask_from_grid(index_grid: torch.Tensor, tokens: Sequence[str]) -> np.ndarray:
    return mask_from_token_predicate(index_grid, tokens, is_structural_shell_token)

def window_mask_from_grid(index_grid: torch.Tensor, tokens: Sequence[str]) -> np.ndarray:
    return mask_from_token_predicate(index_grid, tokens, is_window_token)


def door_mask_from_grid(index_grid: torch.Tensor, tokens: Sequence[str]) -> np.ndarray:
    return mask_from_token_predicate(index_grid, tokens, is_door_token)


def ground_support_mask_from_grid(index_grid: torch.Tensor, tokens: Sequence[str]) -> np.ndarray:
    return mask_from_token_predicate(index_grid, tokens, is_ground_support_token)


def _component_bbox(comp_grid: np.ndarray, comp_id: int):
    pos = np.argwhere(comp_grid == comp_id)
    if pos.size == 0:
        return None
    y0, z0, x0 = pos.min(axis=0)
    y1, z1, x1 = pos.max(axis=0) + 1
    return int(y0), int(y1), int(z0), int(z1), int(x0), int(x1)


def _has_adjacent(mask: np.ndarray, y: int, z: int, x: int) -> bool:
    y_max, z_max, x_max = mask.shape
    neighbors = [
        (y, z - 1, x),
        (y, z + 1, x),
        (y, z, x - 1),
        (y, z, x + 1),
    ]
    for yy, zz, xx in neighbors:
        if 0 <= yy < y_max and 0 <= zz < z_max and 0 <= xx < x_max:
            if mask[yy, zz, xx]:
                return True
    return False


def _has_air_access(air_mask: np.ndarray, y: int, z: int, x: int) -> bool:
    return _has_adjacent(air_mask, y, z, x)


def _ray_hits(mask: np.ndarray, y: int, z: int, x: int, dz: int, dx: int, max_steps: int = 8) -> bool:
    z_cur, x_cur = z, x
    for _ in range(max_steps):
        z_cur += dz
        x_cur += dx
        if z_cur < 0 or z_cur >= mask.shape[1] or x_cur < 0 or x_cur >= mask.shape[2]:
            return False
        if mask[y, z_cur, x_cur]:
            return True
    return False


def _is_enclosed_air_voxel(house_mask: np.ndarray, y: int, z: int, x: int, max_steps: int = 8) -> bool:
    """
    Approximate interior-air test.

    A voxel is considered interior-like if house blocks exist on both opposite
    sides along X and both opposite sides along Z within a limited ray distance.
    """
    hit_x = (
        _ray_hits(house_mask, y, z, x, dz=0, dx=-1, max_steps=max_steps)
        and _ray_hits(house_mask, y, z, x, dz=0, dx=1, max_steps=max_steps)
    )
    hit_z = (
        _ray_hits(house_mask, y, z, x, dz=-1, dx=0, max_steps=max_steps)
        and _ray_hits(house_mask, y, z, x, dz=1, dx=0, max_steps=max_steps)
    )
    return bool(hit_x and hit_z)


def _footprint(mask_3d: np.ndarray) -> np.ndarray:
    """
    Converts a 3D mask [Y, Z, X] to a top-down footprint [Z, X].
    """
    return mask_3d.any(axis=0)


def _perimeter_mask_2d(footprint: np.ndarray) -> np.ndarray:
    """
    Returns perimeter cells of a 2D footprint using 4-neighborhood.
    """
    if footprint.sum() == 0:
        return np.zeros_like(footprint, dtype=bool)

    padded = np.pad(footprint.astype(bool), 1, constant_values=False)
    center = padded[1:-1, 1:-1]

    up = padded[:-2, 1:-1]
    down = padded[2:, 1:-1]
    left = padded[1:-1, :-2]
    right = padded[1:-1, 2:]

    has_outside_neighbor = ~(up & down & left & right)
    return center & has_outside_neighbor


def _shell_consistency_score(component_mask: np.ndarray, shell_mask: np.ndarray) -> Tuple[float, int]:
    """
    Measures how much of the house component is made of structural shell blocks.

    This combines wall-like and roof-like blocks into one feature.
    Higher value means the component has enough structural material.
    """
    shell_in_component = component_mask & shell_mask

    shell_voxels = int(shell_in_component.sum())
    component_voxels = int(component_mask.sum())

    shell_ratio = shell_voxels / max(component_voxels, 1)

    return float(shell_ratio), shell_voxels


def _door_validity(component_mask: np.ndarray, door_mask: np.ndarray, air_mask: np.ndarray, support_mask: np.ndarray) -> dict:
    door_positions = np.argwhere(component_mask & door_mask)

    if door_positions.size == 0:
        return {
            "door_exists": 0,
            "valid_door_exists": 0,
            "door_count": 0,
            "valid_door_count": 0,
            "door_validity_ratio": 0.0,
        }

    valid = 0
    for y, z, x in door_positions:
        y = int(y)
        z = int(z)
        x = int(x)

        grounded = False
        if y == 0:
            grounded = True
        else:
            grounded = bool(support_mask[y - 1, z, x] or component_mask[y - 1, z, x])

        air_access = _has_air_access(air_mask, y, z, x)

        if grounded and air_access:
            valid += 1

    total = int(len(door_positions))
    return {
        "door_exists": 1,
        "valid_door_exists": int(valid > 0),
        "door_count": total,
        "valid_door_count": int(valid),
        "door_validity_ratio": float(valid / max(total, 1)),
    }


def _window_validity(component_mask: np.ndarray, window_mask: np.ndarray, wall_mask: np.ndarray) -> dict:
    window_positions = np.argwhere(component_mask & window_mask)

    if window_positions.size == 0:
        return {
            "window_exists": 0,
            "valid_window_exists": 0,
            "window_count": 0,
            "valid_window_count": 0,
            "window_validity_ratio": 0.0,
        }

    valid = 0
    for y, z, x in window_positions:
        y = int(y)
        z = int(z)
        x = int(x)

        # Window should be attached to wall-like material.
        attached_to_wall = _has_adjacent(wall_mask & component_mask, y, z, x)

        if attached_to_wall:
            valid += 1

    total = int(len(window_positions))
    return {
        "window_exists": 1,
        "valid_window_exists": int(valid > 0),
        "window_count": total,
        "valid_window_count": int(valid),
        "window_validity_ratio": float(valid / max(total, 1)),
    }


def _interior_air_score(component_mask: np.ndarray, air_mask: np.ndarray, max_ray_steps: int = 8) -> dict:
    bbox = np.argwhere(component_mask)
    if bbox.size == 0:
        return {
            "interior_air_exists": 0,
            "interior_air_count": 0,
            "interior_air_ratio": 0.0,
        }

    y0, z0, x0 = bbox.min(axis=0)
    y1, z1, x1 = bbox.max(axis=0) + 1

    # Need at least small volume to have an interior.
    if (y1 - y0) < 3 or (z1 - z0) < 3 or (x1 - x0) < 3:
        return {
            "interior_air_exists": 0,
            "interior_air_count": 0,
            "interior_air_ratio": 0.0,
        }

    inner_air_positions = np.argwhere(
        air_mask[y0 + 1:y1 - 1, z0 + 1:z1 - 1, x0 + 1:x1 - 1]
    )

    if inner_air_positions.size == 0:
        return {
            "interior_air_exists": 0,
            "interior_air_count": 0,
            "interior_air_ratio": 0.0,
        }

    enclosed = 0
    total_inner_air = int(len(inner_air_positions))

    for yy, zz, xx in inner_air_positions:
        y = int(yy + y0 + 1)
        z = int(zz + z0 + 1)
        x = int(xx + x0 + 1)

        if _is_enclosed_air_voxel(component_mask, y, z, x, max_steps=max_ray_steps):
            enclosed += 1

    ratio = enclosed / max(total_inner_air, 1)

    return {
        "interior_air_exists": int(enclosed > 0),
        "interior_air_count": int(enclosed),
        "interior_air_ratio": float(ratio),
    }


def house_completeness_component_stats(
    index_grid: torch.Tensor,
    tokens: Sequence[str],
    min_house_component_size: int = 20,
    shell_consistency_threshold: float = 0.30,
) -> Tuple[List[dict], dict]:
    """
    Computes door/window/roof/interior/wall completeness for each meaningful house component.

    Shape convention:
      index_grid: [Y, Z, X]
    """
    if ndimage is None:
        raise ImportError("scipy is required for connected components: pip install scipy")

    house_mask = house_mask_from_grid(index_grid, tokens)
    air_mask = air_mask_from_grid(index_grid, tokens)
    shell_mask = shell_mask_from_grid(index_grid, tokens)
    window_mask = window_mask_from_grid(index_grid, tokens)
    door_mask = door_mask_from_grid(index_grid, tokens)

    support_mask = (
            ground_support_mask_from_grid(index_grid, tokens)
            | house_mask
            | shell_mask
    )

    structure = ndimage.generate_binary_structure(rank=3, connectivity=1)
    comp_grid, num_components = ndimage.label(house_mask, structure=structure)
    sizes = np.bincount(comp_grid.reshape(-1))[1:].astype(np.int64)

    rows = []

    for comp_id, size in enumerate(sizes, start=1):
        size = int(size)
        if size < min_house_component_size:
            continue

        component_mask = comp_grid == comp_id

        shell_score, shell_voxels = _shell_consistency_score(component_mask, shell_mask)
        shell_consistent = int(shell_score >= shell_consistency_threshold)

        door_stats = _door_validity(component_mask, door_mask, air_mask, support_mask)
        window_stats = _window_validity(component_mask, window_mask, shell_mask)
        interior_stats = _interior_air_score(component_mask, air_mask)

        # Simple interpretable score.
        # Keep both raw feature flags and score in CSV.
        feature_flags = [
            shell_consistent,
            door_stats["valid_door_exists"],
            window_stats["valid_window_exists"],
            interior_stats["interior_air_exists"],
        ]

        completeness_score = float(np.mean(feature_flags))

        bbox_stats = _component_bbox_stats(comp_grid, comp_id, size)

        rows.append({
            "component_id": comp_id,
            "component_size": size,
            **bbox_stats,

            "shell_consistency_score": float(shell_score),
            "shell_consistent": shell_consistent,
            "shell_voxels": int(shell_voxels),

            **door_stats,
            **window_stats,
            **interior_stats,

            "house_completeness_score": completeness_score,
            "num_completeness_features_present": int(sum(feature_flags)),
            "num_completeness_features_total": len(feature_flags),
        })

    if not rows:
        summary = {
            "num_meaningful_house_components": 0,
            "mean_house_completeness_score": 0.0,
            "max_house_completeness_score": 0.0,
            "mean_shell_consistency_score": 0.0,
            "shell_consistent_rate": 0.0,
            "valid_door_exists_rate": 0.0,
            "valid_window_exists_rate": 0.0,
            "interior_air_exists_rate": 0.0,
        }
        return rows, summary

    summary = {
        "num_meaningful_house_components": len(rows),
        "mean_house_completeness_score": float(np.mean([r["house_completeness_score"] for r in rows])),
        "max_house_completeness_score": float(np.max([r["house_completeness_score"] for r in rows])),
        "mean_shell_consistency_score": float(np.mean([r["shell_consistency_score"] for r in rows])),
        "shell_consistent_rate": float(np.mean([r["shell_consistent"] for r in rows])),
        "valid_door_exists_rate": float(np.mean([r["valid_door_exists"] for r in rows])),
        "valid_window_exists_rate": float(np.mean([r["valid_window_exists"] for r in rows])),
        "interior_air_exists_rate": float(np.mean([r["interior_air_exists"] for r in rows])),
    }

    return rows, summary


def house_completeness_summary(
    real_grid: torch.Tensor,
    generated_grids: Sequence[torch.Tensor],
    tokens: Sequence[str],
    scale: int,
    min_house_component_size: int = 20,
    shell_consistency_threshold: float = 0.30,
) -> Tuple[List[dict], List[dict], dict]:
    """
    Returns:
      1) per-sample summary rows
      2) per-component detail rows
      3) aggregate comparison row
    """
    sample_rows = []
    detail_rows = []

    real_components, real_summary = house_completeness_component_stats(
        real_grid,
        tokens,
        min_house_component_size=min_house_component_size,
        shell_consistency_threshold=shell_consistency_threshold
    )

    sample_rows.append({
        "scale": scale,
        "sample_id": "real",
        **real_summary,
    })

    for comp in real_components:
        detail_rows.append({
            "scale": scale,
            "sample_id": "real",
            **comp,
        })

    generated_summaries = []

    for sample_id, fake in enumerate(generated_grids):
        fake_components, fake_summary = house_completeness_component_stats(
            fake,
            tokens,
            min_house_component_size=min_house_component_size,
            shell_consistency_threshold=shell_consistency_threshold,
        )

        generated_summaries.append(fake_summary)

        sample_rows.append({
            "scale": scale,
            "sample_id": sample_id,
            **fake_summary,
        })

        for comp in fake_components:
            detail_rows.append({
                "scale": scale,
                "sample_id": sample_id,
                **comp,
            })

    metric_keys = [
        "num_meaningful_house_components",
        "mean_house_completeness_score",
        "max_house_completeness_score",
        "mean_shell_consistency_score",
        "shell_consistent_rate",
        "valid_door_exists_rate",
        "valid_window_exists_rate",
        "interior_air_exists_rate",
    ]

    aggregate = {
        "scale": scale,
        "num_generated_samples": len(generated_summaries),
        "min_house_component_size": min_house_component_size,
        "shell_consistency_threshold": shell_consistency_threshold
    }

    for key in metric_keys:
        real_val = float(real_summary[key])
        gen_vals = [float(s[key]) for s in generated_summaries]

        aggregate[f"real_{key}"] = real_val
        aggregate[f"generated_{key}_mean"] = float(np.mean(gen_vals)) if gen_vals else float("nan")
        aggregate[f"generated_{key}_std"] = float(np.std(gen_vals)) if gen_vals else float("nan")
        aggregate[f"abs_diff_{key}"] = (
            float(abs(np.mean(gen_vals) - real_val)) if gen_vals else float("nan")
        )

    return sample_rows, detail_rows, aggregate

# ---------------------------------------------------------------------
# PyVista debug visualization for house metrics, only final scale
# ---------------------------------------------------------------------

def save_pyvista_orbit_gif(
    plotter,
    out_path: Path,
    n_frames: int = 120,
    fps: int = 20,
):
    """
    Saves rotating orbit animation as GIF.

    The plotter must already contain all meshes.
    """
    out_path = Path(out_path)
    ensure_dir(out_path.parent)

    plotter.set_background("white")
    plotter.add_axes()
    plotter.show_grid()
    plotter.view_isometric()

    # Important for off-screen rendering
    plotter.show(auto_close=False, interactive_update=False)

    plotter.open_gif(str(out_path), fps=fps)

    for _ in range(n_frames):
        plotter.camera.Azimuth(360.0 / n_frames)
        plotter.write_frame()

    plotter.close()

def _sample_coords(coords: np.ndarray, max_points: int) -> np.ndarray:
    if len(coords) <= max_points:
        return coords
    rng = np.random.default_rng(0)
    idx = rng.choice(len(coords), size=max_points, replace=False)
    return coords[idx]


def mask_to_voxel_points(mask: np.ndarray, max_points: int = 30000) -> np.ndarray:
    """
    mask shape: [Y, Z, X]
    returns points: [N, 3] in PyVista coordinates [X, Y, Z]
    """
    coords = np.argwhere(mask)

    if coords.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)

    coords = _sample_coords(coords, max_points=max_points)

    # Convert [Y, Z, X] -> [X, Y, Z]
    xyz = np.stack(
        [coords[:, 2], coords[:, 0], coords[:, 1]],
        axis=1,
    ).astype(np.float32)

    return xyz


def add_voxel_mask_to_plotter(
    plotter,
    mask: np.ndarray,
    color: str,
    label: str,
    opacity: float = 1.0,
    max_points: int = 30000,
    show_edges: bool = True,
):
    import pyvista as pv

    points = mask_to_voxel_points(mask, max_points=max_points)

    if points.shape[0] == 0:
        return None

    pdata = pv.PolyData(points)

    cube = pv.Cube(
        center=(0, 0, 0),
        x_length=1,
        y_length=1,
        z_length=1,
    )

    glyphs = pdata.glyph(scale=False, geom=cube)

    plotter.add_mesh(
        glyphs,
        color=color,
        opacity=opacity,
        show_edges=show_edges,
        edge_color="black",
        line_width=0.25,
        name=label,
    )

    return [label, color]


def save_pyvista_plot_or_animation(
    plotter,
    out_path: Path,
    make_gif: bool = False,
    n_frames: int = 120,
    fps: int = 20,
):
    out_path = Path(out_path)
    ensure_dir(out_path.parent)

    plotter.set_background("white")
    plotter.add_axes()
    plotter.show_grid()
    plotter.view_isometric()

    if make_gif:
        save_pyvista_orbit_gif(
            plotter,
            out_path,
            n_frames=n_frames,
            fps=fps,
        )
    else:
        plotter.screenshot(str(out_path))
        plotter.close()


def visualize_house_components_pyvista(
    index_grid: torch.Tensor,
    tokens: Sequence[str],
    out_path: Path,
    title: str = "House-like blocks",
    min_house_component_size: int = 20,
    small_house_component_size: int = 5,
    max_points: int = 30000,
    window_size: tuple[int, int] = (1400, 1000),
    make_gif: bool = False,
    n_frames: int = 120,
    fps: int = 20,
) -> None:
    """
    Visualizes all blocks used for the House Fragmentation Index.

    This version does not split houses into largest/other/small components.
    It simply shows all house-like blocks with one color.
    """
    import pyvista as pv

    house_mask = house_mask_from_grid(index_grid, tokens)

    plotter = pv.Plotter(off_screen=True, window_size=window_size)
    legend_entries = []

    item = add_voxel_mask_to_plotter(
        plotter,
        house_mask,
        color="#8B4513",
        label="house-like blocks",
        opacity=1.0,
        max_points=max_points,
    )
    if item:
        legend_entries.append(item)

    if legend_entries:
        plotter.add_legend(legend_entries, bcolor="white", border=True)

    plotter.add_text(title, font_size=12, color="black")

    save_pyvista_plot_or_animation(
        plotter,
        out_path,
        make_gif=make_gif,
        n_frames=n_frames,
        fps=fps,
    )

def interior_air_mask_from_grid(
    index_grid: torch.Tensor,
    tokens: Sequence[str],
    min_house_component_size: int = 20,
    max_ray_steps: int = 8,
) -> np.ndarray:
    """
    Creates a boolean mask for interior air used in House Completeness debug visualization.
    """
    if ndimage is None:
        raise ImportError("scipy is required for connected components: pip install scipy")

    house_mask = house_mask_from_grid(index_grid, tokens)
    air_mask = air_mask_from_grid(index_grid, tokens)

    structure = ndimage.generate_binary_structure(rank=3, connectivity=1)
    comp_grid, num_components = ndimage.label(house_mask, structure=structure)

    sizes = np.bincount(comp_grid.reshape(-1))[1:].astype(np.int64)
    interior_mask = np.zeros_like(house_mask, dtype=bool)

    for comp_id, size in enumerate(sizes, start=1):
        size = int(size)

        if size < min_house_component_size:
            continue

        component_mask = comp_grid == comp_id
        bbox = np.argwhere(component_mask)

        if bbox.size == 0:
            continue

        y0, z0, x0 = bbox.min(axis=0)
        y1, z1, x1 = bbox.max(axis=0) + 1

        if (y1 - y0) < 3 or (z1 - z0) < 3 or (x1 - x0) < 3:
            continue

        inner_air_positions = np.argwhere(
            air_mask[y0 + 1:y1 - 1, z0 + 1:z1 - 1, x0 + 1:x1 - 1]
        )

        for yy, zz, xx in inner_air_positions:
            y = int(yy + y0 + 1)
            z = int(zz + z0 + 1)
            x = int(xx + x0 + 1)

            if _is_enclosed_air_voxel(component_mask, y, z, x, max_steps=max_ray_steps):
                interior_mask[y, z, x] = True

    return interior_mask


def visualize_house_completeness_features_pyvista(
    index_grid: torch.Tensor,
    tokens: Sequence[str],
    out_path: Path,
    title: str = "House completeness features",
    min_house_component_size: int = 20,
    max_points: int = 30000,
    window_size: tuple[int, int] = (1400, 1000),
    make_gif: bool = False,
    n_frames: int = 120,
    fps: int = 20,
) -> None:
    """
    Visualizes exact feature masks used for House Completeness Score.

    Colors:
      brown  = wall blocks
      orange = roof blocks
      cyan   = window blocks
      purple = door blocks
      yellow = detected interior air
    """
    import pyvista as pv

    house_mask = house_mask_from_grid(index_grid, tokens)

    shell_mask = shell_mask_from_grid(index_grid, tokens) & house_mask
    window_mask = window_mask_from_grid(index_grid, tokens) & house_mask
    door_mask = door_mask_from_grid(index_grid, tokens) & house_mask

    interior_mask = interior_air_mask_from_grid(
        index_grid,
        tokens,
        min_house_component_size=min_house_component_size,
    )

    plotter = pv.Plotter(off_screen=True, window_size=window_size)
    legend_entries = []

    item = add_voxel_mask_to_plotter(
        plotter,
        shell_mask,
        color="#8B4513",
        label="structural shell",
        opacity=1.0,
        max_points=max_points,
    )
    if item:
        legend_entries.append(item)

    item = add_voxel_mask_to_plotter(
        plotter,
        window_mask,
        color="#00BFFF",
        label="windows",
        opacity=1.0,
        max_points=max_points,
    )
    if item:
        legend_entries.append(item)

    item = add_voxel_mask_to_plotter(
        plotter,
        door_mask,
        color="#6A3D9A",
        label="doors",
        opacity=1.0,
        max_points=max_points,
    )
    if item:
        legend_entries.append(item)

    item = add_voxel_mask_to_plotter(
        plotter,
        interior_mask,
        color="#FFD700",
        label="interior air",
        opacity=0.45,
        max_points=max_points,
    )
    if item:
        legend_entries.append(item)

    if legend_entries:
        plotter.add_legend(legend_entries, bcolor="white", border=True)

    plotter.add_text(title, font_size=12, color="black")
    save_pyvista_plot_or_animation(
        plotter,
        out_path,
        make_gif=make_gif,
        n_frames=n_frames,
        fps=fps,
    )

# ---------------------------------------------------------------------
# Metric 5: TPKL-Div
# ---------------------------------------------------------------------


def pattern_counts_3d(level: np.ndarray, pattern_size: int) -> Counter:
    if any(s < pattern_size for s in level.shape):
        return Counter()
    arr = np.asarray(level, dtype=np.int16)
    counts = Counter()
    y_max = arr.shape[0] - pattern_size + 1
    z_max = arr.shape[1] - pattern_size + 1
    x_max = arr.shape[2] - pattern_size + 1
    for y in range(y_max):
        for z in range(z_max):
            for x in range(x_max):
                patch = arr[y:y+pattern_size, z:z+pattern_size, x:x+pattern_size]
                counts[patch.tobytes()] += 1
    return counts


def kl_from_counts(p_counts: Counter, q_counts: Counter, eps=1e-8) -> float:
    keys = set(p_counts.keys()) | set(q_counts.keys())
    if not keys:
        return float("nan")
    p_total = sum(p_counts.values())
    q_total = sum(q_counts.values())
    k = len(keys)
    out = 0.0
    for key in keys:
        p = (p_counts.get(key, 0) + eps) / (p_total + eps * k)
        q = (q_counts.get(key, 0) + eps) / (q_total + eps * k)
        out += p * math.log(p / q)
    return float(out)


def symmetric_tpkldiv(real_grid, fake_grid, pattern_size: int, weight=0.5, eps=1e-8) -> float:
    p = pattern_counts_3d(real_grid.cpu().numpy(), pattern_size)
    q = pattern_counts_3d(fake_grid.cpu().numpy(), pattern_size)
    if not p or not q:
        return float("nan")
    d_pq = kl_from_counts(p, q, eps)
    d_qp = kl_from_counts(q, p, eps)
    return float((1.0 - weight) * d_pq + weight * d_qp)


def tpkldiv_summary(real_grid, generated_grids, scale: int, pattern_sizes=(5, 10), weight=0.5):
    rows = []
    summary = []
    for k in pattern_sizes:
        vals = []
        for sample_id, fake in enumerate(generated_grids):
            v = symmetric_tpkldiv(real_grid, fake, k, weight=weight)
            rows.append({"scale": scale, "sample_id": sample_id, "pattern_size": k, "tpkldiv": v})
            if not math.isnan(v):
                vals.append(v)
        summary.append({
            "scale": scale,
            "pattern_size": k,
            "mean": float(np.mean(vals)) if vals else float("nan"),
            "std": float(np.std(vals)) if vals else float("nan"),
            "num_valid_samples": len(vals),
        })
    return rows, summary

def average_tpkldiv_over_pattern_sizes(tpkldiv_summary_rows: List[dict]) -> List[dict]:
    """
    Computes one average TPKL-Div value per scale across pattern sizes.
    Example: average over pattern sizes 5 and 10.
    """
    by_scale = defaultdict(list)

    for row in tpkldiv_summary_rows:
        by_scale[row["scale"]].append(row)

    avg_rows = []

    for scale, rows in sorted(by_scale.items()):
        values = [
            r["mean"]
            for r in rows
            if "mean" in r and not math.isnan(r["mean"])
        ]

        avg_rows.append({
            "scale": scale,
            "pattern_sizes": ",".join(str(r["pattern_size"]) for r in rows),
            "avg_tpkldiv": float(np.mean(values)) if values else float("nan"),
            "std_tpkldiv_across_pattern_sizes": float(np.std(values)) if values else float("nan"),
            "num_pattern_sizes": len(values),
        })

    return avg_rows

# ---------------------------------------------------------------------
# Metric 6: Levenshtein diversity
# ---------------------------------------------------------------------


def grid_to_levenshtein_string(grid: torch.Tensor) -> str:
    arr = grid.cpu().numpy().astype(np.int64).reshape(-1)
    if arr.size == 0:
        return ""
    max_id = int(arr.max())
    if max_id > 60000:
        raise ValueError("Too many token ids for Unicode encoding")
    return "".join(chr(int(v) + 1) for v in arr)


def levenshtein_diversity_summary(
    generated_grids: Sequence[torch.Tensor],
    scale: int,
) -> Tuple[List[dict], dict]:
    if levenshtein_distance is None:
        raise ImportError("python-Levenshtein is required: pip install python-Levenshtein")

    strings = [grid_to_levenshtein_string(g) for g in generated_grids]
    rows = []
    raw_vals = []
    norm_vals = []

    for i, j in combinations(range(len(strings)), 2):
        d = levenshtein_distance(strings[i], strings[j])
        d_norm = d / max(len(strings[i]), len(strings[j]), 1)

        raw_vals.append(d)
        norm_vals.append(d_norm)

        rows.append({
            "scale": scale,
            "sample_i": i,
            "sample_j": j,
            "levenshtein_raw": int(d),
            "levenshtein_normalized": float(d_norm),
        })

    summary = {
        "scale": scale,
        "num_samples": len(generated_grids),
        "num_pairs": len(raw_vals),

        # Use this for comparison with Wor(l)d-GAN paper
        "mean_raw": float(np.mean(raw_vals)) if raw_vals else 0.0,
        "std_raw": float(np.std(raw_vals)) if raw_vals else 0.0,
        "min_raw": float(np.min(raw_vals)) if raw_vals else 0.0,
        "max_raw": float(np.max(raw_vals)) if raw_vals else 0.0,

        # Use this when comparing maps with different sizes
        "mean_normalized": float(np.mean(norm_vals)) if norm_vals else 0.0,
        "std_normalized": float(np.std(norm_vals)) if norm_vals else 0.0,
        "min_normalized": float(np.min(norm_vals)) if norm_vals else 0.0,
        "max_normalized": float(np.max(norm_vals)) if norm_vals else 0.0,
    }
    return rows, summary

# ---------------------------------------------------------------------
# Metric 7: Token entropy
# ---------------------------------------------------------------------

def pairwise_token_entropy(index_grid: torch.Tensor) -> dict:
    """
    Computes pairwise token entropy in the three cardinal directions.
    Tensor shape is expected as (Y, Z, X).
    """
    arr = index_grid.cpu().numpy().astype(np.int64)

    axis_info = [
        (0, "entropy_y"),
        (1, "entropy_z"),
        (2, "entropy_x"),
    ]

    out = {}
    vals = []

    for axis, name in axis_info:
        if arr.shape[axis] < 2:
            out[name] = float("nan")
            continue

        a = np.take(arr, indices=range(arr.shape[axis] - 1), axis=axis)
        b = np.take(arr, indices=range(1, arr.shape[axis]), axis=axis)

        pairs = np.stack([a.reshape(-1), b.reshape(-1)], axis=1)
        _, counts = np.unique(pairs, axis=0, return_counts=True)

        probs = counts.astype(np.float64) / max(counts.sum(), 1)
        entropy = float(-(probs * np.log(probs)).sum())

        out[name] = entropy
        vals.append(entropy)

    out["entropy_mean"] = float(np.mean(vals)) if vals else float("nan")
    return out


def token_entropy_summary(
    real_grid: torch.Tensor,
    generated_grids: Sequence[torch.Tensor],
    scale: int,
) -> Tuple[List[dict], dict]:
    rows = []

    real_entropy = pairwise_token_entropy(real_grid)
    rows.append({
        "scale": scale,
        "sample_id": "real",
        **real_entropy,
    })

    generated_means = []

    for sample_id, fake in enumerate(generated_grids):
        fake_entropy = pairwise_token_entropy(fake)
        generated_means.append(fake_entropy["entropy_mean"])
        rows.append({
            "scale": scale,
            "sample_id": sample_id,
            **fake_entropy,
        })

    valid = [v for v in generated_means if not math.isnan(v)]

    summary = {
        "scale": scale,
        "real_entropy_mean": real_entropy["entropy_mean"],
        "generated_entropy_mean": float(np.mean(valid)) if valid else float("nan"),
        "generated_entropy_std": float(np.std(valid)) if valid else float("nan"),
        "abs_diff_from_real": float(abs(np.mean(valid) - real_entropy["entropy_mean"])) if valid else float("nan"),
        "num_valid_samples": len(valid),
    }

    return rows, summary

# ---------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------


def prepare_data(args):
    device = torch.device(args.device)
    run_dir = Path(args.run_dir).expanduser().resolve() if args.run_dir else None
    samples_dir = Path(args.samples_dir).expanduser().resolve() if args.samples_dir else None

    tokens = load_token_list(run_dir, samples_dir)

    if args.generate_per_scale:
        if run_dir is None:
            raise ValueError("--run_dir is required when --generate_per_scale True")
        generators, noise_maps, reals, noise_amplitudes = load_old_pyramid(run_dir, device)
        num_layer = load_num_layer(run_dir, args.num_layer)
        channel_dim = int(reals[0].shape[1])
        block2repr = load_block2repr(args, tokens, channel_dim)
        real_by_scale = [decode_tensor_to_indices(r.detach().cpu(), tokens, block2repr) for r in reals]
        fake_cont = generate_old_per_scale(
            generators=generators,
            noise_maps=noise_maps,
            reals=reals,
            noise_amplitudes=noise_amplitudes,
            num_samples=args.num_samples,
            num_layer=num_layer,
            device=device,
            scale_v=args.scale_v,
            scale_h=args.scale_h,
            scale_d=args.scale_d,
            gen_start_scale=args.gen_start_scale,
        )
        fake_by_scale = defaultdict(list)
        for scale_id, samples in fake_cont.items():
            for x in samples:
                fake_by_scale[scale_id].append(decode_tensor_to_indices(x, tokens, block2repr))
        return real_by_scale, fake_by_scale, tokens, {"num_layer": num_layer, "channel_dim": channel_dim}

    else:
        if samples_dir is None:
            raise ValueError("--samples_dir is required when --generate_per_scale False")
        real_final, fake_final = load_existing_final_samples(samples_dir, args.num_samples)
        final_scale = args.final_scale_id
        real_by_scale = [None] * (final_scale + 1)
        real_by_scale[final_scale] = real_final
        fake_by_scale = defaultdict(list)
        fake_by_scale[final_scale] = fake_final
        return real_by_scale, fake_by_scale, tokens, {"num_layer": None, "channel_dim": None}


def run_diagnostics(args):
    out_dir = Path(args.out_dir).expanduser().resolve()
    ensure_dir(out_dir)

    real_by_scale, fake_by_scale, tokens, extra = prepare_data(args)

    save_json(out_dir / "diagnostics_config.json", {
        "run_dir": str(Path(args.run_dir).expanduser().resolve()) if args.run_dir else None,
        "samples_dir": str(Path(args.samples_dir).expanduser().resolve()) if args.samples_dir else None,
        "out_dir": str(out_dir),
        "num_tokens": len(tokens),
        "tokens": tokens,
        "num_real_scales": len(real_by_scale),
        "fake_scales": sorted(list(fake_by_scale.keys())),
        "num_samples": args.num_samples,
        "generate_per_scale": args.generate_per_scale,
        "pattern_sizes": args.pattern_sizes,
        "repr_type": args.repr_type,
        "repr_pkl": args.repr_pkl,
        **extra,
    })

    compact_summary_rows = []

    tpk_summary_rows = []
    tpk_average_rows = []

    lev_summary_rows = []
    entropy_summary_rows = []
    rare_summary_rows = []

    house_fragmentation_rows = []
    house_completeness_score_rows = []

    for scale_id, real_grid in enumerate(real_by_scale):
        if real_grid is None:
            continue

        generated = fake_by_scale.get(scale_id, [])
        if not generated:
            continue

        print(f"[scale {scale_id}] real={tuple(real_grid.shape)} generated={len(generated)}")

        # 1) TPKL-Div 5, 10
        _, tpk_summary = tpkldiv_summary(
            real_grid,
            generated,
            scale_id,
            pattern_sizes=args.pattern_sizes,
            weight=args.tpkldiv_weight,
        )

        tpk_avg = average_tpkldiv_over_pattern_sizes(tpk_summary)

        # 2) Levenshtein normalized
        _, lev_summary = levenshtein_diversity_summary(generated, scale_id)

        # 3) Entropy abs diff from real
        _, entropy_summary = token_entropy_summary(real_grid, generated, scale_id)

        # 4) Rare Block Recall + Rare Count Ratio
        _, rare_summary = rare_block_recall_summary(
            real_grid,
            generated,
            tokens,
            scale_id,
            rare_max_count=args.rare_max_count,
            rare_max_freq=args.rare_max_freq,
        )

        # 5) House Fragmentation Index
        _, _, house_fragmentation_summary = house_component_coherence_summary(
            real_grid,
            generated,
            tokens,
            scale_id,
            min_house_component_size=args.min_house_component_size,
            small_house_component_size=args.small_house_component_size,
        )

        # 6) House Completeness Score
        _, _, house_completeness_summary_row = house_completeness_summary(
            real_grid,
            generated,
            tokens,
            scale_id,
            min_house_component_size=args.min_house_component_size,
            shell_consistency_threshold=args.shell_consistency_threshold,
        )

        tpk_by_size = {row["pattern_size"]: row for row in tpk_summary}
        avg_tpk_value = tpk_avg[0]["avg_tpkldiv"] if tpk_avg else float("nan")

        # Compact final row for thesis table
        compact_summary_rows.append({
            "scale": scale_id,

            "tpkldiv_5_mean": tpk_by_size.get(5, {}).get("mean", float("nan")),
            "tpkldiv_10_mean": tpk_by_size.get(10, {}).get("mean", float("nan")),
            "tpkldiv_avg_5_10": avg_tpk_value,

            "levenshtein_normalized_mean": lev_summary["mean_normalized"],
            "levenshtein_normalized_std": lev_summary["std_normalized"],

            "entropy_abs_diff_from_real": entropy_summary["abs_diff_from_real"],
            "real_entropy_mean": entropy_summary["real_entropy_mean"],
            "generated_entropy_mean": entropy_summary["generated_entropy_mean"],

            "rare_block_recall_mean": rare_summary["recall_mean"],
            "rare_block_recall_std": rare_summary["recall_std"],
            "rare_count_ratio_mean": rare_summary["rare_count_ratio_mean"],
            "rare_count_ratio_std": rare_summary["rare_count_ratio_std"],

            "real_house_fragmentation_index": house_fragmentation_summary["real_house_fragmentation_index"],
            "generated_house_fragmentation_index_mean": house_fragmentation_summary[
                "generated_house_fragmentation_index_mean"],
            "generated_house_fragmentation_index_std": house_fragmentation_summary[
                "generated_house_fragmentation_index_std"],
            "abs_diff_house_fragmentation_index": house_fragmentation_summary["abs_diff_house_fragmentation_index"],

            "real_house_completeness_score": house_completeness_summary_row["real_mean_house_completeness_score"],
            "generated_house_completeness_score_mean": house_completeness_summary_row[
                "generated_mean_house_completeness_score_mean"],
            "generated_house_completeness_score_std": house_completeness_summary_row[
                "generated_mean_house_completeness_score_std"],
            "abs_diff_house_completeness_score": house_completeness_summary_row[
                "abs_diff_mean_house_completeness_score"],
        })

        tpk_summary_rows.extend(tpk_summary)
        tpk_average_rows.extend(tpk_avg)

        lev_summary_rows.append(lev_summary)
        entropy_summary_rows.append(entropy_summary)
        rare_summary_rows.append(rare_summary)

        house_fragmentation_rows.append({
            "scale": scale_id,
            "real_house_fragmentation_index": house_fragmentation_summary["real_house_fragmentation_index"],
            "generated_house_fragmentation_index_mean": house_fragmentation_summary[
                "generated_house_fragmentation_index_mean"],
            "generated_house_fragmentation_index_std": house_fragmentation_summary[
                "generated_house_fragmentation_index_std"],
            "abs_diff_house_fragmentation_index": house_fragmentation_summary["abs_diff_house_fragmentation_index"],
        })

        house_completeness_score_rows.append({
            "scale": scale_id,
            "real_house_completeness_score": house_completeness_summary_row["real_mean_house_completeness_score"],
            "generated_house_completeness_score_mean": house_completeness_summary_row[
                "generated_mean_house_completeness_score_mean"],
            "generated_house_completeness_score_std": house_completeness_summary_row[
                "generated_mean_house_completeness_score_std"],
            "abs_diff_house_completeness_score": house_completeness_summary_row[
                "abs_diff_mean_house_completeness_score"],
        })

        if args.save_debug_visuals and scale_id == args.debug_scale_id:
            debug_dir = out_dir / "debug_visuals" / f"scale_{scale_id}"
            ensure_dir(debug_dir)

            debug_sample_id = min(args.debug_sample_id, len(generated) - 1)

            ext = "gif" if args.save_debug_gif else "png"

            visualize_house_components_pyvista(
                real_grid,
                tokens,
                debug_dir / f"real_house_components_pyvista.{ext}",
                title=f"Real scale {scale_id}: house components",
                min_house_component_size=args.min_house_component_size,
                small_house_component_size=args.small_house_component_size,
                max_points=args.debug_max_points,
                make_gif=args.save_debug_gif,
                n_frames=args.debug_gif_frames,
                fps=args.debug_gif_fps,
            )

            visualize_house_completeness_features_pyvista(
                real_grid,
                tokens,
                debug_dir / f"real_house_completeness_features_pyvista.{ext}",
                title=f"Real scale {scale_id}: completeness features",
                min_house_component_size=args.min_house_component_size,
                max_points=args.debug_max_points,
                make_gif=args.save_debug_gif,
                n_frames=args.debug_gif_frames,
                fps=args.debug_gif_fps,
            )

            visualize_house_components_pyvista(
                generated[debug_sample_id],
                tokens,
                debug_dir / f"generated_{debug_sample_id}_house_components_pyvista.{ext}",
                title=f"Generated {debug_sample_id} scale {scale_id}: house components",
                min_house_component_size=args.min_house_component_size,
                small_house_component_size=args.small_house_component_size,
                max_points=args.debug_max_points,
                make_gif=args.save_debug_gif,
                n_frames=args.debug_gif_frames,
                fps=args.debug_gif_fps,
            )

            visualize_house_completeness_features_pyvista(
                generated[debug_sample_id],
                tokens,
                debug_dir / f"generated_{debug_sample_id}_house_completeness_features_pyvista.{ext}",
                title=f"Generated {debug_sample_id} scale {scale_id}: completeness features",
                min_house_component_size=args.min_house_component_size,
                max_points=args.debug_max_points,
                make_gif=args.save_debug_gif,
                n_frames=args.debug_gif_frames,
                fps=args.debug_gif_fps,
            )

    write_csv(out_dir / "metrics_summary.csv", compact_summary_rows)

    write_csv(out_dir / "tpkldiv_summary.csv", tpk_summary_rows)
    write_csv(out_dir / "tpkldiv_average_summary.csv", tpk_average_rows)

    write_csv(out_dir / "levenshtein_summary.csv", lev_summary_rows)
    write_csv(out_dir / "token_entropy_summary.csv", entropy_summary_rows)

    write_csv(out_dir / "rare_block_recall_summary.csv", rare_summary_rows)

    write_csv(out_dir / "house_fragmentation_summary.csv", house_fragmentation_rows)
    write_csv(out_dir / "house_completeness_score_summary.csv", house_completeness_score_rows)

    print("\nSaved diagnostics to:", out_dir)
    print("Main files:")
    for name in [
        "metrics_summary.csv",
        "tpkldiv_summary.csv",
        "tpkldiv_average_summary.csv",
        "levenshtein_summary.csv",
        "token_entropy_summary.csv",
        "rare_block_recall_summary.csv",
        "house_fragmentation_summary.csv",
        "house_completeness_score_summary.csv",
    ]:
        print("  -", out_dir / name)


class FlexibleBoolAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        val = str(values).lower()
        if val in {"true", "1", "yes", "y"}:
            setattr(namespace, self.dest, True)
        elif val in {"false", "0", "no", "n"}:
            setattr(namespace, self.dest, False)
        else:
            raise argparse.ArgumentError(self, f"Expected boolean, got {values}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", default=None, help="Old World-GAN training run folder with generators.pth/reals.pth")
    p.add_argument("--samples_dir", default="output_test/wandb/run-20260512_165346-5mc2kbfy-original_no_norm_village3/files/arbitrary_random_samples_v1.00000_h1.00000_st0", help="Existing samples folder with real_bdata.pt and torch_blockdata/*.pt")
    p.add_argument("--out_dir", default="output_test/wandb/run-20260512_165346-5mc2kbfy-original_no_norm_village3/files/diagnostics_worldgan", help="Where to save CSV diagnostics")
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")

    p.add_argument("--generate_per_scale", default=False, action=FlexibleBoolAction,
                   help="True: generate fresh samples and collect all scales. False: use existing final samples only.")
    p.add_argument("--num_samples", type=int, default=25)
    p.add_argument("--num_layer", type=int, default=3, help="Fallback if num_layer.pth is missing")
    p.add_argument("--gen_start_scale", type=int, default=0)
    p.add_argument("--scale_v", type=float, default=1.0)
    p.add_argument("--scale_h", type=float, default=1.0)
    p.add_argument("--scale_d", type=float, default=1.0)
    p.add_argument("--final_scale_id", type=int, default=3, help="Used only for existing final samples mode")

    p.add_argument("--repr_type", default="None",
                   choices=["None", "block2vec", "bert", "bert_naive", "neighbert", "one-hot-neighbors"],
                   help="Use None for one-hot/logit mode. Use bert/block2vec for embedding runs.")
    p.add_argument("--repr_pkl", default=None, help="Path to representation pkl used for embedding runs")
    p.add_argument("--repr_root", default="input/minecraft", help="Root for auto-loading representations")
    p.add_argument("--input_area_name", default=None, help="Area folder name for auto-loading representations")
    p.add_argument("--repr_dim", type=int, default=None, help="Representation dimension, e.g. 32 or 8")

    p.add_argument("--rare_max_count", type=int, default=10)
    p.add_argument("--rare_max_freq", type=float, default=0.005)
    p.add_argument("--min_component_size", type=int, default=5)
    p.add_argument("--pattern_sizes", type=int, nargs="+", default=[5, 10])
    p.add_argument("--tpkldiv_weight", type=float, default=0.5)
    p.add_argument("--min_house_component_size", type=int, default=20)
    p.add_argument("--small_house_component_size", type=int, default=5)
    p.add_argument("--shell_consistency_threshold", type=float, default=0.30)
    p.add_argument("--save_debug_visuals", default=True, action=FlexibleBoolAction)
    p.add_argument("--debug_sample_id", type=int, default=0)
    p.add_argument("--debug_max_points", type=int, default=30000)
    p.add_argument("--debug_scale_id", type=int, default=3)
    p.add_argument("--save_debug_gif", default=True, action=FlexibleBoolAction)
    p.add_argument("--debug_gif_frames", type=int, default=120)
    p.add_argument("--debug_gif_fps", type=int, default=20)

    return p.parse_args()


if __name__ == "__main__":
    run_diagnostics(parse_args())
