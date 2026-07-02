import os
from typing import Dict, List

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches
import numpy as np
import torch


LAYOUT_LABELS = [
    "Structure footprint",
    "Ground surface",
    "Water / liquid",
    "Vegetation",
    "Decorative blocks",
]

LAYOUT_FILE_NAMES = [
    "structure",
    "ground",
    "liquid",
    "foliage",
    "decor",
]

# Paper-friendly semantic colors. Feel free to adjust hex values.
CHANNEL_COLORS: Dict[str, str] = {
    "Structure footprint": "#8c564b",
    "Ground surface": "#7f7f7f",
    "Water / liquid": "#1f77b4",
    "Vegetation": "#2ca02c",
    "Decorative blocks": "#f5830a",
}


def _prepare_topdown_image(img: np.ndarray, flip_x: bool = True, flip_z: bool = False) -> np.ndarray:
    """
    Convert (X, Z) layout channel to image coordinates.

    img.T makes horizontal axis correspond to X and vertical axis correspond to Z.
    flip_x fixes left-right mirroring if the plot appears horizontally mirrored.
    flip_z can be enabled if the vertical direction is also reversed.
    """
    out = img.T
    if flip_x:
        out = np.fliplr(out)
    if flip_z:
        out = np.flipud(out)
    return out


def save_layout2d_channels_paper(
    layout2d: torch.Tensor,
    out_dir: str,
    prefix: str = "input",
    flip_x: bool = True,
    flip_z: bool = False,
    show_axes: bool = False,
    dpi: int = 300,
) -> List[str]:
    """
    Saves paper-ready top-down footprint maps, one PNG per semantic channel.

    layout2d: (1, 5, X, Z), channels are structure, ground, liquid, foliage, decor.
    """
    os.makedirs(out_dir, exist_ok=True)
    arr = layout2d.detach().cpu().squeeze(0).clamp(0.0, 1.0).numpy()

    paths = []
    for i, label in enumerate(LAYOUT_LABELS):
        img = _prepare_topdown_image(arr[i], flip_x=flip_x, flip_z=flip_z)

        fig, ax = plt.subplots(figsize=(4.2, 4.2))
        im = ax.imshow(
            img,
            origin="lower",
            interpolation="nearest",
            cmap="magma",
            vmin=0.0,
            vmax=1.0,
        )

        ax.set_title(label, fontsize=11, fontweight="bold", pad=6)
        if show_axes:
            ax.set_xlabel("Horizontal map coordinate", fontsize=9)
            ax.set_ylabel("Depth map coordinate", fontsize=9)
            ax.tick_params(labelsize=7)
        else:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("")
            ax.set_ylabel("")

        for spine in ax.spines.values():
            spine.set_linewidth(0.8)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label("Footprint probability", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

        path = os.path.join(out_dir, f"{prefix}_{LAYOUT_FILE_NAMES[i]}_footprint.png")
        fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.03)
        plt.close(fig)
        paths.append(path)

    return paths


def save_semantic_topdown_paper(
    layout2d: torch.Tensor,
    out_dir: str,
    prefix: str = "input",
    flip_x: bool = True,
    flip_z: bool = False,
    dpi: int = 300,
) -> str:
    """
    Saves one compact top-down semantic layout figure for the paper.

    Priority is used when several classes overlap after vertical projection:
    decor > structure > foliage > liquid > ground > empty.
    """
    os.makedirs(out_dir, exist_ok=True)
    arr = layout2d.detach().cpu().squeeze(0).clamp(0.0, 1.0).numpy()

    # Convert soft footprints to hard presence masks.
    masks = arr > 0.5

    # class ids: 0 empty, 1 ground, 2 liquid, 3 foliage, 4 structure, 5 decor
    semantic = np.zeros_like(arr[0], dtype=np.int32)
    semantic[masks[1]] = 1
    semantic[masks[2]] = 2
    semantic[masks[0]] = 4
    semantic[masks[3]] = 3
    semantic[masks[4]] = 5

    semantic_img = _prepare_topdown_image(semantic, flip_x=flip_x, flip_z=flip_z)

    colors = [
        "#ffffff",  # empty
        "#b8a77a",  # ground
        "#4c78a8",  # liquid
        "#59a14f",  # foliage
        "#8c564b",  # structure
        "#f5830a",  # decor
    ]
    labels = ["Empty", "Ground", "Liquid", "Foliage", "Structure", "Decor"]

    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, 6.5, 1), cmap.N)

    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.imshow(semantic_img, origin="lower", interpolation="nearest", cmap=cmap, norm=norm)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")

    patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(1, 6)]
    ax.legend(
        handles=patches,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.03),
        ncol=3,
        frameon=False,
        fontsize=8,
        handlelength=1.0,
        columnspacing=1.0,
    )

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    path = os.path.join(out_dir, f"{prefix}_semantic_topdown.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return path
