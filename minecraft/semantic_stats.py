import numpy as np
import pyvista as pv
import torch


SEM_AIR = 0
SEM_GROUND = 1
SEM_LIQUID = 2
SEM_FOLIAGE = 3
SEM_STRUCTURE = 4
SEM_DECOR = 5


SEM_NAMES = {
    SEM_AIR: "AIR",
    SEM_GROUND: "GROUND",
    SEM_LIQUID: "LIQUID",
    SEM_FOLIAGE: "FOLIAGE",
    SEM_STRUCTURE: "STRUCTURE",
    SEM_DECOR: "DECOR",
}


SEM_COLORS = {
    SEM_AIR: "#F2F2F2",        # light gray
    SEM_GROUND: "#A6761D",     # brown
    SEM_LIQUID: "#1F78B4",     # blue
    SEM_FOLIAGE: "#33A02C",    # green
    SEM_STRUCTURE: "#E31A1C",  # red
    SEM_DECOR: "#FF7F00",      # orange
}


def semantic_labels_to_voxel_points(sem_labels: torch.Tensor, include_classes=None):
    """
    sem_labels: (H, D, W) = (y, z, x)

    returns:
        dict[class_id] -> Nx3 numpy array of voxel centers in (x, y, z)
    """
    arr = sem_labels.detach().cpu().numpy()

    if include_classes is None:
        include_classes = [
            SEM_GROUND,
            SEM_LIQUID,
            SEM_FOLIAGE,
            SEM_STRUCTURE,
            SEM_DECOR,
        ]

    out = {}

    for cls in include_classes:
        coords = np.argwhere(arr == cls)  # coords in (y, z, x)

        if coords.shape[0] == 0:
            out[cls] = np.zeros((0, 3), dtype=np.float32)
            continue

        # convert (y, z, x) -> (x, y, z)
        xyz = np.stack(
            [coords[:, 2], coords[:, 0], coords[:, 1]],
            axis=1
        ).astype(np.float32)

        out[cls] = xyz

    return out


def show_semantic_pyvista(
    sem_labels: torch.Tensor,
    include_classes=None,
    opacity_map=None,
    point_size=1.0,
    show_edges=True,
    edge_color="black",
    line_width=0.3,
    window_size=(1200, 900),
):
    """
    Interactive 3D visualization of semantic voxels.

    sem_labels: (H, D, W)
    """
    if include_classes is None:
        include_classes = [
            SEM_GROUND,
            SEM_LIQUID,
            SEM_FOLIAGE,
            SEM_STRUCTURE,
            SEM_DECOR,
        ]

    if opacity_map is None:
        opacity_map = {
            SEM_GROUND: 1.0,
            SEM_LIQUID: 0.45,
            SEM_FOLIAGE: 0.75,
            SEM_STRUCTURE: 1.0,
            SEM_DECOR: 1.0,
        }

    voxel_dict = semantic_labels_to_voxel_points(
        sem_labels,
        include_classes=include_classes,
    )

    plotter = pv.Plotter(window_size=window_size)
    plotter.set_background("white")

    legend_entries = []

    for cls in include_classes:
        pts = voxel_dict[cls]

        if pts.shape[0] == 0:
            continue

        pdata = pv.PolyData(pts)

        cube = pv.Cube(
            center=(0, 0, 0),
            x_length=1,
            y_length=1,
            z_length=1,
        )

        glyphs = pdata.glyph(scale=False, geom=cube)

        plotter.add_mesh(
            glyphs,
            color=SEM_COLORS[cls],
            opacity=opacity_map.get(cls, 1.0),
            show_edges=show_edges,
            edge_color=edge_color,
            line_width=line_width,
            name=SEM_NAMES[cls],
        )

        legend_entries.append([SEM_NAMES[cls], SEM_COLORS[cls]])

    if legend_entries:
        plotter.add_legend(
            legend_entries,
            bcolor="white",
            border=True,
        )

    plotter.add_axes()
    plotter.show_grid()
    plotter.show()