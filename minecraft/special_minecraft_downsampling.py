import torch


def _is_air(token: str) -> bool:
    return token in {"minecraft:air", "minecraft:cave_air"}


def _block_weight(token: str) -> float:
    t = token.lower()

    if _is_air(token):
        return 0.05

    if "water" in t or "lava" in t:
        return 1.2

    if any(k in t for k in ["leaves", "vine"]):
        return 2.0

    if any(k in t for k in [
        "log", "wood", "planks", "stairs", "slab",
        "fence", "fence_gate", "cobblestone", "stone", "brick", "wall",
        "glass_pane", "iron_bars"
    ]):
        return 4.0

    if any(k in t for k in [
        "door", "trapdoor", "oak_door"
    ]):
        return 8.0

    if any(k in t for k in [
        "torch", "flower", "grass", "seagrass", "lily_pad", "carpet",
        "ladder", "bell", "bed", "chest", "composter", "wheat",
        "brewing_stand", "grindstone", "smoker", "furnace",
        "blast_furnace", "pressure_plate"
    ]):
        return 0.35

    if any(k in t for k in [
        "dirt", "grass_block", "grass_path", "sand", "clay", "gravel", "farmland"
    ]):
        return 1.0

    return 1.0


def _make_axis_bins(old_size: int, new_size: int):
    edges = torch.linspace(0, old_size, new_size + 1)
    bins = []
    for i in range(new_size):
        start = int(torch.floor(edges[i]).item())
        end = int(torch.floor(edges[i + 1]).item())

        if end <= start:
            end = min(old_size, start + 1)

        if i == new_size - 1:
            end = old_size

        bins.append((start, end))
    return bins


def _choose_block_for_cell(cell: torch.Tensor, token_list):
    """
    cell: (hy, hz, hx) long tensor of block indices
    returns: chosen block index (int)
    """
    if cell.numel() == 1:
        return int(cell.item())

    cy = (cell.shape[0] - 1) / 2.0
    cz = (cell.shape[1] - 1) / 2.0
    cx = (cell.shape[2] - 1) / 2.0

    flat_vals = cell.flatten().tolist()
    non_air_present = any(not _is_air(token_list[int(v)]) for v in flat_vals)

    scores = {}

    for y in range(cell.shape[0]):
        for z in range(cell.shape[1]):
            for x in range(cell.shape[2]):
                idx = int(cell[y, z, x].item())
                token = token_list[idx]

                score = _block_weight(token)

                if non_air_present and _is_air(token):
                    score *= 0.02

                dist = abs(y - cy) + abs(z - cz) + abs(x - cx)
                score *= 1.0 + 0.35 / (1.0 + dist)

                if any(k in token.lower() for k in ["log", "planks", "stairs", "cobblestone", "wood", "brick"]):
                    score *= 1.35

                scores[idx] = scores.get(idx, 0.0) + score

    best_idx = max(scores.items(), key=lambda kv: kv[1])[0]
    return int(best_idx)


def downsample_index_map(index_map: torch.Tensor, target_shape, token_list):
    """
    index_map: (H, D, W) long tensor
    target_shape: (new_H, new_D, new_W)
    """
    old_h, old_d, old_w = index_map.shape
    new_h, new_d, new_w = target_shape

    y_bins = _make_axis_bins(old_h, new_h)
    z_bins = _make_axis_bins(old_d, new_d)
    x_bins = _make_axis_bins(old_w, new_w)

    out = torch.zeros((new_h, new_d, new_w), dtype=torch.long)

    for iy, (y0, y1) in enumerate(y_bins):
        for iz, (z0, z1) in enumerate(z_bins):
            for ix, (x0, x1) in enumerate(x_bins):
                cell = index_map[y0:y1, z0:z1, x0:x1]
                out[iy, iz, ix] = _choose_block_for_cell(cell, token_list)

    return out


def special_minecraft_downsampling_discrete(num_scales, scales, index_map, token_list):
    """
    index_map: (H, D, W) long tensor of discrete block ids
    scales: list of [sy, sz, sx]
    returns: list of downsampled discrete maps from coarse -> fine
    """
    scaled_list = []

    for sc in range(num_scales):
        sy, sz, sx = scales[sc]

        target_shape = (
            max(1, int(round(index_map.shape[0] * sy))),
            max(1, int(round(index_map.shape[1] * sz))),
            max(1, int(round(index_map.shape[2] * sx))),
        )

        scaled = downsample_index_map(index_map, target_shape, token_list)
        scaled_list.append(scaled)

    scaled_list.reverse()
    return scaled_list