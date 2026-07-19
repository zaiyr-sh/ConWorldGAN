import os
import shutil
from typing import List, Dict, Tuple, Any, Optional
import torch
from loguru import logger
from PyAnvilEditor.pyanvil import World, BlockState
from config import Config
from utils import collect_neighbors_for_voxel, make_repr_tensor, make_index_tensor, is_repr_mode, to_one_hot
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image
import math
from pathlib import Path
import torch.nn.functional as F

def decode_logits_grid_to_indices(logit_map: torch.Tensor) -> torch.Tensor:
    """Convert an index or channel-first logit grid to CPU block indices."""

    if logit_map.ndim == 3:               # already indices (H,D,W)
        return logit_map.cpu()
    if logit_map.ndim == 5:               # (1,C,H,D,W)
        logit_map = logit_map.squeeze(0)  # (C,H,D,W)
        return logit_map.argmax(dim=0).cpu()
    if logit_map.ndim == 4:               # (C,H,D,W)
        return logit_map.argmax(dim=0).cpu()
    raise ValueError(f"Unexpected shape: {tuple(logit_map.shape)}")

def decode_repr_grid_to_indices(repr_map: torch.Tensor, repr_table: torch.Tensor) -> torch.Tensor:
    """Decode each voxel to the nearest representation by Euclidean distance."""

    voxel_repr = repr_map.squeeze(0).permute(1, 2, 3, 0)[..., None]  # (H, D, W, C, 1)
    table_repr = repr_table.T[None, None, None, ...] # (1, 1, 1, C, N)
    dist = (voxel_repr - table_repr).pow(2).sum(dim=-2) # squared L2 distance over C
    return dist.argmin(dim=-1).cpu()  # nearest token

@torch.no_grad()
def decode_repr_grid_to_indices_cos(opt, repr_map: torch.Tensor, tokens) -> torch.Tensor:
    """Decode a representation grid to CPU token indices by cosine similarity."""

    device = repr_map.device
    dtype = repr_map.dtype

    # (N, C)
    repr_table = torch.stack([opt.block2repr[t] for t in tokens]).to(device=device, dtype=dtype)

    # (H, D, W, C)
    x = repr_map.squeeze(0).permute(1, 2, 3, 0).contiguous()

    # L2-normalize (cosine)
    x = F.normalize(x, dim=-1, eps=1e-8)
    t = F.normalize(repr_table, dim=-1, eps=1e-8)

    # cosine sim: (H, D, W, N)
    sim = torch.einsum("hdwc,nc->hdwn", x, t)

    # (H, D, W)
    idx = sim.argmax(dim=-1)
    return idx.to(torch.long).cpu()

def decode_repr_map_to_blocks(opt: Config, repr_map, tokens):
    """Decode a model output to a discrete ``(Y, Z, X)`` block-index grid."""

    with torch.no_grad():
        if is_repr_mode(opt):
            repr_table = torch.stack([opt.block2repr[t] for t in tokens])
            block_index_grid = decode_repr_grid_to_indices(repr_map, repr_table)
        else:
            block_index_grid = decode_logits_grid_to_indices(repr_map)
    return block_index_grid

SEM_AIR = 0
SEM_GROUND = 1
SEM_LIQUID = 2
SEM_FOLIAGE = 3
SEM_STRUCTURE = 4
SEM_DECOR = 5


def semantic_group_of_token(token: str) -> int:
    """Map a Minecraft block token to one of the six layout semantic classes."""

    t = token.lower()

    if t in {"minecraft:air", "minecraft:cave_air"}:
        return SEM_AIR

    if "water" in t or "lava" in t:
        return SEM_LIQUID

    # structural first, before generic stone/wood checks
    if any(k in t for k in [
        "cobblestone", "brick", "terracotta", "planks", "stairs", "slab",
        "door", "trapdoor", "fence", "fence_gate", "glass_pane", "iron_bars",
        "smooth_stone", "wall", "log", "wood"
    ]):
        return SEM_STRUCTURE

    if any(k in t for k in [
        "leaves", "vine"
    ]):
        return SEM_FOLIAGE

    if any(k in t for k in [
        "dirt", "grass_block", "grass_path", "sand", "gravel", "clay",
        "farmland", "stone", "andesite", "diorite", "granite", "ore"
    ]):
        return SEM_GROUND

    if any(k in t for k in [
        "torch", "bed", "chest", "carpet", "ladder", "bell",
        "composter", "wheat", "flower", "grass", "seagrass",
        "lily_pad", "dead_bush", "smoker", "furnace", "blast_furnace",
        "grindstone", "brewing_stand", "pressure_plate"
    ]):
        return SEM_DECOR

    return SEM_DECOR


def build_semantic_group_matrix(tokens, device):
    """Build a class-by-token matrix used to aggregate block probabilities."""

    mat = torch.zeros((6, len(tokens)), device=device, dtype=torch.float32)
    for i, tok in enumerate(tokens):
        g = semantic_group_of_token(tok)
        mat[g, i] = 1.0
    return mat


def repr_to_block_probs(opt: Config, repr_map: torch.Tensor, tokens, tau: float = 1.0) -> torch.Tensor:
    """Convert model outputs to per-voxel block probabilities."""

    if not is_repr_mode(opt):
        # logits mode
        probs = torch.softmax(repr_map, dim=1)
        return probs.permute(0, 2, 3, 4, 1).contiguous()

    device = repr_map.device
    dtype = repr_map.dtype

    repr_table = torch.stack([opt.block2repr[t] for t in tokens]).to(device=device, dtype=dtype)  # (N, C)

    x = repr_map.permute(0, 2, 3, 4, 1).contiguous()  # (B, H, D, W, C)

    # squared L2 distance to each token embedding
    dist = (x.unsqueeze(-2) - repr_table.view(1, 1, 1, 1, repr_table.shape[0], repr_table.shape[1])).pow(2).sum(dim=-1)

    logits = -dist / max(float(tau), 1e-6)
    probs = torch.softmax(logits, dim=-1)
    return probs


def repr_to_semantic_map(opt: Config, repr_map: torch.Tensor, tokens, tau: float = 1.0) -> torch.Tensor:
    """Project model outputs to six semantic channels for layout training."""

    block_probs = repr_to_block_probs(opt, repr_map, tokens, tau=tau)  # (B, H, D, W, N)
    group_mat = build_semantic_group_matrix(tokens, repr_map.device)    # (K, N)

    sem = torch.einsum("bhdwn,kn->bkhdw", block_probs, group_mat)
    return sem.contiguous()

def build_semantic_map_from_discrete_blocks(
    index_map: torch.Tensor,
    tokens,
    device=None,
    context_kernel_size: int = 3,
) -> torch.Tensor:
    """
    Convert discrete block indices to semantic probability map.

    index_map: (H, D, W) long tensor
    tokens: list[str]
    returns: (1, 6, H, D, W)

    context_kernel_size is kept only for compatibility with the newer code.
    In the old 6-class implementation, semantic labels are assigned directly
    by token name, without context-aware smoothing.
    """
    if device is None:
        device = index_map.device

    index_map = index_map.long().to(device)

    block_probs = F.one_hot(
        index_map,
        num_classes=len(tokens)
    ).float()  # (H, D, W, N)

    block_probs = block_probs.unsqueeze(0)  # (1, H, D, W, N)

    group_mat = build_semantic_group_matrix(tokens, device)  # (6, N)

    sem = torch.einsum("bhdwn,kn->bkhdw", block_probs, group_mat)

    return sem.contiguous()


def build_semantic_label_map_from_discrete_blocks(
    index_map: torch.Tensor,
    tokens,
    device=None,
    context_kernel_size: int = 3,
) -> torch.Tensor:
    """
    Convert discrete block indices to hard semantic labels.

    index_map: (H, D, W)
    returns: (H, D, W), labels in:
        0 = AIR
        1 = GROUND
        2 = LIQUID
        3 = FOLIAGE
        4 = STRUCTURE
        5 = DECOR
    """
    sem = build_semantic_map_from_discrete_blocks(
        index_map=index_map,
        tokens=tokens,
        device=device,
        context_kernel_size=context_kernel_size,
    )

    return sem.argmax(dim=1).squeeze(0).cpu()

def read_map(opt: Config):
    """Load the configured world region and attach its metadata to ``opt``."""

    level, uniques, props, neighbor_info = read_map_from_file(opt=opt)
    opt.token_list = uniques
    opt.neighbor_info = neighbor_info
    opt.props = props
    logger.info(f"There are {len(opt.token_list)} tokens in this map: {opt.token_list}")
    opt.repr_channels = level.shape[1]
    return level

def init_map(opt: Config):
    """Allocate an empty map and token metadata for the configured representation."""

    if opt.repr_type in ["bert", "clip"]:
        uniques = [u for u in opt.block2repr.keys()]
        props = [None for _ in range(len(uniques))]
        map = make_repr_tensor(opt)
    else:
        uniques = []
        props = []
        map = make_index_tensor(opt)
    neighbor_info: Optional[Dict[Tuple[int, int, int], List[Dict[str, Any]]]] = {}
    return map, uniques, props, neighbor_info

def resolve_repr_key(opt, block, b_name, y, z, x):
    """Return the representation-table key for a block at a world coordinate."""

    if opt.neighbors_type != "local":
        return b_name

    if b_name == "minecraft:air":
        return "air"

    clean = block.get_state().get_clean_name()
    return f"{clean}_{(y, z, x)}"

def read_map_from_file(opt: Config):
    """Read the configured Minecraft region into a model-ready tensor."""

    (y0, y1), (z0, z1), (x0, x1) = opt.coords
    repr_mode = is_repr_mode(opt)

    map, uniques, props, neighbor_info = init_map(opt=opt)

    with open("blocks.txt", "w", encoding="utf-8") as f:
        with World(opt.input_name, opt.input_dir, debug=opt.debug) as wrld:
            for y in range(y0, y1):
                for z in range(z0, z1):
                    for x in range(x0, x1):
                        block = wrld.get_block((y, z, x))
                        b_name = block.get_state().name
                        neighbor_info[(y, z, x)] = collect_neighbors_for_voxel(wrld, (y, z, x), opt.coords)

                        iy, iz, ix = y - y0, z - z0, x - x0

                        if repr_mode:
                            repr_key = resolve_repr_key(opt, block, b_name, y, z, x)
                            f.write(f"({y}, {z}, {x}): {repr_key}\n")
                            map[0, :, iy, iz, ix] = opt.block2repr[repr_key]
                            if props[uniques.index(repr_key)] is None:
                                props[uniques.index(repr_key)] = block.get_state().props
                        else:
                            f.write(f"({y}, {z}, {x}): {b_name}\n")
                            if b_name not in uniques:
                                uniques.append(b_name)
                                props.append(block.get_state().props)
                            map[iy, iz, ix] = uniques.index(b_name)

    if repr_mode:
        final_map = map
    else:
        final_map = to_one_hot(map, uniques)
    return final_map, uniques, props, neighbor_info

def read_discrete_map_from_file(opt: Config):
    """
    Reads the map as discrete block ids, without repr interpolation.
    Returns:
        index_map: (H, D, W) long
        tokens: list[str]
        props: list[dict|None]
    """
    if is_repr_mode(opt) and opt.neighbors_type == "local":
        raise NotImplementedError(
            "Discrete pyramid patch currently supports neighbors_type=None only."
        )

    (y0, y1), (z0, z1), (x0, x1) = opt.coords
    H, D, W = y1 - y0, z1 - z0, x1 - x0

    index_map = torch.zeros((H, D, W), dtype=torch.long)

    # In repr mode, we take the token order from block2repr so that the indices match the repr lookup table
    if is_repr_mode(opt):
        tokens = list(opt.block2repr.keys())
        props = [None for _ in tokens]
        token_to_idx = {tok: i for i, tok in enumerate(tokens)}
    else:
        tokens = []
        props = []
        token_to_idx = {}

    with World(opt.input_name, opt.input_dir, debug=opt.debug) as wrld:
        for y in range(y0, y1):
            for z in range(z0, z1):
                for x in range(x0, x1):
                    block = wrld.get_block((y, z, x))
                    b_name = block.get_state().name
                    iy, iz, ix = y - y0, z - z0, x - x0

                    if b_name not in token_to_idx:
                        token_to_idx[b_name] = len(tokens)
                        tokens.append(b_name)
                        props.append(block.get_state().props)

                    idx = token_to_idx[b_name]
                    index_map[iy, iz, ix] = idx

                    if props[idx] is None:
                        props[idx] = block.get_state().props

    return index_map, tokens, props


def convert_index_map_to_model_input(index_map: torch.Tensor, tokens, opt: Config) -> torch.Tensor:
    """
    Converts a discrete block map back to a model input:
    - repr tensor, if repr_type in ['bert', 'clip']
    - one-hot tensor, if repr_type is None
    Returns tensor shape: (1, C, H, D, W)
    """
    if is_repr_mode(opt):
        idx = index_map.to(torch.long).to(opt.device)  # (H, D, W)
        repr_table = torch.stack([opt.block2repr[t] for t in tokens]).to(opt.device)  # (N, C)

        # repr_table[idx] -> (H, D, W, C)
        out = repr_table[idx]
        out = out.permute(3, 0, 1, 2).unsqueeze(0).contiguous()  # (1, C, H, D, W)
        return out.float()

    return to_one_hot(index_map.cpu(), tokens).to(opt.device).float()

def resolve_block_name(opt, token: str) -> str:
    if opt.neighbors_type != "local":
        return token

    clean = token.split("_(")[0]
    return f"minecraft:{clean.replace(' ', '_')}"

def save_level_to_world(opt: Config, start_coords, blocks):
    """Write a block-index tensor into the output world at ``start_coords``."""

    if opt.props is None:
        props = [{} for _ in range(len(opt.token_list))]
    else:
        props = opt.props

    with World(opt.output_name, opt.output_dir, debug=opt.debug) as wrld:
        y0, z0, x0 = start_coords
        H, D, W = blocks.shape

        for y in range(y0, y0 + H):
            for z in range(z0, z0 + D):
                for x in range(x0, x0 + W):
                    iy, iz, ix = y - y0, z - z0, x - x0
                    token_idx = int(blocks[iy, iz, ix])
                    try:
                        token = opt.token_list[token_idx]
                        block_name = resolve_block_name(opt, token)
                        block_props = props[token_idx]
                        block = wrld.get_block((y, z, x))
                        block.set_state(BlockState(block_name, block_props))
                    except Exception as e:
                        logger.error(f"[ERROR] Failed to set block at {(y, z, x)}")
                        logger.error(f"  local idx: {(iy, iz, ix)}")
                        logger.error(f"  token idx: {token_idx}")
                        logger.error(f"  error: {e}")

def clear_empty_world(worlds_folder, empty_world_name='Curr_Empty_World'):
    src = os.path.join(worlds_folder, 'Drehmal v2.1 PRIMORDIAL')
    dst = os.path.join(worlds_folder, empty_world_name)
    shutil.rmtree(dst)
    shutil.copytree(src, dst)

def render_world(render_path: str, opt: Config, num_viewpoints: int = 20):
    from pytorch3d.io import load_objs_as_meshes, load_obj
    from pytorch3d.renderer import (look_at_view_transform, FoVPerspectiveCameras, PointLights, Materials,
                                    RasterizationSettings, MeshRenderer, MeshRasterizer, SoftPhongShader)

    obj = Path(render_path)
    logger.info("OBJ:", obj)
    logger.info("MTL:", obj.with_suffix(".mtl"), obj.with_suffix(".mtl").exists())

    # Check texture filenames referenced by MTL exist
    if obj.with_suffix(".mtl").exists():
        mtl_txt = obj.with_suffix(".mtl").read_text(errors="ignore").splitlines()
        tex_lines = [l.strip() for l in mtl_txt if l.strip().lower().startswith("map_kd")]
        logger.info("map_Kd lines:", tex_lines[:5])
        for l in tex_lines:
            tex = l.split(maxsplit=1)[1].strip()
            tex_path = (obj.parent / tex).resolve()
            logger.info("tex exists:", tex_path.exists(), "->", tex_path)

    # What load_objs_as_meshes gives you
    mesh0 = load_objs_as_meshes([render_path], device=opt.device)
    logger.info("load_objs_as_meshes texture type:", type(mesh0.textures))

    # What load_obj gives you (more detailed)
    verts, faces, aux = load_obj(render_path, load_textures=True, create_texture_atlas=True)
    logger.info("aux.texture_images keys:", list(aux.texture_images.keys())[:5])
    if aux.verts_uvs is not None:
        uvmin = aux.verts_uvs.min(dim=0).values
        uvmax = aux.verts_uvs.max(dim=0).values
        logger.info("UV min:", uvmin, "UV max:", uvmax)

    mesh = load_objs_as_meshes(
        [render_path],
        device=opt.device,
        load_textures=True,
        create_texture_atlas=True,
        texture_atlas_size=32,  # try 16 or 32 for Minecraft textures
    )
    meshes = mesh.extend(num_viewpoints)
    lights = PointLights(device=opt.device, location=[[0.0, 0.0, -3.0]])

    elev = torch.full((num_viewpoints,), 20.0, device=opt.device)
    azim = torch.linspace(-180, 180, num_viewpoints, device=opt.device)

    R, T = look_at_view_transform(dist=3, elev=elev, azim=azim)
    cameras = FoVPerspectiveCameras(device=opt.device, R=R, T=T)
    materials = Materials(
        device=opt.device,
        shininess=0.0
    )
    raster_settings = RasterizationSettings(
        image_size=512,
        blur_radius=0.0,
        faces_per_pixel=1,
    )
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            device=opt.device,
            cameras=cameras,
            lights=lights
        )
    )

    # Move the light back in front of the cow which is facing the -z direction.
    lights.location = torch.tensor([[1.0, 1.0, -3.0]], device=opt.device)
    images = renderer(meshes, cameras=cameras,
                      lights=lights, materials=materials)
    image_grid = to_pil_image(make_grid(torch.permute(images[..., :3], dims=[0, 3, 1, 2]), nrow=math.floor(math.sqrt(num_viewpoints))))
    return image_grid
