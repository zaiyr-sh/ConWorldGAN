from config import Config
import os

import torch
import wandb
import math
from loguru import logger

from minecraft.level_renderer import render_minecraft
from models import init_D, init_D_sem, init_G, init_D_layout
from training.train_con_single_scale import train_single_scale
from minecraft.special_minecraft_downsampling import special_minecraft_downsampling_discrete
from minecraft.level_utils import (
    decode_repr_map_to_blocks,
    save_level_to_world,
    read_discrete_map_from_file,
    convert_index_map_to_model_input,
    build_semantic_label_map_from_discrete_blocks,
    build_semantic_map_from_discrete_blocks,
)

from minecraft.layout_utils import semantic_to_layout2d
from minecraft.semantic_stats import show_semantic_pyvista
from utils import call_wine, is_repr_mode


def calc_lowest_possible_scale(level, kernel_size, num_layers):
    """Calculates the lowest size the generator will accept in each dimension.
    It depends on the number/size of layers."""
    needed_pad = math.floor(kernel_size/2) * num_layers
    min_size = (needed_pad * 2) + 2
    sizes = level.shape[2:]
    lowest_scales = []
    for dim in sizes:
        lowest_scales.append(min_size/dim)
    return lowest_scales

def render_real_pyramid(reals, opt: Config):
    """
    Save the original world slice + every downsampled level in `reals`
    as separate OBJ files under objects/real.
    """
    obj_pth = os.path.join(opt.out_, "objects/real")
    os.makedirs(obj_pth, exist_ok=True)

    # 1) Original slice from the input world, as before
    # real_obj_pth = render_minecraft(opt.input_name, opt.coords, obj_pth, "real")
    # wandb.log({"real": wandb.Object3D(open(real_obj_pth))}, commit=False)

    # 2) All tensor scales from `reals`
    token_list = opt.token_list
    worldname = opt.output_name  # temporary world to write tensors into

    # clean world before writing
    # clear_empty_world(opt.output_dir, worldname)

    pos_offset = 0
    for scale_idx, real_tensor in enumerate(reals):
        # real_tensor: [1, C, Y, Z, X] (one-hot or embedding)
        level_scaled = decode_repr_map_to_blocks(
            opt, real_tensor.detach(), token_list
        )  # -> [Y, Z, X] of block ids

        # place each scale next to each other along X so they don't overlap
        dy, dz, dx = level_scaled.shape
        pos = pos_offset
        save_level_to_world(opt,(pos, 0, 0), level_scaled)
        curr_coords = [[pos, pos + dy],
                       [0, dz],
                       [0, dx]]

        obj_name = f"real@scale{scale_idx}"
        obj_path = render_minecraft(worldname, curr_coords, obj_pth, obj_name)
        wandb.log({obj_name: wandb.Object3D(open(obj_path))}, commit=False)

        pos_offset += dy + 5  # leave gap between scales

def train(real, opt: Config):

    os.makedirs(opt.out_, exist_ok=True)
    os.makedirs(os.path.join(opt.out_, "state_dicts"), exist_ok=True)

    min_scales = calc_lowest_possible_scale(real, opt.ker_size, opt.num_layer)

    scales = []
    for s in opt.scales:
        scales.append([max(s, min_scales[0]), max(s, min_scales[1]), max(s, min_scales[2])])
    print(scales)
    opt.num_scales = len(scales)

    index_map, pyramid_tokens, pyramid_props = read_discrete_map_from_file(opt)

    if is_repr_mode(opt) and opt.neighbors_type is None:
        if list(opt.token_list) != list(pyramid_tokens):
            raise RuntimeError(
                "Token order mismatch between read_map() and read_discrete_map_from_file(). "
                "The repr lookup table and discrete pyramid must use the same token order."
            )

    sem_labels = build_semantic_label_map_from_discrete_blocks(
        index_map,
        pyramid_tokens,
        context_kernel_size=5
    )

    # show_semantic_pyvista(sem_labels)

    sem_map = build_semantic_map_from_discrete_blocks(
        index_map,
        pyramid_tokens,
        device=opt.device,
        context_kernel_size=5
    )

    layout2d = semantic_to_layout2d(sem_map)

    print("sem_map:", sem_map.shape)  # (1, 6, H, D, W)
    print("layout2d:", layout2d.shape)  # (1, 5, D, W)
    print("layout min/max:", layout2d.min().item(), layout2d.max().item())
    print("structure footprint mean:", layout2d[:, 0:1].mean().item())

    from minecraft.layout_debug import (
        save_layout2d_channels_paper,
        save_semantic_topdown_paper,
    )

    paths = save_layout2d_channels_paper(
        layout2d,
        out_dir=os.path.join(opt.out_, "debug_layout"),
        prefix="real",
        flip_x=True,
    )

    semantic_path = save_semantic_topdown_paper(
        layout2d,
        out_dir=os.path.join(opt.out_, "debug_layout"),
        prefix="real",
        flip_x=True,
    )

    paths.append(semantic_path)

    for p in paths:
        wandb.log({os.path.basename(p): wandb.Image(p)}, commit=False)

    scaled_index_list = special_minecraft_downsampling_discrete(
        opt.num_scales,
        scales,
        index_map,
        pyramid_tokens
    )

    reals = [
        convert_index_map_to_model_input(level_idx, pyramid_tokens, opt)
        for level_idx in [*scaled_index_list, index_map]
    ]

    opt.stop_scale = len(reals) - 1

    print("Pyramid shapes:")
    for i, r in enumerate(reals):
        print(i, tuple(r.shape))

    # Log the original input level(s) as an image
    call_wine(on_success=lambda: render_real_pyramid(reals, opt))
    os.makedirs("%s/state_dicts" % (opt.out_), exist_ok=True)

    opt.repr_channels = real.shape[1]
    netG = init_G(opt)

    fixed_noise = []  # list of noise tensors per scale
    noise_amp = []    # list of scalars per scale

    # Training Loop
    for depth in range(opt.stop_scale + 1):
        opt.outf = os.path.join(opt.out_, str(depth))
        os.makedirs(opt.outf, exist_ok=True)

        torch.save(reals[depth].detach().cpu(), os.path.join(opt.outf, "real_scale.pth"))

        netD = init_D(opt)
        netD_sem = init_D_sem(opt) if opt.use_semantic_disc else None
        netD_layout = init_D_layout(opt) if opt.use_layout_disc else None

        if depth > 0:
            prev_d = os.path.join(opt.out_, str(depth - 1), "D.pth")
            if os.path.exists(prev_d):
                netD.load_state_dict(torch.load(prev_d, map_location=opt.device))

            if netD_sem is not None:
                prev_d_sem = os.path.join(opt.out_, str(depth - 1), "D_sem.pth")
                if os.path.exists(prev_d_sem):
                    netD_sem.load_state_dict(torch.load(prev_d_sem, map_location=opt.device))

            if netD_layout is not None:
                prev_d_layout = os.path.join(opt.out_, str(depth - 1), "D_layout.pth")
                if os.path.exists(prev_d_layout):
                    netD_layout.load_state_dict(torch.load(prev_d_layout, map_location=opt.device))

            netG.init_next_stage()
        logger.info(netG)

        # train one scale (your ConSinGAN-style train_single_scale)
        fixed_noise, noise_amp, netG, netD, netD_sem, netD_layout = train_single_scale(
            D=netD,
            D_sem=netD_sem,
            D_layout=netD_layout,
            G=netG,
            reals=reals,
            fixed_noise=fixed_noise,
            noise_amp=noise_amp,
            opt=opt,
            depth=depth
        )

        # ---------- saving ----------
        # per-scale weights
        torch.save(netG.state_dict(), os.path.join(opt.outf, "G.pth"))
        torch.save(netD.state_dict(), os.path.join(opt.outf, "D.pth"))

        if netD_sem is not None:
            torch.save(netD_sem.state_dict(), os.path.join(opt.outf, "D_sem.pth"))

        if netD_layout is not None:
            torch.save(netD_layout.state_dict(), os.path.join(opt.outf, "D_layout.pth"))

        # global artifacts (latest snapshot)
        torch.save(fixed_noise, os.path.join(opt.out_, "fixed_noise.pth"))
        torch.save(noise_amp, os.path.join(opt.out_, "noise_amp.pth"))
        torch.save(reals, os.path.join(opt.out_, "reals.pth"))

        # extra metadata
        torch.save(opt.token_list, os.path.join(opt.out_, "token_list.pth"))
        torch.save(opt.num_layer, os.path.join(opt.out_, "num_layer.pth"))

        # state_dict history
        torch.save(netG.state_dict(), os.path.join(opt.out_, "state_dicts", f"G_{depth}.pth"))
        torch.save(netD.state_dict(), os.path.join(opt.out_, "state_dicts", f"D_{depth}.pth"))

        if netD_sem is not None:
            torch.save(netD_sem.state_dict(), os.path.join(opt.out_, "state_dicts", f"D_sem_{depth}.pth"))

        if netD_layout is not None:
            torch.save(netD_layout.state_dict(), os.path.join(opt.out_, "state_dicts", f"D_layout_{depth}.pth"))

        # wandb artifacts (optional)
        try:
            wandb.save(os.path.join(opt.outf, "*.pth"))
            wandb.save(os.path.join(opt.out_, "state_dicts", "*.pth"))
        except Exception:
            pass

        del netD
        if netD_sem is not None:
            del netD_sem

        if netD_layout is not None:
            del netD_layout

    return netG, fixed_noise, reals, noise_amp
