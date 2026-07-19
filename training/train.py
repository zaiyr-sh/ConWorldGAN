import math
import os

import torch
import wandb
from loguru import logger

from config import Config
from minecraft.level_renderer import render_minecraft
from minecraft.level_utils import (
    build_semantic_map_from_discrete_blocks,
    convert_index_map_to_model_input,
    decode_repr_map_to_blocks,
    read_discrete_map_from_file,
    save_level_to_world,
)
from minecraft.layout_utils import semantic_to_layout2d
from minecraft.special_minecraft_downsampling import (
    special_minecraft_downsampling_discrete,
)
from models import init_D, init_D_layout, init_G
from training.train_con_single_scale import train_single_scale
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
    """Render every real pyramid level as a separate, non-overlapping OBJ."""

    obj_pth = os.path.join(opt.out_, "objects/real")
    os.makedirs(obj_pth, exist_ok=True)

    token_list = opt.token_list
    worldname = opt.output_name

    pos_offset = 0
    for scale_idx, real_tensor in enumerate(reals):
        level_scaled = decode_repr_map_to_blocks(opt, real_tensor.detach(), token_list)

        dy, dz, dx = level_scaled.shape
        pos = pos_offset
        save_level_to_world(opt, (pos, 0, 0), level_scaled)
        curr_coords = [[pos, pos + dy], [0, dz], [0, dx]]

        obj_name = f"real@scale{scale_idx}"
        obj_path = render_minecraft(worldname, curr_coords, obj_pth, obj_name)
        wandb.log({obj_name: wandb.Object3D(open(obj_path))}, commit=False)

        pos_offset += dy + 5


def train(real, opt: Config):
    """Train the progressive generator and discriminators across all scales."""

    os.makedirs(opt.out_, exist_ok=True)
    os.makedirs(os.path.join(opt.out_, "state_dicts"), exist_ok=True)

    min_scales = calc_lowest_possible_scale(real, opt.ker_size, opt.num_layer)

    scales = []
    for s in opt.scales:
        scales.append(
            [max(s, min_scales[0]), max(s, min_scales[1]), max(s, min_scales[2])]
        )
    print(scales)
    opt.num_scales = len(scales)

    index_map, pyramid_tokens, _ = read_discrete_map_from_file(opt)

    if is_repr_mode(opt) and opt.neighbors_type is None:
        if list(opt.token_list) != list(pyramid_tokens):
            raise RuntimeError(
                "Token order mismatch between read_map() and read_discrete_map_from_file(). "
                "The repr lookup table and discrete pyramid must use the same token order."
            )

    sem_map = build_semantic_map_from_discrete_blocks(
        index_map, pyramid_tokens, device=opt.device, context_kernel_size=5
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
        opt.num_scales, scales, index_map, pyramid_tokens
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
    noise_amp = []  # list of scalars per scale

    # Train one model stage for each level in the input pyramid.
    for depth in range(opt.stop_scale + 1):
        opt.outf = os.path.join(opt.out_, str(depth))
        os.makedirs(opt.outf, exist_ok=True)

        torch.save(
            reals[depth].detach().cpu(), os.path.join(opt.outf, "real_scale.pth")
        )

        netD = init_D(opt)
        netD_layout = init_D_layout(opt) if opt.use_layout_disc else None

        if depth > 0:
            prev_d = os.path.join(opt.out_, str(depth - 1), "D.pth")
            if os.path.exists(prev_d):
                netD.load_state_dict(torch.load(prev_d, map_location=opt.device))

            if netD_layout is not None:
                prev_d_layout = os.path.join(opt.out_, str(depth - 1), "D_layout.pth")
                if os.path.exists(prev_d_layout):
                    netD_layout.load_state_dict(torch.load(prev_d_layout, map_location=opt.device))

            netG.init_next_stage()
        logger.info(netG)

        # Train the current scale before saving its latest checkpoints.
        fixed_noise, noise_amp, netG, netD, netD_layout = train_single_scale(
            D=netD,
            D_layout=netD_layout,
            G=netG,
            reals=reals,
            fixed_noise=fixed_noise,
            noise_amp=noise_amp,
            opt=opt,
            depth=depth,
        )

        # Save per-scale weights.
        torch.save(netG.state_dict(), os.path.join(opt.outf, "G.pth"))
        torch.save(netD.state_dict(), os.path.join(opt.outf, "D.pth"))

        if netD_layout is not None:
            torch.save(netD_layout.state_dict(), os.path.join(opt.outf, "D_layout.pth"))

        # Update the run-level artifacts with the latest scale.
        torch.save(fixed_noise, os.path.join(opt.out_, "fixed_noise.pth"))
        torch.save(noise_amp, os.path.join(opt.out_, "noise_amp.pth"))
        torch.save(reals, os.path.join(opt.out_, "reals.pth"))

        # extra metadata
        torch.save(opt.token_list, os.path.join(opt.out_, "token_list.pth"))
        torch.save(opt.num_layer, os.path.join(opt.out_, "num_layer.pth"))

        # state_dict history
        torch.save(netG.state_dict(), os.path.join(opt.out_, "state_dicts", f"G_{depth}.pth"))
        torch.save(netD.state_dict(), os.path.join(opt.out_, "state_dicts", f"D_{depth}.pth"))

        if netD_layout is not None:
            torch.save(
                netD_layout.state_dict(),
                os.path.join(opt.out_, "state_dicts", f"D_layout_{depth}.pth"),
            )

        # wandb artifacts (optional)
        try:
            wandb.save(os.path.join(opt.outf, "*.pth"))
            wandb.save(os.path.join(opt.out_, "state_dicts", "*.pth"))
        except Exception:
            pass

        del netD
        if netD_layout is not None:
            del netD_layout

    return netG, fixed_noise, reals, noise_amp
