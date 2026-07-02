import math
import os
import subprocess
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

from config import Config
from generation.generate_noise import generate_spatial_noise
from minecraft.level_renderer import render_minecraft
from minecraft.level_utils import decode_repr_map_to_blocks, decode_repr_grid_to_indices_cos, save_level_to_world, clear_empty_world
from minecraft.level_utils import read_map as mc_read_level
from utils import interpolate3D

def _find_last_depth(out_dir: str) -> int:
    depths = []
    for name in os.listdir(out_dir):
        if name.isdigit() and os.path.isfile(os.path.join(out_dir, name, "G.pth")):
            depths.append(int(name))
    return max(depths) if depths else 0


def _build_gen_shapes_from_reals(reals, scale_v: float, scale_h: float, scale_d: float):
    """
    Minecraft ordering: [B, C, Y, Z, X]
    scale_h -> Y, scale_d -> Z, scale_v -> X
    """
    shapes = []
    for r in reals:
        b, c, y, z, x = r.shape
        y2 = max(1, int(round(y * scale_h)))
        z2 = max(1, int(round(z * scale_d)))
        x2 = max(1, int(round(x * scale_v)))
        shapes.append(torch.Size([b, c, y2, z2, x2]))
    return shapes


def _resize_noise_like(n: torch.Tensor, target_spatial):
    # target_spatial: (Y, Z, X)
    if tuple(n.shape[2:]) == tuple(target_spatial):
        return n
    return interpolate3D(n, target_spatial, align_corners=True)


def _sample_noise_list_3d(
    opt,
    gen_shapes,
    noise_amp,
    fixed_noise=None,
    gen_start_scale: int = 0,
):
    """
    ConSinGAN-like:
      noise[0]  : [1, repr_channels, Y0, Z0, X0]
      noise[d>0]: [1, hidden_channel, Yd+extra, Zd+extra, Xd+extra]
    """
    device = opt.device
    repr_ch = int(opt.repr_channels)
    hid_ch = int(opt.hidden_channel)
    eff = max(1, int(opt.num_layer))  # same as G.n_blocks
    extra = eff * 2

    depth_max = len(noise_amp) - 1
    noise = []

    for d in range(depth_max + 1):
        y, z, x = int(gen_shapes[d][2]), int(gen_shapes[d][3]), int(gen_shapes[d][4])

        if d == 0:
            target = (y, z, x)
            if fixed_noise is not None and d < gen_start_scale:
                n = fixed_noise[d].to(device)
                n = _resize_noise_like(n, target)
            else:
                n = generate_spatial_noise((1, repr_ch, y, z, x), device=device).detach()
        else:
            target = (y + extra, z + extra, x + extra)
            if fixed_noise is not None and d < gen_start_scale:
                n = fixed_noise[d].to(device)
                n = _resize_noise_like(n, target)
            else:
                n = generate_spatial_noise((1, hid_ch, *target), device=device).detach()

        noise.append(n)

    return noise

from models import init_G  # добавь импорт

def load_trained_pyramid_cons(opt):
    last_depth = _find_last_depth(opt.out_)
    reals = torch.load(os.path.join(opt.out_, "reals.pth"), map_location=opt.device)
    fixed_noise = torch.load(os.path.join(opt.out_, "fixed_noise.pth"), map_location=opt.device)
    noise_amp = torch.load(os.path.join(opt.out_, "noise_amp.pth"), map_location=opt.device)

    tl_path = os.path.join(opt.out_, "token_list.pth")
    if os.path.exists(tl_path):
        opt.token_list = torch.load(tl_path)

    netG = init_G(opt).to(opt.device)
    for _ in range(last_depth):
        netG.init_next_stage()

    g_path = os.path.join(opt.out_, str(last_depth), "G.pth")
    netG.load_state_dict(torch.load(g_path, map_location=opt.device))
    netG.eval()

    return netG, fixed_noise, reals, noise_amp, last_depth


class GenerateSamplesConfig(Config):
    scale_v: float = 1.0    # vertical scale factor
    scale_h: float = 1.0  # horizontal scale factor
    scale_d: float = 1.0  # horizontal scale factor
    gen_start_scale: int = 0  # scale to start generating in
    num_samples: int = 25 # number of samples to be generated
    save_tensors: bool = False  # save pytorch .pt tensors?
    not_cuda: bool = True  # disables cuda
    generators_dir: Optional[str] = None

    def process_args(self):
        super().process_args()
        self.seed_road: Optional[torch.Tensor] = None
        self.out_: Optional[str] = "output_test/wandb/run-20260506_144855-j96a8j8z/files"
        if not self.out_:
            raise Exception('--out_ is required')


def generate_samples_cons(
    netG,
    fixed_noise,
    reals,
    noise_amp,
    opt: GenerateSamplesConfig,
    scale_v=1.0,
    scale_h=1.0,
    scale_d=1.0,
    gen_start_scale=0,
    num_samples=15,
    render_images=True,
    save_tensors=False,
    save_dir="random_samples",
):
    dir2save = os.path.join(opt.out_, save_dir)
    os.makedirs(dir2save, exist_ok=True)
    if save_tensors:
        os.makedirs(os.path.join(dir2save, "torch"), exist_ok=True)
    os.makedirs(os.path.join(dir2save, "torch_blockdata"), exist_ok=True)

    gen_shapes = _build_gen_shapes_from_reals(reals, scale_v=scale_v, scale_h=scale_h, scale_d=scale_d)

    # token list
    if opt.repr_type is not None:
        token_list = list(opt.block2repr.keys()) if hasattr(opt, "block2repr") and opt.block2repr is not None else opt.token_list
    else:
        token_list = opt.token_list

    real_level = decode_repr_map_to_blocks(opt, reals[-1].detach(), token_list)
    torch.save(real_level, os.path.join(dir2save, "real_bdata.pt"))
    torch.save(token_list, os.path.join(dir2save, "token_list.pt"))

    if render_images:
        try:
            real_pth = os.path.join(dir2save, "reals")
            os.makedirs(real_pth, exist_ok=True)
            base_x = opt.coords[0][0]
            base_y = opt.coords[1][0]
            base_z = opt.coords[2][0]

            save_level_to_world(opt, (base_x, base_y, base_z), real_level)
            curr_coords = [
                [base_x, base_x + real_level.shape[0]],
                [base_y, base_y + real_level.shape[1]],
                [base_z, base_z + real_level.shape[2]],
            ]
            render_minecraft(opt.output_name, curr_coords, real_pth, "real_last_scale")
        except Exception as e:
            print("Render REAL failed:", repr(e))

    # samples
    for n in tqdm(range(num_samples), desc="sampling"):
        noise_list = _sample_noise_list_3d(
            opt=opt,
            gen_shapes=gen_shapes,
            noise_amp=noise_amp,
            fixed_noise=fixed_noise,
            gen_start_scale=gen_start_scale,
        )

        with torch.no_grad():
            sample = netG(noise_list, gen_shapes, noise_amp)

        # decode + save blockdata
        level = decode_repr_map_to_blocks(opt, sample.detach(), token_list)
        torch.save(level, os.path.join(dir2save, "torch_blockdata", f"{n}.pt"))

        if render_images:
            obj_pth = os.path.join(dir2save, "objects", "last")
            os.makedirs(obj_pth, exist_ok=True)
            try:
                subprocess.call(["/Applications/Wine Stable.app/Contents/Resources/wine/bin/wine", "--version"])

                len_n = math.ceil(math.sqrt(num_samples))
                xg, zg = np.unravel_index(n, [len_n, len_n])
                base_x = opt.coords[0][0]
                base_y = opt.coords[1][0]
                base_z = opt.coords[2][0]

                posx = base_x + xg * (level.shape[0] + 5)
                posz = base_z + zg * (level.shape[2] + 5)

                save_level_to_world(opt, (posx, base_y, posz), level)
                curr_coords = [
                    [posx, posx + level.shape[0]],
                    [base_y, base_y + level.shape[1]],
                    [posz, posz + level.shape[2]],
                ]
                render_minecraft(opt.output_name, curr_coords, obj_pth, f"{n}")
            except Exception as e:
                print("Render failed:", repr(e))

        if save_tensors:
            torch.save(sample.detach().cpu(), os.path.join(dir2save, "torch", f"{n}.pt"))

    return



if __name__ == '__main__':
    # NOTICE: The "output" dir is where the generator is located as with main.py, even though it is the "input" here

    opt = GenerateSamplesConfig().parse_args()

    clear_empty_world(opt.output_dir, opt.output_name)

    # Read level according to input arguments
    real = mc_read_level(opt)

    opt.map_shape = real.shape[2:]

    # Load Generator
    netG, fixed_noise, reals, noise_amp, last_depth = load_trained_pyramid_cons(opt)

    prefix = "arbitrary"

    # Directory name
    s_dir_name = "%s_random_samples_v%.5f_h%.5f_st%d" % (
        prefix, opt.scale_v, opt.scale_h, opt.gen_start_scale)

    generate_samples_cons(
        netG=netG,
        fixed_noise=fixed_noise,
        reals=reals,
        noise_amp=noise_amp,
        opt=opt,
        scale_v=opt.scale_v,
        scale_h=opt.scale_h,
        scale_d=opt.scale_d,
        gen_start_scale=opt.gen_start_scale,
        num_samples=opt.num_samples,
        render_images=True,
        save_tensors=opt.save_tensors,
        save_dir=s_dir_name,
    )
