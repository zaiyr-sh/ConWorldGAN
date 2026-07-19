import os

import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb
from loguru import logger
from tqdm import tqdm

from config import Config
from generation.generate_noise import generate_spatial_noise
from minecraft.layout_utils import calc_gradient_penalty_2d, semantic_to_layout2d
from minecraft.level_renderer import render_minecraft
from minecraft.level_utils import (
    decode_repr_map_to_blocks,
    save_level_to_world,
    repr_to_semantic_map,
)
from models import calc_gradient_penalty


def set_requires_grad(net, flag: bool):
    """Enable or disable gradient calculation for every parameter in a network."""

    if net is None:
        return

    for p in net.parameters():
        p.requires_grad_(flag)


def _get_train_depth(train_depth, netG) -> int:
    """Clamp the requested train depth to the generator's available stages."""

    return max(1, min(train_depth, len(netG.body)))


def sample_random_noise_3d(depth, reals_shapes, opt, hidden_channels: int):
    """Sample one 3D noise tensor for each generator stage through ``depth``."""

    noise = []
    eff = max(1, int(opt.num_layer))
    extra = eff * 2

    for d in range(depth + 1):
        Y, Z, X = reals_shapes[d][2], reals_shapes[d][3], reals_shapes[d][4]

        if d == 0:
            n = generate_spatial_noise(
                (1, int(opt.repr_channels), Y, Z, X),
                device=opt.device,
            ).detach()
        else:
            n = generate_spatial_noise(
                (1, int(hidden_channels), Y + extra, Z + extra, X + extra),
                device=opt.device,
            ).detach()

        noise.append(n)

    return noise


def build_z_opt_for_depth(depth, reals, reals_shapes, opt, hidden_channels: int):
    """Build the fixed reconstruction input for a generator depth."""

    eff = max(1, int(opt.num_layer))
    extra = eff * 2

    if depth == 0:
        z_opt = reals[0].detach()
    else:
        Y, Z, X = reals_shapes[depth][2], reals_shapes[depth][3], reals_shapes[depth][4]

        z_opt = generate_spatial_noise(
            (1, int(hidden_channels), Y + extra, Z + extra, X + extra),
            device=opt.device,
        ).detach()

    return z_opt


def train_single_scale(
    D, D_layout, G, reals, fixed_noise, noise_amp, opt: Config, depth
):
    """Train the generator and active discriminators at one pyramid scale."""

    reals_shapes = [r.shape for r in reals]
    real = reals[depth]

    alpha = max(
        float(opt.alpha_min),
        float(opt.alpha) * (float(opt.alpha_decay) ** depth),
    )

    noise_amp_init = float(opt.noise_amp)
    lr_scale = float(opt.lr_scale)
    hidden_channels = int(opt.hidden_channel)

    # Prepare the fixed reconstruction input and this scale's noise amplitude.
    z_opt = build_z_opt_for_depth(
        depth=depth,
        reals=reals,
        reals_shapes=reals_shapes,
        opt=opt,
        hidden_channels=hidden_channels,
    )

    fixed_noise.append(z_opt)

    if depth == 0:
        noise_amp.append(torch.tensor(1.0, device=opt.device))
    else:
        noise_amp.append(torch.tensor(0.0, device=opt.device))

        with torch.no_grad():
            z_reconstruction = G(fixed_noise, reals_shapes, noise_amp)
            rmse = torch.sqrt(F.mse_loss(z_reconstruction, real)).detach()
            noise_amp[-1] = torch.tensor(noise_amp_init, device=opt.device) * rmse

    # Create fresh optimizers for the networks trained at this scale.
    optimizerD = optim.Adam(
        D.parameters(),
        lr=opt.lr_d,
        betas=(opt.beta1, 0.999),
    )

    optimizerD_layout = None
    if D_layout is not None:
        optimizerD_layout = optim.Adam(
            D_layout.parameters(),
            lr=opt.lr_d,
            betas=(opt.beta1, 0.999),
        )

    # Freeze generator stages outside the configured progressive training window.
    train_depth = _get_train_depth(opt.train_depth, G)

    for stage in G.body[:-train_depth]:
        for p in stage.parameters():
            p.requires_grad = False

    for stage in G.body[-train_depth:]:
        for p in stage.parameters():
            p.requires_grad = True

    # Older trainable stages receive geometrically smaller learning rates.
    trainable_stages = list(G.body[-train_depth:])
    parameter_list = []

    for idx, stage in enumerate(trainable_stages):
        # The first item is the oldest trainable stage.
        k = len(trainable_stages) - 1 - idx
        parameter_list.append(
            {
                "params": stage.parameters(),
                "lr": opt.lr_g * (lr_scale**k),
            }
        )

    # Train the head only while the initial scales remain in the active window.
    if depth - train_depth < 0:
        parameter_list.append(
            {
                "params": G.head.parameters(),
                "lr": opt.lr_g * (lr_scale**depth),
            }
        )

    # The output tail always uses the base generator learning rate.
    parameter_list.append(
        {
            "params": G.tail.parameters(),
            "lr": opt.lr_g,
        }
    )

    optimizerG = optim.Adam(
        parameter_list,
        lr=opt.lr_g,
        betas=(opt.beta1, 0.999),
    )

    logger.info(
        f"[ConSinGAN-3D] Training depth={depth}, "
        f"train_depth={train_depth}, "
        f"noise_amp={float(noise_amp[-1])}"
    )

    # Semantic targets remain constant throughout this scale, so compute them once.
    with torch.no_grad():
        if D_layout is not None or opt.lambda_sem_rec > 0:
            real_sem = repr_to_semantic_map(
                opt, real, opt.token_list, tau=opt.semantic_tau
            )
        else:
            real_sem = None

        if D_layout is not None:
            real_layout = semantic_to_layout2d(real_sem).detach()
        else:
            real_layout = None

    for it in tqdm(range(opt.niter), desc=f"scale {depth}"):
        step = depth * opt.niter + it

        noise = sample_random_noise_3d(
            depth=depth,
            reals_shapes=reals_shapes,
            opt=opt,
            hidden_channels=hidden_channels,
        )

        # Train the representation discriminator.
        set_requires_grad(D, True)

        for j in range(opt.Dsteps):
            D.zero_grad(set_to_none=True)

            if j == opt.Dsteps - 1:
                fake = G(noise, reals_shapes, noise_amp)
            else:
                with torch.no_grad():
                    fake = G(noise, reals_shapes, noise_amp)

            real_d = real
            fake_d = fake.detach()

            out_real = D(real_d)
            errD_real = -out_real.mean()

            out_fake = D(fake_d)
            errD_fake = out_fake.mean()

            gp = calc_gradient_penalty(
                D,
                real_d,
                fake_d,
                opt.lambda_grad,
                opt.device,
            )

            errD_total = errD_real + errD_fake + gp
            errD_total.backward()
            optimizerD.step()

        # Train the optional 2D layout discriminator.
        if D_layout is not None:
            set_requires_grad(D_layout, True)
            D_layout.train()
            D_layout.zero_grad(set_to_none=True)

            with torch.no_grad():
                fake_for_layout = G(noise, reals_shapes, noise_amp)

                fake_sem_layout_d = repr_to_semantic_map(
                    opt, fake_for_layout.detach(), opt.token_list, tau=opt.semantic_tau
                )

                fake_layout_d = semantic_to_layout2d(fake_sem_layout_d).detach()

            errD_layout_real = -D_layout(real_layout).mean()
            errD_layout_fake = D_layout(fake_layout_d).mean()

            gp_layout = calc_gradient_penalty_2d(
                D_layout,
                real_layout,
                fake_layout_d,
                lambda_gp=opt.layout_gp_lambda,
                device=opt.device,
            )

            errD_layout = errD_layout_real + errD_layout_fake + gp_layout
            errD_layout.backward()
            optimizerD_layout.step()
        else:
            errD_layout_real = torch.tensor(0.0, device=opt.device)
            errD_layout_fake = torch.tensor(0.0, device=opt.device)
            gp_layout = torch.tensor(0.0, device=opt.device)

        # Train the generator while keeping both discriminators fixed.
        set_requires_grad(D, False)
        set_requires_grad(D_layout, False)

        for _ in range(opt.Gsteps):
            G.zero_grad(set_to_none=True)

            fake = G(noise, reals_shapes, noise_amp)

            errG_adv = -D(fake).mean()

            if D_layout is not None:
                fake_sem_g = repr_to_semantic_map(
                    opt, fake, opt.token_list, tau=opt.semantic_tau
                )
                fake_layout_g = semantic_to_layout2d(fake_sem_g)
                errG_layout_adv = -D_layout(fake_layout_g).mean()
            else:
                errG_layout_adv = torch.tensor(0.0, device=opt.device)

            if alpha != 0.0:
                rec = G(fixed_noise, reals_shapes, noise_amp)
                rec_loss = alpha * F.mse_loss(rec, real)

                if opt.lambda_sem_rec > 0:
                    rec_sem = repr_to_semantic_map(
                        opt, rec, opt.token_list, tau=opt.semantic_tau
                    )
                    sem_rec_loss = opt.lambda_sem_rec * F.l1_loss(
                        rec_sem, real_sem
                    )
                else:
                    sem_rec_loss = torch.tensor(0.0, device=opt.device)
            else:
                rec = None
                rec_loss = torch.tensor(0.0, device=opt.device)
                sem_rec_loss = torch.tensor(0.0, device=opt.device)

            errG_total = (
                opt.lambda_repr_adv * errG_adv
                + opt.lambda_layout_adv * errG_layout_adv
                + rec_loss
                + sem_rec_loss
            )

            errG_total.backward()
            optimizerG.step()

        set_requires_grad(D, True)
        set_requires_grad(D_layout, True)

        # Log the current losses without committing a separate W&B step.
        if step % 10 == 0:
            wandb.log(
                {
                    f"D_real@{depth}": (-errD_real).item(),
                    f"D_fake@{depth}": errD_fake.item(),
                    f"gp@{depth}": gp.item(),
                    f"G_adv@{depth}": errG_adv.item(),
                    f"rec_loss@{depth}": rec_loss.item(),
                    f"sem_rec_loss@{depth}": sem_rec_loss.item(),
                    f"noise_amp@{depth}": float(noise_amp[-1]),
                    f"D_layout_real@{depth}": (-errD_layout_real).item(),
                    f"D_layout_fake@{depth}": errD_layout_fake.item(),
                    f"gp_layout@{depth}": gp_layout.item(),
                    f"G_layout_adv@{depth}": errG_layout_adv.item(),
                    f"alpha_curr@{depth}": float(alpha),
                },
                step=step,
            )

        # Render the final real, generated, and reconstructed levels.
        if it == (opt.niter - 1):
            token_list = opt.token_list

            try:
                real_scaled = decode_repr_map_to_blocks(
                    opt,
                    real.detach(),
                    token_list,
                )

                fake_scaled = decode_repr_map_to_blocks(
                    opt,
                    fake.detach(),
                    token_list,
                )

                to_render = [real_scaled, fake_scaled]
                render_names = [f"real@{depth}", f"fake@{depth}"]

                if rec is not None:
                    rec_scaled = decode_repr_map_to_blocks(
                        opt,
                        rec.detach(),
                        token_list,
                    )

                    to_render.append(rec_scaled)
                    render_names.append(f"rec@{depth}")

                obj_pth = os.path.join(opt.out_, f"objects/{depth}")
                os.makedirs(obj_pth, exist_ok=True)

                for n, level in enumerate(to_render):
                    pos = n * (level.shape[0] + 5)

                    save_level_to_world(
                        opt,
                        (pos, 0, 0),
                        level,
                    )

                    curr_coords = [
                        [pos, pos + real_scaled.shape[0]],
                        [0, real_scaled.shape[1]],
                        [0, real_scaled.shape[2]],
                    ]

                    render_minecraft(
                        opt.output_name,
                        curr_coords,
                        obj_pth,
                        render_names[n],
                        opt,
                    )

            except Exception as e:
                logger.warning(f"Render failed at scale={depth}, it={it}: {e}")

    # Persist per-scale training state and discriminator weights.
    torch.save(fixed_noise, os.path.join(opt.outf, "fixed_noise.pth"))
    torch.save(noise_amp, os.path.join(opt.outf, "noise_amp.pth"))

    torch.save(fixed_noise[-1], os.path.join(opt.outf, "z_opt.pth"))

    if D_layout is not None:
        torch.save(
            D_layout.state_dict(),
            os.path.join(opt.outf, f"netD_layout_scale_{depth}.pth"),
        )

    return fixed_noise, noise_amp, G, D, D_layout
