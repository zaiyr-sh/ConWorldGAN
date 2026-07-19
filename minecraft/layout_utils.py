import torch


SEM_GROUND = 1
SEM_LIQUID = 2
SEM_FOLIAGE = 3
SEM_STRUCTURE = 4
SEM_DECOR = 5


def _soft_or_height(x: torch.Tensor, vertical_dim: int) -> torch.Tensor:
    """
    x: semantic probability map for one class

    If x shape is (B, 1, X, Y, Z), vertical_dim should be 3.
    If x shape is (B, 1, Y, Z, X), vertical_dim should be 2.
    """
    x = x.clamp(0.0, 1.0)
    out = 1.0 - torch.prod(1.0 - x, dim=vertical_dim)
    return out.clamp(0.0, 1.0)


def semantic_to_layout2d(
    sem: torch.Tensor,
    vertical_dim: int = 3,
) -> torch.Tensor:
    """
    Convert 3D semantic map to 2D top-down layout.

    If sem is (B, 6, X, Y, Z), use vertical_dim=3.
    Output will be (B, 5, X, Z).

    channels:
        0 = STRUCTURE footprint
        1 = GROUND footprint
        2 = LIQUID footprint
        3 = FOLIAGE footprint
        4 = DECOR footprint
    """
    ground = sem[:, SEM_GROUND:SEM_GROUND + 1]
    liquid = sem[:, SEM_LIQUID:SEM_LIQUID + 1]
    foliage = sem[:, SEM_FOLIAGE:SEM_FOLIAGE + 1]
    structure = sem[:, SEM_STRUCTURE:SEM_STRUCTURE + 1]
    decor = sem[:, SEM_DECOR:SEM_DECOR + 1]

    structure_fp = _soft_or_height(structure, vertical_dim=vertical_dim)
    ground_fp = _soft_or_height(ground, vertical_dim=vertical_dim)
    liquid_fp = _soft_or_height(liquid, vertical_dim=vertical_dim)
    foliage_fp = _soft_or_height(foliage, vertical_dim=vertical_dim)
    decor_fp = _soft_or_height(decor, vertical_dim=vertical_dim)

    layout = torch.cat(
        [
            structure_fp,
            ground_fp,
            liquid_fp,
            foliage_fp,
            decor_fp,
        ],
        dim=1,
    )

    return layout.contiguous()


def calc_gradient_penalty_2d(
    D,
    real_data: torch.Tensor,
    fake_data: torch.Tensor,
    lambda_gp: float = 10.0,
    device=None,
) -> torch.Tensor:
    """
    WGAN-GP gradient penalty for 2D discriminator.

    real_data: (B, C, D, W)
    fake_data: (B, C, D, W)
    """
    if device is None:
        device = real_data.device

    batch_size = real_data.size(0)

    alpha = torch.rand(
        batch_size,
        1,
        1,
        1,
        device=device,
        dtype=real_data.dtype,
    )

    interpolates = alpha * real_data + (1.0 - alpha) * fake_data
    interpolates.requires_grad_(True)

    disc_interpolates = D(interpolates)

    grad_outputs = torch.ones_like(disc_interpolates, device=device)

    gradients = torch.autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(batch_size, -1)

    gradient_penalty = (
        (gradients.norm(2, dim=1) - 1.0) ** 2
    ).mean() * lambda_gp

    return gradient_penalty
