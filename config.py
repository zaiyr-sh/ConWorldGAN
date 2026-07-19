"""Command-line configuration for training and generation experiments."""

from __future__ import annotations

import argparse
import os.path as osp
import random
from collections.abc import Sequence
from typing import Any, Optional, Protocol

import torch

from constants import PROJECT_PATH, SUB_COORDS
from region_profiles import DEFAULT_REGION, REGION_PROFILES, RegionBounds
from utils import get_subdir_path, load_pt, set_seed


def _optional_int(value: str) -> Optional[int]:
    """Parse an integer or ``none`` for a randomly generated seed."""

    if value.lower() == "none":
        return None
    try:
        return int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected an integer or 'none'") from error


def _coordinate_component(value: str) -> int | float:
    """Preserve integers as offsets and decimals as relative coordinates."""

    try:
        if any(marker in value.lower() for marker in (".", "e")):
            return float(value)
        return int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "coordinates must be integer offsets or decimal fractions"
        ) from error


class _ArgumentContainer(Protocol):
    """Common interface shared by parsers and argument groups."""

    def add_argument(self, *args: Any, **kwargs: Any) -> argparse.Action: ...


def _add_argument(
    parser: _ArgumentContainer,
    name: str,
    *legacy_names: str,
    **kwargs: Any,
) -> None:
    """Add a dashed option and any backwards-compatible aliases."""

    kwargs.setdefault("dest", name.replace("-", "_"))
    parser.add_argument(f"--{name}", *legacy_names, **kwargs)


class Config(argparse.Namespace):
    """Parsed project settings plus values derived during initialization.

    Instances should be created with :func:`parse_args`; the annotations make
    the mutable namespace easier to understand in modules that receive it.
    """

    netG: str
    netD: str
    manualSeed: Optional[int]
    out: str
    input_dir: str
    input_name: str
    input_area_name: str
    output_dir: str
    output_name: str
    region: str
    sub_coords: list[int | float]

    hidden_channel: int
    ker_size: int
    num_layer: int
    scales: list[float]
    noise_update: float
    niter: int
    milestones: int
    gamma: float
    lr_g: float
    lr_d: float
    beta1: float
    Gsteps: int
    Dsteps: int
    lambda_grad: float
    alpha: int
    activation: str
    debug: bool
    repr_type: Optional[str]
    repr_dim: int
    padd_size: int
    stride_size: int
    lr_scale: float
    train_depth: int
    norm_layer: str
    loss: str
    clip_type: str
    neighbors_type: Optional[str]
    use_semantic_disc: bool
    semantic_channels: int
    semantic_tau: float
    lambda_sem_adv: float
    lambda_sem_rec: float
    alpha_decay: float
    alpha_min: float
    use_layout_disc: bool
    layout_channels: int
    layout_nfc: int
    lambda_layout_adv: float
    layout_gp_lambda: float
    lambda_repr_adv: float

    device: torch.device
    coords: RegionBounds
    block2repr: Optional[dict]
    token_list: Optional[list[str]]
    neighbor_info: Any
    props: list
    map_shape: list
    repr_channels: int
    out_: Optional[str]
    outf: str
    num_scales: int
    noise_amp: float
    stop_scale: int
    _hyperparameter_keys: list[str]

    def save_hyperparameters(self) -> None:
        """Write the reproducible command-line settings for the current run."""

        if not self.out_:
            raise RuntimeError("The run output directory has not been initialized")

        keys = set(self._hyperparameter_keys)
        keys.update({
            "coords",
            "token_list",
            "repr_channels",
            "num_scales",
            "stop_scale",
        })
        with open(osp.join(self.out_, "parameters.txt"), "w", encoding="utf-8") as file:
            for key in sorted(keys):
                file.write(f"{key}\t-\t{getattr(self, key)}\n")


def build_parser(description: Optional[str] = None) -> argparse.ArgumentParser:
    """Build the base command-line parser used by project entry points."""

    parser = argparse.ArgumentParser(
        description=description or "Train ConWorldGAN on a selected Minecraft region.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    data = parser.add_argument_group("world and region")
    _add_argument(
        data,
        "input-dir",
        "--input_dir",
        default=osp.join(PROJECT_PATH, "minecraft_worlds"),
        help="directory containing input worlds",
    )
    _add_argument(
        data,
        "input-name",
        "--input_name",
        default="Drehmal v2.1 PRIMORDIAL",
        help="input Minecraft world name",
    )
    _add_argument(
        data,
        "input-area-name",
        "--input_area_name",
        default="vanilla_village",
        help="representation-data subdirectory and subregion preset name",
    )
    _add_argument(
        data,
        "output-dir",
        "--output_dir",
        default=get_subdir_path("minecraft_worlds"),
        help="directory containing the output world",
    )
    _add_argument(
        data,
        "output-name",
        "--output_name",
        default="drehmal_generated",
        help="Minecraft world used for generated samples",
    )
    _add_argument(
        data,
        "region",
        choices=sorted(REGION_PROFILES),
        default=DEFAULT_REGION,
        help="named experiment profile providing base coordinates and related defaults",
    )
    _add_argument(
        data,
        "sub-coords",
        "--sub_coords",
        nargs=6,
        type=_coordinate_component,
        metavar=("Y0", "Y1", "Z0", "Z1", "X0", "X1"),
        default=None,
        help="crop within the region: decimal fractions or integer offsets",
    )

    paths = parser.add_argument_group("checkpoints and output")
    _add_argument(
        paths,
        "net-g",
        "--netG",
        "--net_g",
        dest="netG",
        default="",
        help="generator checkpoint used to continue training",
    )
    _add_argument(
        paths,
        "net-d",
        "--netD",
        "--net_d",
        dest="netD",
        default="",
        help="discriminator checkpoint used to continue training",
    )
    _add_argument(paths, "out", default="output", help="experiment output root")

    model = parser.add_argument_group("model")
    _add_argument(
        model,
        "hidden-channel",
        "--hidden_channel",
        type=int,
        default=None,
        help="convolution channel count; defaults to the selected region profile",
    )
    _add_argument(
        model,
        "kernel-size",
        "--ker_size",
        dest="ker_size",
        type=int,
        default=3,
        help="convolution kernel size",
    )
    _add_argument(
        model,
        "num-layer",
        "--num_layer",
        type=int,
        default=3,
        help="number of convolution blocks",
    )
    _add_argument(
        model,
        "padding-size",
        "--padd_size",
        dest="padd_size",
        type=int,
        default=0,
        help="convolution padding",
    )
    _add_argument(
        model,
        "stride-size",
        "--stride_size",
        type=int,
        default=1,
        help="convolution stride",
    )
    _add_argument(model, "activation", default="lrelu", help="activation function")
    _add_argument(
        model,
        "norm-layer",
        "--norm_layer",
        choices=("instance", "group", "batch"),
        default="instance",
        help="normalization layer",
    )

    representation = parser.add_argument_group("block representation")
    _add_argument(
        representation,
        "repr-type",
        "--repr_type",
        choices=("none", "bert", "clip"),
        default="bert",
        help="block representation; 'none' uses discrete blocks",
    )
    _add_argument(
        representation,
        "repr-dim",
        "--repr_dim",
        type=int,
        default=57,
        help="representation vector size",
    )
    _add_argument(
        representation,
        "clip-type",
        "--clip_type",
        choices=("descriptions", "images", "descriptions_and_images"),
        default="descriptions_and_images",
        help="CLIP input type",
    )
    _add_argument(
        representation,
        "neighbors-type",
        "--neighbors_type",
        choices=("local", "global"),
        default=None,
        help="optional neighboring-block context",
    )

    training = parser.add_argument_group("training")
    _add_argument(
        training,
        "seed",
        "--manualSeed",
        "--manual-seed",
        "--manual_seed",
        dest="manualSeed",
        type=_optional_int,
        default=42,
        help="random seed, or 'none' to generate one",
    )
    _add_argument(
        training,
        "scales",
        nargs="+",
        type=float,
        default=[0.75, 0.5, 0.35],
        help="descending scale factors between zero and one",
    )
    _add_argument(
        training,
        "noise-update",
        "--noise_update",
        type=float,
        default=0.1,
        help="additive noise weight",
    )
    _add_argument(
        training,
        "niter",
        type=int,
        default=None,
        help="iterations per scale; defaults to the selected region profile",
    )
    _add_argument(
        training,
        "milestones",
        type=int,
        default=0,
        help="learning-rate scheduler milestone",
    )
    _add_argument(
        training, "gamma", type=float, default=0.1, help="learning-rate scheduler gamma"
    )
    _add_argument(
        training,
        "lr-g",
        "--lr_g",
        type=float,
        default=0.0005,
        help="generator learning rate",
    )
    _add_argument(
        training,
        "lr-d",
        "--lr_d",
        type=float,
        default=0.0005,
        help="discriminator learning rate",
    )
    _add_argument(training, "beta1", type=float, default=0.5, help="Adam beta1")
    _add_argument(
        training,
        "g-steps",
        "--Gsteps",
        dest="Gsteps",
        type=int,
        default=2,
        help="generator updates per iteration",
    )
    _add_argument(
        training,
        "d-steps",
        "--Dsteps",
        dest="Dsteps",
        type=int,
        default=3,
        help="discriminator updates per iteration",
    )
    _add_argument(
        training,
        "lambda-grad",
        "--lambda_grad",
        type=float,
        default=0.1,
        help="gradient penalty weight",
    )
    _add_argument(
        training, "alpha", type=int, default=150, help="reconstruction loss weight"
    )
    _add_argument(
        training,
        "alpha-decay",
        "--alpha_decay",
        type=float,
        default=0.30,
        help="per-scale reconstruction-loss decay",
    )
    _add_argument(
        training,
        "alpha-min",
        "--alpha_min",
        type=float,
        default=10.0,
        help="minimum reconstruction-loss weight",
    )
    _add_argument(
        training,
        "lr-scale",
        "--lr_scale",
        type=float,
        default=0.1,
        help="learning-rate multiplier for earlier stages",
    )
    _add_argument(
        training,
        "train-depth",
        "--train_depth",
        type=int,
        default=1,
        help="number of growing-generator stages to train",
    )
    _add_argument(training, "loss", default="WGAN-GP", help="adversarial loss name")
    _add_argument(
        training,
        "debug",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="enable debug output",
    )

    semantic = parser.add_argument_group("semantic and layout losses")
    _add_argument(
        semantic,
        "use-semantic-disc",
        "--use_semantic_disc",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="enable the semantic discriminator",
    )
    _add_argument(
        semantic, "semantic-channels", "--semantic_channels", type=int, default=6
    )
    _add_argument(semantic, "semantic-tau", "--semantic_tau", type=float, default=1.0)
    _add_argument(
        semantic, "lambda-sem-adv", "--lambda_sem_adv", type=float, default=0.5
    )
    _add_argument(
        semantic, "lambda-sem-rec", "--lambda_sem_rec", type=float, default=10.0
    )
    _add_argument(
        semantic,
        "use-layout-disc",
        "--use_layout_disc",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable the layout discriminator",
    )
    _add_argument(semantic, "layout-channels", "--layout_channels", type=int, default=5)
    _add_argument(semantic, "layout-nfc", "--layout_nfc", type=int, default=32)
    _add_argument(
        semantic, "lambda-layout-adv", "--lambda_layout_adv", type=float, default=0.2
    )
    _add_argument(
        semantic, "layout-gp-lambda", "--layout_gp_lambda", type=float, default=10.0
    )
    _add_argument(
        semantic, "lambda-repr-adv", "--lambda_repr_adv", type=float, default=0.8
    )

    return parser


def _resolve_subregion(
    base_coords: RegionBounds,
    sub_coords: Sequence[int | float],
    parser: argparse.ArgumentParser,
) -> RegionBounds:
    """Resolve fractional crops or absolute offsets within base coordinates."""

    if len(sub_coords) != 6:
        parser.error("sub-coords requires exactly six values")

    pairs = tuple(zip(sub_coords[::2], sub_coords[1::2]))
    resolved: list[tuple[int, int]] = []

    for axis, ((base_start, base_end), (start, end)) in enumerate(
        zip(base_coords, pairs)
    ):
        if type(start) is not type(end):
            parser.error(
                f"sub-coords pair {axis + 1} must use either two integers or two decimals"
            )
        if start > end:
            parser.error(
                f"sub-coords pair {axis + 1} must be ordered from start to end"
            )

        if isinstance(start, float):
            if not 0.0 <= start <= 1.0 or not 0.0 <= end <= 1.0:
                parser.error("decimal sub-coords must be between 0.0 and 1.0")
            axis_length = base_end - base_start
            resolved.append(
                (
                    int(base_start + axis_length * start),
                    int(base_start + axis_length * end),
                )
            )
        else:
            resolved.append((base_start + start, base_start + end))

    return resolved[0], resolved[1], resolved[2]


def _load_block_representations(config: Config) -> Optional[dict]:
    """Load the representation table selected by the command-line settings."""

    if config.repr_type is None:
        return None

    representation_dir = get_subdir_path(f"input/minecraft/{config.input_area_name}")
    if config.repr_type == "bert":
        # filename = f"natural_representations_small_{config.repr_dim}"
        filename = f"natural_representations_small_no_norm_{config.repr_dim}"
    else:
        filename = (
            f"clip_representations_small_{config.clip_type}_no_norm_{config.repr_dim}"
        )
    return load_pt(filename, representation_dir)


def _initialize_config(config: Config, parser: argparse.ArgumentParser) -> Config:
    """Apply profile defaults and initialize values derived from parsed options."""

    profile = REGION_PROFILES[config.region]
    if config.hidden_channel is None:
        config.hidden_channel = profile.hidden_channel
    if config.niter is None:
        config.niter = profile.niter
    if config.hidden_channel <= 0:
        parser.error("hidden-channel must be greater than zero")
    if config.niter <= 0:
        parser.error("niter must be greater than zero")

    if config.input_area_name not in SUB_COORDS and config.sub_coords is None:
        parser.error(
            f"unknown input-area-name '{config.input_area_name}'; provide --sub-coords "
            f"or choose one of: {', '.join(sorted(SUB_COORDS))}"
        )
    if config.sub_coords is None:
        config.sub_coords = list(SUB_COORDS[config.input_area_name])
    config.coords = _resolve_subregion(profile.coords, config.sub_coords, parser)

    if config.repr_type == "none":
        config.repr_type = None

    config.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if config.manualSeed is None:
        config.manualSeed = random.randint(1, 10000)
    print("Random Seed:", config.manualSeed)
    set_seed(config.manualSeed)

    # These values are populated or updated later in the training pipeline.
    config.token_list = None
    config.neighbor_info = None
    config.props = []
    config.map_shape = []
    config.out_ = None
    config.outf = "0"
    config.num_scales = len(config.scales)
    config.noise_amp = 1.0
    config.stop_scale = config.num_scales + 1
    config.block2repr = _load_block_representations(config)
    return config


def parse_args(
    args: Optional[Sequence[str]] = None,
    *,
    parser: Optional[argparse.ArgumentParser] = None,
) -> Config:
    """Parse command-line arguments and return an initialized configuration."""

    parser = parser or build_parser()
    config = parser.parse_args(args=args, namespace=Config())
    config._hyperparameter_keys = list(vars(config))
    return _initialize_config(config, parser)
