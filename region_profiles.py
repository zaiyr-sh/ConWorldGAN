"""Named coordinate and hyperparameter presets for reproducible experiments."""

from dataclasses import dataclass
from typing import Dict, Tuple

AxisBounds = Tuple[int, int]
RegionBounds = Tuple[AxisBounds, AxisBounds, AxisBounds]

@dataclass(frozen=True)
class RegionProfile:
    """Values that should change together when selecting an experiment region."""

    coords: RegionBounds
    hidden_channel: int = 64
    niter: int = 7000
    description: str = ""


DEFAULT_REGION = "village1"

# Coordinates use the same three-axis order expected by ``opt.coords`` throughout
# the existing Minecraft loading and generation code. Set ``hidden_channel`` or
# ``niter`` on an individual entry whenever that experiment differs from the
# defaults declared by ``RegionProfile``.
REGION_PROFILES: Dict[str, RegionProfile] = {
    "village_large": RegionProfile(
        coords=((25165, 25286), (60, 73), (-770, -634)),
        description="Largest village crop",
    ),
    "village1": RegionProfile(
        coords=((25235, 25285), (60, 73), (-655, -634)),
        description="Small village crop 1",
    ),
    "village2": RegionProfile(
        coords=((25168, 25218), (60, 73), (-711, -688)),
        description="Small village crop 2",
    ),
    "village3": RegionProfile(
        coords=((25217, 25253), (60, 73), (-712, -678)),
        description="Small village crop 3",
        hidden_channel=128,
        niter=2850
    ),
}
