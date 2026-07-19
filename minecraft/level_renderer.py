import subprocess
from pathlib import Path

from constants import (
    DEFAULT_MINEWAYS_EXECUTABLE,
    DEFAULT_MINEWAYS_SCRIPT_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_WINE_EXECUTABLE,
)

def make_render_script(
    script_dir, script_name, obj_dir, obj_name, world_name, coords
) -> Path:
    """Write one Mineways export script and return its path."""

    script_dir = Path(script_dir)
    obj_dir = Path(obj_dir)
    script_dir.mkdir(parents=True, exist_ok=True)
    obj_dir.mkdir(parents=True, exist_ok=True)
    script_path = script_dir / f"{script_name}.mwscript"

    with script_path.open("w", encoding="utf-8") as file:
        file.write(f"Save Log file: {script_dir / (script_name + '.log')}\n")
        file.write("Set render type: Wavefront OBJ absolute indices\n")
        file.write(f"Minecraft world: {world_name}\n")
        file.write('Selection location min to max: {}, {}, {} to {}, {}, {}\n'.format(
            coords[0][0], coords[1][0], coords[2][0],
            coords[0][1] - 1, coords[1][1] - 1, coords[2][1] - 1
        ))
        file.write("Scale model by making each block 100 cm high\n")
        file.write(f"Export for Rendering: {obj_dir / (obj_name + '.obj')}")

    return script_path


def make_obj(
    script_dir,
    script_names,
    world_dir=DEFAULT_OUTPUT_DIR,
    wine_executable=DEFAULT_WINE_EXECUTABLE,
    mineways_executable=DEFAULT_MINEWAYS_EXECUTABLE,
):
    """Run Mineways for a collection of scripts using local tool paths."""

    script_dir = Path(script_dir)
    commands = [
        str(wine_executable),
        str(mineways_executable),
        "-m",
        "-s",
        str(world_dir),
    ]
    commands.extend(str(script_dir / f"{name}.mwscript") for name in script_names)

    process = subprocess.Popen(commands,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               universal_newlines=True)

    stdout, stderr = process.communicate()
    print(stdout)
    print(stderr)


def render_minecraft(world_name, coords_to_read, obj_dir, obj_name, opt=None):
    """Export a Minecraft region to OBJ using the configured local Mineways setup."""

    script_dir = (
        opt.mineways_script_dir if opt is not None else DEFAULT_MINEWAYS_SCRIPT_DIR
    )
    make_render_script(
        script_dir, obj_name, obj_dir, obj_name, world_name, coords_to_read
    )
    make_obj(
        script_dir,
        [obj_name, "close"],
        world_dir=opt.output_dir if opt is not None else DEFAULT_OUTPUT_DIR,
        wine_executable=(
            opt.wine_executable if opt is not None else DEFAULT_WINE_EXECUTABLE
        ),
        mineways_executable=(
            opt.mineways_executable
            if opt is not None
            else DEFAULT_MINEWAYS_EXECUTABLE
        ),
    )
    return str(Path(obj_dir) / f"{obj_name}.obj")
