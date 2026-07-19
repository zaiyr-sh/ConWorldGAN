# ConWorldGAN

ConWorldGAN is a research prototype for generating 3D Minecraft voxel worlds from a single example region. It extends the Wor(l)d-GAN / World-GAN single-example generation pipeline.

## Attribution

This codebase is based on the public Wor(l)d-GAN / World-GAN implementation by Maren Awiszus, Frederik Schubert, and Bodo Rosenhahn. Their original work introduced GAN-based single-example generation for Minecraft worlds using 3D convolutions and block embeddings.

The original code was released under the MIT License. This repository preserves the original license and copyright notice, and the modifications in ConWorldGAN are intended for research purposes.

If you use this repository, please cite both ConWorldGAN and the original Wor(l)d-GAN work. Citation metadata is provided in [`CITATION.cff`](CITATION.cff).

## Local setup

ConWorldGAN targets Python 3.9. Create a virtual environment and install the
runtime dependencies with:

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

World saves, representation tables, experiment outputs, and Mineways/Wine
executables are developer-specific. Copy `.env.example` to `.env`, set the
values needed on your machine, and export them before running the project:

```bash
set -a
source .env
set +a
```

All path settings can also be supplied as command-line options; run
`python main.py --help` for the full list. Command-line values override the
environment-backed defaults.

## Required world data and tools

### Drehmal world

Training uses the **Drehmal v2.1 PRIMORDIAL** Minecraft world. Download the
archived v2.1 world from the [Drehmal v2.1 wiki page](https://wiki.drehmal.cyou/Misc/Versions/Drehmal_v21/),
extract it, and keep the extracted directory name as
`Drehmal v2.1 PRIMORDIAL`. The configured input directory must contain the
world folder itself:

```text
<CONWORLDGAN_INPUT_WORLDS_DIR>/
└── Drehmal v2.1 PRIMORDIAL/
    ├── level.dat
    └── region/
```

### Mineways

Download and extract the Windows version of
[Mineways](https://www.realtimerendering.com/erich/minecraft/public/mineways/),
then set `CONWORLDGAN_MINEWAYS_BIN` in `.env` to the absolute path of its
`Mineways.exe`. Mineways converts Minecraft regions to Wavefront OBJ files;
these files are required for the Blender visualization described below.

Mineways is a Windows application. On macOS or Linux, first download and
install [Wine from WineHQ](https://www.winehq.org/); ConWorldGAN uses Wine to
run `Mineways.exe`. Confirm that Wine is available before training:

```bash
wine --version
```

Then set the Wine, Mineways, and script-directory paths for your machine:

```dotenv
CONWORLDGAN_WINE_BIN=/absolute/path/to/wine
CONWORLDGAN_MINEWAYS_BIN=/absolute/path/to/Mineways.exe
CONWORLDGAN_MINEWAYS_SCRIPT_DIR=/absolute/path/to/ConWorldGAN/minecraft/mineways
```

The script directory must contain `close.mwscript`. Mineways is not a Python
dependency and should not be added to `requirements.txt` or committed to the
repository.

## Generate BERT embeddings before training

Set `CONWORLDGAN_INPUT_WORLDS_DIR` and `CONWORLDGAN_REPR_DIR` in `.env`, export
the file as shown above, and run the embedding script before starting any
training with `--repr-type bert`:

```bash
python bert_embeddings_experiment.py --region village_large
```

The embedding script downloads `bert-base-uncased` on its first run and writes
the generated `.pt` files under
`<CONWORLDGAN_REPR_DIR>/vanilla_village/`. Training with `--repr-type bert`
expects those files to exist and uses the representation dimension selected by
`--repr-dim` (57 by default).

After the representation files have been created, start training:

```bash
python main.py --region village1 --repr-type bert
```

After training, `main.py` automatically generates samples. When Mineways is
configured correctly, their OBJ files are written inside the W&B run directory,
normally under a path similar to:

```text
<CONWORLDGAN_RUNS_DIR>/wandb/<run>/files/random_samples/objects/last/0.obj
```

## Create a 360-degree video in Blender

After training and Mineways OBJ export, download and install
[Blender](https://www.blender.org/download/). To render an orbit video of a
generated sample:

1. Open Blender and switch to the **Scripting** workspace.
2. Open or paste [`scripts/visualization/blender_video_record.py`](scripts/visualization/blender_video_record.py)
   into the Blender text editor.
3. In the script's **USER SETTINGS** section, set `obj_path` to the absolute
   path of a generated OBJ such as `objects/last/0.obj` and set `output_video`
   to the desired absolute `.mp4` path.
4. Click **Run Script**. The script imports the OBJ, creates a camera and
   lighting, and renders a near-360-degree orbit as an H.264 MP4 video.

The default animation contains 240 frames at 24 FPS, producing a ten-second
video. Rendering can take some time depending on the selected resolution and
hardware.

Do not commit `.env` or a W&B API key. W&B credentials should normally be
stored with `wandb login`; `WANDB_ENTITY`, `WANDB_PROJECT`, and
`WANDB_MODE=offline` are safe per-developer environment settings. Local paths
and world names are deliberately excluded from the saved hyperparameter file.

## Minecraft disclaimer

This project uses Minecraft-style voxel data for research purposes only. It is not affiliated with, endorsed by, or sponsored by Mojang or Microsoft.
