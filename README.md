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
python main.py --region village1
```

All path settings can also be supplied as command-line options; run
`python main.py --help` for the full list. Command-line values override the
environment-backed defaults.

Do not commit `.env` or a W&B API key. W&B credentials should normally be
stored with `wandb login`; `WANDB_ENTITY`, `WANDB_PROJECT`, and
`WANDB_MODE=offline` are safe per-developer environment settings. Local paths
and world names are deliberately excluded from the saved hyperparameter file.

## Minecraft disclaimer

This project uses Minecraft-style voxel data for research purposes only. It is not affiliated with, endorsed by, or sponsored by Mojang or Microsoft.
