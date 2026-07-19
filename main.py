import time
from pathlib import Path

from loguru import logger

from config import parse_args
from utils import archive_directory, init_logger, init_wandb
from generation.generate_samples import generate_samples_cons
from minecraft.level_utils import read_map
from training.train import train

def main():
    init_logger()
    opt = parse_args()
    init_wandb(opt)

    real = read_map(opt).to(opt.device)
    opt.map_shape = real.shape[2:]
    opt.save_hyperparameters()

    start_time = time.time()
    generators, noise_maps, reals, noise_amplitudes = train(real, opt)
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info("Time for training: {} seconds".format(elapsed_time))
    logger.info("Finished training! Generating random samples...")

    try:
        generate_samples_cons(
            netG=generators,
            fixed_noise=noise_maps,
            reals=reals,
            noise_amp=noise_amplitudes,
            opt=opt,
        )
    except Exception as e:
        logger.error(f"Failed to generate samples: {e}")

    clean_path = Path(opt.out_).parent.as_posix()
    archive_directory(clean_path, clean_path)


if __name__ == "__main__":
    main()
