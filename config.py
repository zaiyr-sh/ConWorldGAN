# Code based on https://github.com/tamarott/SinGAN
import random
from typing import List, Optional, Any
from constants import *

import torch
from torch import cuda
from tap import Tap
import os.path as osp
from utils import set_seed, load_pkl, get_subdir_path, load_pt


class Config(Tap):
    netG: str = ""  # path to netG (to continue training)
    netD: str = ""  # path to netD (to continue training)
    manualSeed: Optional[int] = 42
    out: str = "output"  # output directory
    input_dir: str = "%s/%s/" % (PROJECT_PATH, "minecraft_worlds")  # input directory
    input_name: str = "drehmal_3"  # input level filename

    # if minecraft is used, which coords are used from the world? Which world do we save to?
    input_area_name: str = "vanilla_village"  # needs to be a string from the coord dictionary in input folder
    output_dir: str = get_subdir_path("minecraft_worlds")
    output_name: str = "drehmal_test"  # name of the world to generate in
    sub_coords: List[float] = SUB_COORDS[input_area_name]  # defines which coords of the full coord are
    # taken (if float -> percentage, if int -> absolute)

    hidden_channel: int = 64  # number of filters for conv layers
    ker_size: int = 3  # kernel size for conv layers
    num_layer: int = 3  # number of layers
    scales: List[float] = [0.75, 0.5, 0.35] # Scales descending (< 1 and > 0)
    noise_update: float = 0.1  # additive noise weight
    props = []
    map_shape = []
    niter: int = 7000  # number of epochs to train per scale
    milestones: int = 0  # number of epochs to train per scale
    gamma: float = 0.1  # scheduler gamma
    lr_g: float = 0.0005  # generator learning rate
    lr_d: float = 0.0005  # discriminator learning rate
    beta1: float = 0.5  # optimizer beta
    Gsteps: int = 2  # generator inner steps
    Dsteps: int = 3  # discriminator inner steps
    lambda_grad: float = 0.1  # gradient penalty weight
    alpha: int = 150  # reconstruction loss weight
    activation: str = "lrelu"
    token_list: List[str] = None  # unique block tokens in the selected region
    debug: bool = False
    repr_type: str = "bert"  # Which representation type to use, currently [None, bert, clip]
    repr_dim: int = 57 # [32, 57, 768]
    padd_size: int = 0
    stride_size: int = 1
    lr_scale: float = 0.1
    train_depth: int = 1 # how many layers are trained if growing
    norm_layer = "instance" # [instance, group, batch]

    loss: str = "WGAN-GP"

    clip_type = "descriptions_and_images" # [descriptions, images, descriptions_and_images]
    neighbors_type = None # # Which neighbors type to use, currently [None, local, global]
    neighbor_info = None

    use_semantic_disc: bool = False
    semantic_channels: int = 6
    semantic_tau: float = 1.0
    lambda_sem_adv: float = 0.5
    lambda_sem_rec: float = 10.0

    alpha_decay: float = 0.30
    alpha_min: float = 10.0

    use_layout_disc: bool = True
    layout_channels: int = 5
    layout_nfc: int = 32
    lambda_layout_adv: float = 0.2
    layout_gp_lambda: float = 10.0

    lambda_repr_adv: float = 0.8

    def __init__(self,
                 *args,
                 underscores_to_dashes: bool = False,
                 explicit_bool: bool = False,
                 **kwargs):
        super().__init__(*args, underscores_to_dashes=underscores_to_dashes, explicit_bool=explicit_bool, **kwargs)

    def process_args(self):
        self.device = torch.device("cuda:0" if cuda.is_available() else "cpu")

        if self.manualSeed is None:
            self.manualSeed = random.randint(1, 10000)
        print("Random Seed: ", self.manualSeed)
        set_seed(self.manualSeed)

        # Defaults for other namespace values that will be overwritten during runtime
        self.repr_channels = 32
        self.out_ = None
        self.outf = "0"  # changes with each scale trained
        self.num_scales = len(self.scales) # number of scales is implicitly defined
        self.noise_amp = 1.0  # noise amp for lowest scale always starts at 1
        self.stop_scale = self.num_scales + 1 # which scale to stop on - usually always last scale defined

        # coord_dict = load_pkl('primordial_coords_dict', 'input/minecraft/')
        # tmp_coords = coord_dict[self.input_area_name]
        # tmp_coords = ((25165, 25286), (60, 73), (-770, -634)) # the largest village
        tmp_coords = ((25235, 25285), (60, 73), (-655, -634)) # small village
        # tmp_coords = ((25168, 25218), (60, 73), (-711, -688)) # small village 2
        # tmp_coords = ((25217, 25253), (60, 73), (-712, -678)) # small village 3
        # tmp_coords = ((25228, 25283), (60, 73), (-661, -633)) # small village 4
        # tmp_coords = ((25227, 25255), (60, 73), (-696, -678)) # small village 5
        # tmp_coords = ((25227, 25248), (60, 73), (-697, -653)) # another small village
        # tmp_coords = ((25181, 25251), (60, 73), (-713, -676)) # middle village
        sub_coords = [(self.sub_coords[0], self.sub_coords[1]),
                      (self.sub_coords[2], self.sub_coords[3]),
                      (self.sub_coords[4], self.sub_coords[5])]
        self.coords = []
        for i, (start, end) in enumerate(sub_coords):
            curr_len = tmp_coords[i][1] - tmp_coords[i][0]
            if isinstance(start, float):
                tmp_start = curr_len * start + tmp_coords[i][0]
                tmp_end = curr_len * end + tmp_coords[i][0]
            elif isinstance(start, int):
                tmp_start = tmp_coords[i][0] + start
                tmp_end = tmp_coords[i][0] + end
            else:
                AttributeError("Unexpected type for sub_coords")
                tmp_start = tmp_coords[i][0]
                tmp_end = tmp_coords[i][1] 

            self.coords.append((int(tmp_start), int(tmp_end)))

        if not self.repr_type:
            self.block2repr = None
        elif self.repr_type == "bert":
            # self.block2repr = load_pt(f"natural_representations_small_neighbors_no_norm_{self.repr_dim}", get_subdir_path(f"input/minecraft/{self.input_area_name}"))
            self.block2repr = load_pt(f"natural_representations_small_no_norm_{self.repr_dim}", get_subdir_path(f"input/minecraft/{self.input_area_name}"))
            # self.block2repr = load_pt(f"natural_representations", get_subdir_path(f"input/minecraft/{self.input_area_name}"))
        elif self.repr_type == "clip":
            self.block2repr = load_pt(f"clip_representations_small_{self.clip_type}_no_norm_{self.repr_dim}", get_subdir_path(f"input/minecraft/{self.input_area_name}"))
        else:
            AttributeError("unexpected repr_type, use [None, clip, bert]")

    def save_hyperparameters(self):
        hp_keys = list(type(self).__annotations__.keys())
        hp = {k: getattr(self, k) for k in hp_keys if hasattr(self, k)}
        with open(osp.join(self.out_, "parameters.txt"), "w") as f:
            for k in sorted(hp.keys()):
                f.write(f"{k}\t-\t{hp[k]}\n")