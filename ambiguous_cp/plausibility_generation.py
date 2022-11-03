
import sys
import os

from pathlib import Path
import pandas as pd
import scipy.special as sp
import numpy as np
from PIL import Image
import torch
import open_clip
import json
import argparse

script_path = Path(os.path.dirname(os.path.abspath(sys.argv[0])))
base_path = script_path.parent.absolute()
sys.path.append(base_path + '\\cp')
sys.path.append(base_path + '\\utils')
from pets_classes import PETS_CLASSES, PETS_GENERIC_CLASSES
from fitz17k_classes import FITZ17K_CLASSES, FITZ17K_GENERIC_CLASSES
from medmnist_classes import MEDMNIST_CLASSES, MEDMNIST_GENERIC_CLASSES

device = "cuda" if torch.cuda.is_available() else "cpu"
print("CUDA ENABLED: {}".format(str(torch.cuda.is_available())))

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str, help='Experiment in experiment_configs to run')
args = parser.parse_args()
