import sys
import os

from pathlib import Path
import pandas as pd
import scipy as sp
import numpy as np
from PIL import Image
import torch
import json
import open_clip
from transformers import AutoTokenizer, CLIPTextModelWithProjection, CLIPModel, CLIPProcessor
import argparse

script_path = Path(os.path.dirname(os.path.abspath(sys.argv[0])))
base_path = script_path.parent.absolute()
sys.path.append(base_path + '\\cp')
sys.path.append(base_path + '\\utils')
from pets_classes import PETS_CLASSES, PETS_GENERIC_CLASSES
from fitz17k_classes import FITZ17K_CLASSES, FITZ17K_GENERIC_CLASSES
from medmnist_classes import MEDMNIST_CLASSES, MEDMNIST_GENERIC_CLASSES
from confo