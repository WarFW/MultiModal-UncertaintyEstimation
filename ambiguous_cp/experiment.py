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
sys.pat