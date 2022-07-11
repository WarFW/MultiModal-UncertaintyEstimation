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
from conformal_prediction_methods import *
from metrics import *

#Parse Arguments
#-----------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str, help='Experiment in experiment_configs to run')
args = parser.parse_args()

#Parameters
#-----------------------------------------------------------------------------------
reader = open(base_path + "\\experiment_configs\\"  + args.exp)
config = json.load(reader)
TEST_IMAGE_DIRECTORY = config["test_image_directory"]
IMAGE_PLAUSIBILITIES = config["intermediate_data_directory"]
RESULTS_DIRECTORY = config["results_data_directory"]
CLASSIFICATION_CHECKPOINT = config["classification_checkpoint"]
if config["dataset"] == 'MedMNIST':
    LABELS = MEDMNIST_CLASSES
elif config["dataset"] == 'FitzPatrick17k':
    LABELS = FITZ17K_CLASSES
else:
    LABELS = None
ALPHA = 0.05
NUM_SAMPLES = 1000
USE_SOFTMAX = True
LOGIT_SCALE = 100.0 if USE_SOFTMAX else 1.0
MODEL_ID = CLASSIFICATION_CHECKPOINT

#Model Methods
#-----------------------------------------------------------------------------------
def openclip_image_preprocess(image):
    image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad(), torch.cuda.amp.autocast():
        image_logits = model.encode_image(image)
        image_logits /= image_logits.norm(dim=-1, keepdim=True)
    return image_logits.to("cpu")

def openclip_text_preprocess(text):
    text = tokenizer(text).to(device)
    with torch.no_grad(), torch.cuda.amp.autocast():
        text_logits = model.encode_text(text)
        text_logits /= text_logits.norm(dim=-1, keepdim=True)
    return text_logits.to("cpu")

def openclip_process(image_logits, text_logits):
    image_logits, text_logits = image_logits.type(torch.float32), text_logits.type(torch.float32)
    return (LOGIT_SCALE * image_logits @ text_logits.T).softmax(dim=-1)[0]

def performance_report(threshold):
    # Get prediction sets
    calib_prediction_set = compute_prediction_sets_threshold(calib_sim_score_arr, threshold)
    test_prediction_set = compute_prediction_sets_threshold(test_sim_score_arr, threshold)
    # Compute performance metrics
    calib_coverage = overall_coverage(calib_prediction_set, calib_true_class_arr)
    test_coverage = overall_coverage(test_prediction_set, test_true_class_arr)
    calib_samplewise_efficiency = samplewise_efficiency(calib_prediction_set, calib_true_class_arr)
    test_samplewise_efficiency = samplewise_efficiency(test_prediction_set, test_true_class_arr)
    # Output Performance Metrics
    print(f"OVERALL COVERAGE (proportion of true labels covered):")
    print(f"Calibration Set: {calib_coverage}")
    print(f"Test Set: {test_coverage}")
    print(f'OVERALL EFFICIENCY (mean num of extraneous classes per sample): ')
    print(f"Calibration Set: {np.mean(calib_samplewise_efficiency)}")
    print(f"Test Set: {np.mean(test_samplewise_efficiency)}")

#Initialize Model
#-----------------------------------------------------------------------------------
print("Initializing Models")
device = "cuda" if torch.cuda.is_available() else "cpu"
print("CUDA ENABLED: {}".format(str(torch.cuda.is_available())))
model, _, preprocess = open_clip.create_model_and_transforms(MODEL_ID)
tokenizer = open_clip.get_tokenizer(MODEL_ID)
model.to(device)

#Generate Label Logits
#-----------------------------------------------------------------------------------
print("Preprocess Text Prompts")
PROMPT_GENERATOR = lambda cls : f"{cls}."
prompt_list = [PROMPT_GENERATOR(cls) for id, cls in LABELS.items()]
label_logits = openclip_text_preprocess(prompt_list)

#Generate Calibration Matrices
#-----------------------------------------------------------------------------------
print("Generating Calibration Matrices")
calib_true_class_arr = []
calib_sim_score_arr = []
calib_plausibility_score_arr = []
