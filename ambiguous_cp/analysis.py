import sys
import os

from pathlib import Path
import pandas as pd
import scipy as sp
import numpy as np
from PIL import Image
import torch
from matplotlib import pyplot as plt
import pickle
import json
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

# Methods
def performance_report(threshold, calib_sim_score_arr, test_sim_score_arr, calib_true_class_arr, test_true_class_arr):
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
    return (calib_coverage, np.mean(calib_samplewise_efficiency), test_coverage, np.mean(test_samplewise_efficiency))

#Parse Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str, help='Experiment in experiment_configs to run')
parser.add_argument('--out', type=str, help='Where to output charts')
args = parser.parse_args()

# Parameters
reader = open(base_path + "\\experiment_configs\\"  + args.exp)
config = json.load(reader)
RESULTS_DIRECTORY = config["results_data_directory"]
OUTPUT_RESULT_DIR = args.out
CALIB_SIZE_CURVE = True
ALPHA_CURVE = True
UNCERTAIN_HIST = True
PLAUSIBILITY_HISTOGRAM = True
ORACLE = True
ALPHA = 0.5
NUM_SAMPLES = 100
LOGIT_SCALE = 100.0

# Load Files
calib_plausibility_score_arr = torch.load(RESULTS_DIRECTORY / "calib_plausibility_score_arr")
calib_sim_score_arr = torch.load(RESULTS_DIRECTORY / "calib_sim_score_arr")
calib_true_class_arr = torch.load(RESULTS_DIRECTORY / "calib_true_class_arr")
test_sim_score_arr = torch.load(RESULTS_DIRECTORY / "test_sim_score_arr")
test_true_class_arr = torch.load(RESULTS_DIRECTORY / "test_true_class_arr")
n_calib = calib_sim_score_arr.shape[0]
n_test = test_sim_score_arr.shape[0]
m = test_true_class_arr.shape[1]

# Amb CP on data-mined vs. Normal CP on original dataset
if ORACLE:
    ratio = 0.5
    print("Begin Alpha Curve")
    # Initialize metrics lists and set size list
    oracle_metrics = []
    norm_metrics = []
    amb_metrics = []
    alpha_values = [0.1*i for i in range(1, 6)]
    # Generate numpy matrices
    calib_sim_score_arr_np = calib_sim_score_arr.detach().cpu().numpy()
    calib_true_class_arr_np = calib_true_class_arr.detach().cpu().numpy()
    test_sim_score_arr_np = test_sim_score_arr.detach().cpu().numpy()
    test_true_class_arr_np = test_true_class_arr.detach().cpu().numpy()
    # Shuffle values
    random_order = np.random.permutation(len(test_sim_score_arr_np))
    test_sim_score_arr_np = test_sim_score_arr_np[random_order]
    test_true_class_arr_np = test_true_class_arr_np[random_order]
    random_order2 = np.random.permutation(len(calib_sim_score_arr_np))
    calib_sim_score_arr_np = calib_sim_score_arr_np[random_order2]
    calib_sim_score_arr = calib_sim_score_arr[random_order2]
    calib_true_class_arr_np = calib_true_class_arr_np[random_order2]
    calib_plausibility_score_arr = calib_plausibility_score_arr[random_order2]
    calib_length = min(int(ratio * len(test_sim_score_arr_np)), len(calib_sim_score_arr))
    # Loop through possible per class set sizes
    for alpha in alpha_values:
        #Perform Conformal Prediction
        print("Performing Conformal Prediction")
        threshold_amb = monte_carlo_cp(calib_sim_score_arr[:calib_length], calib_plausibility_score_arr[:calib_length], alpha, NUM_SAMPLES)
        threshold_oracle = compute_threshold(alpha, test_sim_score_arr_np[:calib_length], test_true_class_arr_np[:calib_length])
        threshold_norm = compute_threshold(alpha, calib_sim_score_arr_np[:calib_length], calib_true_class_arr_np[:calib_length])
        #Output Metrics
        print("\nAlpha Value: {alpha}".format(alpha = alpha))
        print("Normal CP:")
        norm_metrics.append(performance_report(threshold_norm, calib_sim_score_arr_np[:calib_length], test_sim_score_arr_np[calib_length:], \
                                               calib_true_class_arr_np[:calib_length], test_true_class_arr_np[calib_length:]))
        print("Oracle Normal CP:")
        oracle_metrics.append(performance_report(threshold_oracle, calib_sim_score_arr_np[:calib_length], test_sim_score_arr_np[calib_length:], \
                                               calib_true_class_arr_np[:calib_length], test_true_class_arr_np[calib_length:]))
        print("Ambiguous CP:")
        amb_metrics.append(performance_report(threshold_amb, calib_sim_score_arr_np[:calib_length], test_sim_score_arr_np[calib_length:], \
                                              calib_true_class_arr_np[:calib_length], test_true_class_arr_np[calib_length:]))
    # Generate deltas
    delta_norm = [norm_metrics[i][2]+alpha_values[i]-1 for i in range(0, len(norm_metrics))]
    delta_amb = [amb_metrics[i][2]+alpha_values[i]-1 for i in range(0, len(amb_metrics))]
    delta_oracle = [oracle_metrics[i][2]+alpha_values[i]-1 for i in range(0, len(oracle_metrics))]
    eff_norm = [norm_metrics[i][3] for i in range(0, len(norm_metrics))]
    eff_amb = [amb_metrics[i][3] for i in range(0, len(amb_metrics))]
    eff_oracle = [oracle_metrics[i][3] for i in range(0, len(oracle_metrics))]
    raw_data = {"alpha_values": alpha_values, "delta_norm": delta_norm, "delta_amb": delta_amb, "delta_oracle": delta_oracle, "eff_norm": eff_norm, "eff_amb": eff_amb, "eff_oracle": eff_oracle}
    #with open(OUTPUT_RESULT_DIR / "Method_Comparison.pkl", 'wb') as f: pickle.dump(raw_data, f)
    print([oracle_metrics[i][0] for i in range(0, len(oracle_metrics))])
    print([oracle_metrics[i][1] for i in range(0, len(oracle_metrics))])
    print([oracle_metrics[i][2] for i in range(0, len(oracle_metrics))])
    print([oracle_metrics[i][3] for i in range(0, len(oracle_metr