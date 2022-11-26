import numpy as np
from scipy.optimize import brentq
from sklearn.metrics import precision_score, recall_score
import torch
from torch.distributions.one_hot_categorical import OneHotCategorical
from torch.distributions.categorical import Categorical 
# BEGIN CONFORMAL RISK CONTROL
'''
For each sample, returns ratio of number of false negatives to the number of expected true labels for that sample.
Inputs:
    - prediction_set_arr:   a boolean array representing the coverage sets 
                                predicted for each image.
                            dim: (num_examples, num_classes)
                            element at (example_i, class_j) represents 
                                whether class_j is in the prediction set of
                                example_i.
    - true_class_arr:       a boolean array representing the true classes
                                