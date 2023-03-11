import numpy as np
from scipy.optimize import brentq

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
                                for each image.
                            dim: (num_examples, num_classes)
                            element at (example_i, class_j) represents
                                whether class_j is in the prediction set of
                                example_i.
Output:
    an array of length (true_class_arr), containing the loss for each sample
'''
def samplewise_loss(conformal_set_arr: np.array, true_class_arr: np.array):
    return 1 - np.sum(np.logical_and(conformal_set_arr, true_class_arr), axis=1) / np.sum(true_class_arr, axis=1)

'''
Set the Loss Function for Conformal Risk Control here.
'''
LOSS_FUNCTION = lambda pred_arr, true_arr: np.nanmean(samplewise_loss(pred_arr, true_arr))

