import numpy as np
from conformal_prediction_methods import LOSS_FUNCTION
'''
Coverage and Efficiency Metrics:
Note that there are two universal arguments:
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
'''

'''
Computes the overall coverage (total true positive / total number of expected labels).
Inputs:
    - prediction_set_arr:   see above
    - true_class_arr:       see above
Output:
    the overall coverage across all classes and samples, as a proportion in [0, 1].
'''
def overall_coverage(conformal_set_arr: np.array, true_class_arr: np.array):
    return np.sum(np.logical_and(conformal_set_arr, true_class_arr)) / np.sum(true_class_arr)


'''
Computes the coverage for each class.
Inputs:
    - prediction_set_arr:   see above
    - true_class_arr:       see above
Output:
    an array of length (true_class_arr), containing the coverage score for
    each class.
'''
def class_stratified_coverage(conformal_set_arr: np.array, true_class_arr: np.array):
    return np.sum(np.logical_and(conformal_set_arr, true_class_arr), axis=0) / np.sum(true_class_arr, axis=0)


'''
Computes the coverage, stratified across the size of the **expected set** of true labels.
Inputs:
    - prediction_set_arr:   see above
    - true_class_arr:       see above
Output: a variable number of bins of the form (set_size, num_samples_in_bin, mean_coverage),
            represented as 3 arrays of the same variable length:
    - an array containing in increasing order the size of the true label sets represented 
        by each bin.
    - an array containing the number of samples in each b