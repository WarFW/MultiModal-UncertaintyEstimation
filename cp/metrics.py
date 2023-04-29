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
Com