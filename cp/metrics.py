import numpy as np
from conformal_prediction_methods import LOSS_FUNCTION
'''
Coverage and Efficiency Metrics:
Note that there are two universal arguments:
    - prediction_set_arr:   a boolean array representing the coverage sets 
                                predicted for each image.
                            dim: (n