import numpy as np
from scipy.optimize import brentq

'''
For each sample, returns ratio of number of false negatives to the number of expected true labels for that sample.