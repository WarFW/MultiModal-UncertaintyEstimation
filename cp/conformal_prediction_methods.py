import numpy as np
from scipy.optimize import brentq
from sklearn.metrics import precision_score, recall_score
import torch
from torch.distributions.one_hot_categorical import OneHotCategorical
from torch.distributions.categorical import Categorical 
# BEGIN CONFORMAL RISK CONTROL
'''
For each sample, returns ratio of numb