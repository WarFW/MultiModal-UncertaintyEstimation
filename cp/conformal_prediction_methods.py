import numpy as np
from scipy.optimize import brentq
from sklearn.metrics import precision_score, recall_score
import torch
from torch.distributions.one_hot_categorical import OneHotCategori