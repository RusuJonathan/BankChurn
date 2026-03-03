import pandas as pd
import numpy as np
from typing import List, Any
from scipy.stats import rankdata


def rank_average(predictions: List[np.ndarray]) -> np.ndarray:
    ranks = [rankdata(p) / len(p) for p in predictions]
    return np.mean(ranks, axis=0)