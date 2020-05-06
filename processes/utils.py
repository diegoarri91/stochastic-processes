import numpy as np


def get_dt(t):
    arg_dt = 20 if len(t) >= 20 else len(t)
    dt = np.mean(np.diff(t[:arg_dt]))
    return dt