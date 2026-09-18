import numpy as np


def indices(event_s, window, fs, n):
    # Half-open sample-time interval [start, stop); no rounding into future.
    start, stop = [int(np.ceil((event_s + v) * fs - 1e-8)) for v in window]
    if start < 0 or stop > n or stop <= start:
        raise ValueError('window_out_of_bounds')
    return start, stop


def extract(x, event_s, window, fs):
    a, b = indices(event_s, window, fs, x.shape[-1])
    return x[..., a:b].copy(), (a, b)
