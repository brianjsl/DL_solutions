import numpy as np

def RMS(y: np.ndarray, t: np.ndarray) -> np.float32:
    '''
    Root Mean Square Error.

    :params y: (N, ) array of predicted outputs
    :params t: (N, ) array of outputs
    '''
    N: int = y.size
    losses: np.ndarray = (y - t) ** 2
    return np.sqrt(np.sum(losses)/N, dtype=np.float32)
