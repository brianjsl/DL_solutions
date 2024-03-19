import numpy as np
from typing import Union

class PolynomialFeatures:

    def __init__(self, m: int):
        '''
        :params m: degree of the features  
        '''
        assert m >= 0
        self.m = m
    
    def transform(self, x: Union[float, np.ndarray]) -> np.ndarray:
        features = np.ndarray((x.size, self.m + 1))
        for i in range(self.m+1):
            features[:, i] = x ** i
        return features