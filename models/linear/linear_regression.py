from models.linear._regression import Regression
import numpy as np
from typing import Optional, Callable

class LinearRegression(Regression):
    '''
    Linear Regression class. 
    TODO: use singledispatch instead of match
    '''

    def __init__(self, m: int, basis: Callable[[np.ndarray], np.ndarray], loss: str = 'L2', alpha: Optional[np.float32] = None) -> None:
        '''
        :param m: int representing the number of weights to use in the linear model.
        :param basis: callable reprsenting a basis function for inputs
        :param loss: str representing the loss function
        '''
        self.w: Optional[np.ndarray] = None
        self.m: int = m
        self._loss: str = loss
        self._basis: Callable[[np.ndarray], np.ndarray] = basis
        self._alpha: Optional[np.float64] = alpha

    def fit(self, x: np.ndarray, t: np.ndarray) -> None:
        '''
        Fits the regression model.

        :param x: training inputs
        :return: out
        '''
        match self._loss:
            case 'L2':
                processed_x: np.ndarray = self._basis(x)
                self.w = np.linalg.pinv(processed_x) @ t
            case 'Ridge':
                processed_x: np.ndarray = self._basis(x)
                inv = np.linalg.inv(self._alpha*np.identity(processed_x.shape[1]) + processed_x.T @ processed_x)
                self.w = inv @ processed_x.T @ t

    def run(self, x: np.ndarray) -> np.ndarray:
        '''
        Runs the model.

        :param x: input array of size (N, D) representing the inputs
        :return: output array of size (N, ) representing the outputs
        '''
        return self._basis(x) @ self.w 