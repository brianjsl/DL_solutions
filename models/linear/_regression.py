import abc
import numpy as np

class Regression(metaclass = abc.ABCMeta):
    '''
    Regression base ABC
    '''
    
    @abc.abstractmethod
    def fit(self, x: np.ndarray, t: np.ndarray) -> None:
        '''
        Fits the weights of the model

        :param x: input array of size (N, D) representing the input data 
        :param y: output array of size (N, ) representing the output data
        '''

    @abc.abstractmethod
    def run(self, x: np.ndarray) -> np.ndarray:
        '''
        Runs the model.

        :param x: input array of size (N, D) representing the inputs
        :return: output array of size (N, ) representing the outputs
        '''