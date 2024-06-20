import numpy as np


class Parameters(object):
    def __init__(self, name, parameters):
        self.name = name
        self.parameters = parameters
        if isinstance(self.parameters, np.ndarray):
            self.gradients = np.zeros(self.parameters.shape)
        else:
            self.gradients = 0

    def update(self, lr: float):
        self.parameters -= lr * self.gradients
