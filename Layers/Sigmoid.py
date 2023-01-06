import numpy as np
from Layers.Base import Base

class Sigmoid(Base):

    def __init__(self):
        Base.__init__(self)
        self.activation = None

    def forward(self, input_tensor):
        self.activation = 1 + np.exp(-1.0 * input_tensor)
        self.activation = np.divide(1.0, self.activation)
        return self.activation

    def backward(self, error_tensor):
        temp = np.multiply(self.activation, 1.0 - self.activation)
        return np.multiply(temp, error_tensor)

    @property
    def activations(self):
        return self.activation

    @activations.setter
    def activations(self, value):
        self.activation = value
