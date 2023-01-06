import numpy as np
from Layers.Base import Base

class TanH(Base):
    def __init__(self):
        Base.__init__(self)
        self.activation = None

    def forward(self, input_tensor):
        self.activation = np.tanh(input_tensor)
        return self.activation

    def backward(self, error_tensor):
        temp = 1.0 - np.multiply(self.activation, self.activation)
        return np.multiply(temp, error_tensor)

    @property
    def activations(self):
        return self.activation

    @activations.setter
    def activations(self, value):
        self.activation = value
