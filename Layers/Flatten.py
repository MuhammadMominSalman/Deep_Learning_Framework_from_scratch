from Layers.Base import Base
import numpy as np

class Flatten(Base):
    def __init__(self):
        Base.__init__(self)
        self.input_shape = None
    def forward(self, input_tensor):
        self.input_shape = input_tensor.shape
        input_tensor = input_tensor.reshape(input_tensor.shape[0], np.prod(input_tensor.shape[1:]))
        return input_tensor
    def backward(self, error_tensor):
        error_tensor = error_tensor.reshape(self.input_shape)
        return error_tensor