from Layers.Base import Base
import numpy as np

class SoftMax(Base):
    def __init__(self):
        Base.__init__(self)
        self._output = None
    def forward(self, input_tensor):
        # Subtract max tensor from the input tensor to avoid big values
        max_tensor = np.amax(input_tensor, axis=1).reshape(-1, 1)
        input_tensor = input_tensor - max_tensor
        total = np.sum(np.exp(input_tensor), axis=1).reshape(-1, 1)
        self._output = np.divide(np.exp(input_tensor), total)
        return np.divide(np.exp(input_tensor), total)
    def backward(self, error_tensor):
        return np.multiply(self._output,
                           error_tensor - np.einsum('...j,...j', error_tensor, self._output).reshape(-1, 1))