from Layers.Base import Base
import numpy as np


class Dropout(Base):
    def __init__(self, keep_prob):
        Base.__init__(self)
        self.keep_prob = keep_prob
        self.decision = None
        self._testing_phase = False

    def forward(self, input_tensor):
        if self._testing_phase:
            return input_tensor
        else:
            self.decision = np.random.random_sample(input_tensor.shape) < self.keep_prob
            input_tensor = np.multiply(input_tensor, self.decision)
            input_tensor = np.divide(input_tensor, self.keep_prob)
            return input_tensor

    def backward(self, error_tensor):
        error_tensor = np.multiply(error_tensor, self.decision)
        error_tensor = np.divide(error_tensor, self.keep_prob)
        return error_tensor

    @property
    def testing_phase(self):
        return self._testing_phase

    @testing_phase.setter
    def testing_phase(self, val):
        self._testing_phase = val
