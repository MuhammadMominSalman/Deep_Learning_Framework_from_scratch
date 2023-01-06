from Layers.Base import Base
import numpy as np

class FullyConnected(Base):
    def __init__(self, input_size, output_size):
        Base.__init__(self)
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.rand(self.input_size + 1, self.output_size)
        self._optimizer = None
        self.trainable = True
        self._dummy = None

    @property
    def optimizer(self):
        return self._optimizer, self._dummy

    @optimizer.setter
    def optimizer(self, val):
        self._optimizer = val

    def forward(self, input_tensor):
        self.input_tensor = np.append(input_tensor, np.ones((input_tensor.shape[0], 1)), axis=1)
        return np.matmul(self.input_tensor, self.weights)

    def backward(self, error_tensor):
        prev_error_tensor = np.matmul(error_tensor, self.weights.T)
        # gradient with respect to weights
        self._gradient_tensor = np.matmul(self.input_tensor.T, error_tensor)
        self.gradient_calculated = True
        if self._optimizer != None:
            self.weights = self._optimizer.calculate_update(self.weights, self._gradient_tensor)
        return prev_error_tensor[:, :self.input_size]

    @property
    def gradient_weights(self):
        if self.gradient_calculated:
            return self._gradient_tensor

    def initialize(self, weights_initializer, bias_initializer):
        # weight_shape = (self.input_size, self.output_size)
        # bias_shape = (1, self.output_size)

        self.weights[:self.input_size, :] = weights_initializer.initialize((self.input_size, self.output_size),
                                                                           self.input_size, self.output_size)
        self.weights[self.input_size, :] = bias_initializer.initialize((1, self.output_size), 1,
                                                                       self.output_size)