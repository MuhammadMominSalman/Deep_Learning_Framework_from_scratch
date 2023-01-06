import numpy as np
from Layers.Base import Base
from Layers import Helpers
from Optimization import *
import copy


class BatchNormalization(Base):
    def __init__(self, channels):
        Base.__init__(self)
        self.trainable = True
        self.channels = channels
        self.weights = None
        self.bias = None
        self.batch_mean = 0
        self.batch_variance = 0

        self.initialize()
        self.alpha = 0.8
        self.first_time = True
        self._testing_phase = False
        self.optimizer_weights = None
        self.optimizer_bias = None
        self.input_shape = None
        self.gradient_calculated = False
        self.bais_calculated = False

    def initialize(self, weights_initializer=0, bias_initializer=0):
        self.weights = np.ones((1, self.channels))
        self.bias = np.zeros((1, self.channels))

    def forward(self, input_tensor):
        self.input_shape = input_tensor.shape
        self.input_tensor = input_tensor
        self.gradient_calculated = False
        self.bais_calculated = False
        self.base_weights = self.weights
        if not self._testing_phase:
            if len(input_tensor.shape) == 4:
                self.input_tensor = self.reformat(self.input_tensor)

            self.mean = np.mean(self.input_tensor, axis=0).reshape(1, -1)
            self.var = np.var(self.input_tensor, axis=0).reshape(1, -1)
            #for first time
            if self.first_time == True:
                self.batch_mean = self.mean
                self.batch_variance = self.var
                self.first_time = False
            else:
                self.batch_mean = self.alpha * self.batch_mean + (1 - self.alpha) * self.mean
                self.batch_variance = self.alpha * self.batch_variance + (1 - self.alpha) * self.var

            self.normalized_input = np.divide((self.input_tensor - self.mean), np.sqrt(self.var))

        else:
            if len(self.input_shape) == 4:
                self.input_tensor = self.reformat(self.input_tensor)
            self.batch_mean = self.batch_mean.reshape(1, -1)
            self.batch_variance = self.batch_variance.reshape(1, -1)
            self.normalized_input = np.divide((self.input_tensor - self.batch_mean), np.sqrt(self.batch_variance))

        output_tensor = np.multiply(self.weights, self.normalized_input) + self.bias

        if len(self.input_shape) == 4:
            output_tensor = self.reformat(output_tensor)

        return output_tensor

    def backward(self, error_tensor):
        if len(self.input_shape) == 4:
            error_tensor = self.reformat(error_tensor)
        self._gradient_weights = np.multiply(error_tensor, self.normalized_input)
        self._gradient_weights = np.sum(self._gradient_weights, axis=0).reshape(1, -1)
        self._gradient_bias = np.sum(error_tensor, axis=0).reshape(1, -1)
        self.gradient_calculated = True
        self.bais_calculated = True
        previous_error = Helpers.compute_bn_gradients(error_tensor, self.input_tensor, self.weights, self.mean,
                                                  self.var)
        if self.optimizer_weights is not None:
            self.weights = self.optimizer_weights.calculate_update(self.weights, self._gradient_weights)

        if self.optimizer_bias is not None:
            self.bias = self.optimizer_bias.calculate_update(self.bias, self._gradient_bias)

        if len(self.input_shape) == 4:
            previous_error = self.reformat(previous_error)

        return previous_error

    def reformat(self, tensor):
        if len(tensor.shape) == 4:
            self.b = tensor.shape[0]
            self.h = tensor.shape[1]
            self.m = tensor.shape[2]
            self.n = tensor.shape[3]

            tensor = tensor.reshape(self.b, self.h, self.m * self.n) #reshape to BxHxM.N
            tensor = np.transpose(tensor, axes=[0, 2, 1]) #reshape to BxM.NxH
            tensor = tensor.reshape(self.b * self.m * self.n, self.h) #reshape to B.M.NxH

        elif len(tensor.shape) == 2:
            tensor = tensor.reshape(self.b, self.m * self.n, self.h) #reshape to BxM.NxH
            tensor = np.transpose(tensor, axes=[0, 2, 1]) #reshape to BxHxM.N
            tensor = tensor.reshape(self.b, self.h, self.m, self.n) #reshape to BxHxMxN

        return tensor

    @property
    def gradient_weights(self):
        if self.gradient_calculated:
            return self._gradient_weights

    @property
    def gradient_bias(self):
        if self.bais_calculated:
            return self._gradient_bias

    @property
    def testing_phase(self):
        return self._testing_phase

    @testing_phase.setter
    def testing_phase(self, val):
        self._testing_phase = val

    @property
    def optimizer(self):
        return self.optimizer_weights, self.optimizer_bias

    @optimizer.setter
    def optimizer(self, val):
        self.optimizer_weights = copy.deepcopy(val)
        self.optimizer_bias = copy.deepcopy(val)

    @property
    def weights_optimizer(self):
        return self.optimizer_weights
