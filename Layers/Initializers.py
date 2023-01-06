from Layers.Base import Base
import numpy as np
import math


class Constant(Base):
    def __init__(self, cval = 0.1):
        self.constant = cval

    def initialize(self, weights_shape, fan_in, fan_out):
        return np.full(weights_shape, self.constant)

class UniformRandom(Base):
    def initialize(self, weights_shape, fan_in, fan_out):
        return np.random.random_sample(weights_shape)

class Xavier(Base):
    def initialize(self, weights_shape, fan_in, fan_out):
        sigma = math.sqrt(2.0 / (fan_in + fan_out))
        return np.random.normal(0.0, sigma, weights_shape)


class He(Base):
    def initialize(self, weights_shape, fan_in, fan_out):
        sigma = math.sqrt(2.0 / fan_in)
        return np.random.normal(0.0, sigma, weights_shape)