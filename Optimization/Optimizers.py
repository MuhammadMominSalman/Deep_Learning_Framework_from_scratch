import numpy as np


class Optimizer:
    def __init__(self):
        self._regularizer = None

    def add_regularizer(self, regularizer):
        self._regularizer = regularizer

    @property
    def regularizer(self):
        return self._regularizer

    @regularizer.setter
    def regularizer(self, val):
        self._regularizer = val


class Sgd(Optimizer):
    def __init__(self, learning_rate):
        Optimizer.__init__(self)
        self.learning_rate = learning_rate
    def calculate_update(self, weight_tensor, gradient_tensor):
        sub_grad = 0
        if self.regularizer is not None:
            sub_grad = self.regularizer.calculate_gradient(weight_tensor)
        weight_tensor = weight_tensor - self.learning_rate * gradient_tensor - self.learning_rate * sub_grad
        return weight_tensor

class SgdWithMomentum(Optimizer):
    def __init__(self, learning_rate, momentum_rate):
        Optimizer.__init__(self)
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.v = 0.0
    def calculate_update(self, weight_tensor, gradient_tensor):
        sub_grad = 0
        if self.regularizer is not None:
            sub_grad = self.regularizer.calculate_gradient(weight_tensor)
        self.v = self.momentum_rate * self.v - self.learning_rate * gradient_tensor
        weight_tensor = weight_tensor + self.v - self.learning_rate * sub_grad
        return weight_tensor

class Adam(Optimizer):
    def __init__(self, learning_rate, mu, rho):
        Optimizer.__init__(self)
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.v = 0.0
        self.r = 0.0
        self.index = 0
    def calculate_update(self, weight_tensor, gradient_tensor):
        sub_grad = 0
        if self.regularizer is not None:
            sub_grad = self.regularizer.calculate_gradient(weight_tensor)
        eps = np.finfo(float).eps
        self.index += 1
        self.v = self.mu * self.v + (1 - self.mu) * gradient_tensor
        self.r = self.rho * self.r + (1 - self.rho) * np.multiply(gradient_tensor, gradient_tensor)

        v_without_bias = np.divide(self.v, 1 - np.power(self.mu, self.index))
        r_without_bias = np.divide(self.r, 1 - np.power(self.rho, self.index))
        weight_tensor = weight_tensor - self.learning_rate * np.divide(v_without_bias + eps, np.sqrt(r_without_bias) + eps) \
                        - self.learning_rate * sub_grad
        return weight_tensor