import numpy as np
from Layers import Flatten
from Layers import Base
from Layers import FullyConnected
from Layers import ReLU
from Layers import SoftMax
from Layers import Initializers
from Layers import Helpers
from Optimization import *
import copy


class NeuralNetwork:
    def __init__(self, optimizer, weights_initializer, bias_initializer):
        self.optimizer = optimizer
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None
        self.batch_data = None
        self.trainable_layer = []

    def append_layer(self, layer):
        layer.optimizer = copy.deepcopy(self.optimizer)
        if layer.trainable == True:
            layer.initialize(self.weights_initializer, self.bias_initializer)
            self.trainable_layer.append(len(self.layers))
        self.layers.append(layer)

    def forward(self):
        self.batch_data = self.data_layer.next()
        input_tensor = self.batch_data[0]
        reg_loss = 0
        for i in range(len(self.layers)):
            input_tensor = self.layers[i].forward(input_tensor)

        loss = self.loss_layer.forward(input_tensor, self.batch_data[1])
        for i in range(len(self.trainable_layer)):
            optimizer1, optimizer2 = self.layers[self.trainable_layer[i]].optimizer
            if optimizer1 is not None and optimizer1.regularizer is not None:
                reg_loss += optimizer1.regularizer.norm(self.layers[self.trainable_layer[i]].weights)
            # if optimizer2 is not None and optimizer2.regularizer is not None:
            #     reg_loss += optimizer2.regularizer.norm(self.layers[self.trainable_layer[i]].bais)
        return loss + reg_loss

    def backward(self):
        error_tensor = self.loss_layer.backward(self.batch_data[1])
        for i in range(len(self.layers) - 1, -1, -1):
            error_tensor = self.layers[i].backward(error_tensor)

    def train(self, iterations):
        self.testing_phase = False
        for i in range(iterations):
            loss = self.forward()
            self.loss.append(loss)
            self.backward()

    def test(self, input_tensor):
        self.testing_phase = True
        for i in range(len(self.layers)):
            input_tensor = self.layers[i].forward(input_tensor)
        return input_tensor

    @property
    def testing_phase(self):
        return self._testing_phase

    @testing_phase.setter
    def testing_phase(self, val):
        self._testing_phase = val
        for i in range(len(self.layers)):
            self.layers[i]._testing_phase = val