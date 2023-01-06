import numpy as np
from Layers import FullyConnected
from Layers import TanH
from Layers import Sigmoid
from Optimization import *
from Layers.Base import Base


class RNN(Base):
    def __init__(self, input_size, hidden_size, output_size):
        Base.__init__(self)
        self.trainable = True
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        # The final (summed up over all time steps) gradient tensors
        self._hidden_grad = None
        self.output_grad_T = None
        self.Tan_H = TanH.TanH()
        self.SigmoidFn = Sigmoid.Sigmoid()
        self.inputFullyConnected = None
        self._memorize = False
        self.tanH = None
        self.fullyConnectedInBetween = None
        self.sigmoid = None
        self.hidden_states = None
        self.output_tensor = None
        self.batch_size = 1
        self.hidden_state_T = None
        self.gradient_calculated = False
        self._optimizer = None
        self.hidden_grad = None
        # For each time step gradient tensor in matrix
        self.hiddenMatrix_grad = None
        self.outputMatrix_grad = None
        self._dummy = None

        #Trick use Fully connected Layer to get  the hidden state
        self.fullyConnected = FullyConnected.FullyConnected(self.input_size + self.hidden_size, self.hidden_size)
        #Fully Connected Layer for the output y_hat
        self.outputFullyConnected = FullyConnected.FullyConnected(self.hidden_size, self.output_size)

    def forward(self, input_tensor):
        # To store the corresponding input tensors and activations
        self.gradient_calculated = False
        self.batch_size = input_tensor.shape[0]
        #Used in the backward pass
        self.tanH = np.zeros((self.batch_size, 1, self.hidden_size))
        self.fullyConnectedInBetween = np.zeros((self.batch_size, 1, self.hidden_size))
        self.sigmoid = np.zeros((self.batch_size, 1, self.output_size))
        self.inputFullyConnected = np.zeros((self.batch_size, 1, self.input_size + self.hidden_size))
        self.hidden_states = np.zeros((self.batch_size + 1, 1, self.hidden_size))
        #Used in the backward pass
        self.hiddenMatrix_grad_shape = list(self.fullyConnected.weights.shape)
        self.outputMatrix_grad_shape = list(self.outputFullyConnected.weights.shape)
        self.hiddenMatrix_grad_shape.insert(0, self.batch_size)
        self.outputMatrix_grad_shape.insert(0, self.batch_size)
        self.hiddenMatrix_grad = np.zeros(self.hiddenMatrix_grad_shape)
        self.outputMatrix_grad = np.zeros(self.outputMatrix_grad_shape)
        self._hidden_grad = np.zeros(self.fullyConnected.weights.shape)
        self.output_grad_T = np.zeros(self.outputFullyConnected.weights.shape)

        self.output_tensor = np.zeros((self.batch_size, self.output_size))
        #Used in the backward pass

        if self._memorize:
            self.hidden_states[0] = self.hidden_state_T

        for t in range(self.batch_size):
            x_t = input_tensor[t].reshape(1, -1)

            self.inputFullyConnected[t] = np.concatenate((x_t, self.hidden_states[t]), axis=1)
            #Tanh(x_tilda.W_h)
            self.hidden_states[t + 1] = self.Tan_H.forward(self.fullyConnected.forward(self.inputFullyConnected[t]))
            self.tanH[t] = self.hidden_states[t + 1]

            self.fullyConnectedInBetween[t] = self.hidden_states[t + 1]
            self.output_tensor[t] = self.SigmoidFn.forward(self.outputFullyConnected.forward(self.fullyConnectedInBetween[t]))
            self.sigmoid[t] = self.output_tensor[t]

        self.hidden_state_T = self.hidden_states[self.batch_size]
        self.inputFullyConnected = np.append(self.inputFullyConnected, np.ones((self.batch_size, 1, 1)), axis=2)
        self.fullyConnectedInBetween = np.append(self.fullyConnectedInBetween, np.ones((self.batch_size, 1, 1)), axis=2)
        return self.output_tensor

    def backward(self, error_tensor):
        self.hidden_grad = np.zeros((self.batch_size, 1, self.hidden_size))
        self.fullyConnected.optimizer = None
        self.outputFullyConnected.optimizer = None
        last_error = np.zeros((self.batch_size, self.input_size))
        for t in range(self.batch_size - 1, -1, -1):
            # Reset Activation
            error_t = error_tensor[t].reshape(1, -1)
            self.SigmoidFn.activations = self.sigmoid[t]
            error_t = self.SigmoidFn.backward(error_t)
            # backward pass fully connected
            self.outputFullyConnected.input_tensor = self.fullyConnectedInBetween[t]
            error_t = self.outputFullyConnected.backward(error_t)

            self.outputMatrix_grad[t] = self.outputFullyConnected.gradient_weights
            if t < self.batch_size:
                error_t = error_t + self.hidden_grad[t] #add Grad to hidden

            # Reset Activation
            self.Tan_H.activations = self.tanH[t]
            error_t = self.Tan_H.backward(error_t)
            # backward pass fully connected
            self.fullyConnected.input_tensor = self.inputFullyConnected[t]
            error_t = self.fullyConnected.backward(error_t)

            self.hiddenMatrix_grad[t] = self.fullyConnected.gradient_weights

            last_error[t] = error_t[0, 0:self.input_size]
            if t > 0:
                self.hidden_grad[t - 1] = error_t[0, self.input_size:]

        self._hidden_grad = np.sum(self.hiddenMatrix_grad, axis=0)
        self.output_grad_T = np.sum(self.outputMatrix_grad, axis=0)
        self.gradient_calculated = True

        if self._optimizer is not None:
            self.fullyConnected.weights = self._optimizer.calculate_update(self.fullyConnected.weights,
                                                                           self._hidden_grad)
            self.outputFullyConnected.weights = self._optimizer.calculate_update(self.outputFullyConnected.weights,
                                                                           self.output_grad_T)

        return last_error

    def initialize(self, weights_initializer, bias_initializer):
        self.fullyConnected.initialize(weights_initializer, bias_initializer)
        self.outputFullyConnected.initialize(weights_initializer, bias_initializer)
        return None

    def calculate_regularization_loss(self):
        regularization_loss = 0
        if self._optimizer.regularizer is not None:
            regularization_loss = self._optimizer.regularizer.norm(self.fullyConnected.weights)

        return regularization_loss

    @property
    def optimizer(self):
        return self._optimizer, self._dummy

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value
    @property
    def memorize(self):
        return self._memorize

    @memorize.setter
    def memorize(self, val):
        self._memorize = val

    @property
    def weights(self):
        return self.fullyConnected.weights

    @weights.setter
    def weights(self, value):
        self.fullyConnected.weights = value

    @property
    def gradient_weights(self):
        if self.gradient_calculated:
            return self._hidden_grad