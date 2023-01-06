from Layers.Base import Base
import numpy as np
import math
from scipy import signal
import copy

class Conv(Base):
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        Base.__init__(self)
        self.trainable = True
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels
        self.weights = np.random.random_sample((self.num_kernels,)+ self.convolution_shape)
        self.bias = np.random.rand(self.num_kernels)
        self._weights_gradient = np.zeros((self.num_kernels,) + self.convolution_shape)
        self._bias_gradient = np.zeros(self.num_kernels)
        self._bias_optimizer = None
        self._weights_optimizer = None
        self.gradient_calculated = False
        self.fan_in = np.prod(self.convolution_shape)
        self.fan_out = self.num_kernels * np.prod(self.convolution_shape[1:])

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        self.batch_size = input_tensor.shape[0]
        self.input_image = input_tensor.shape[2:]
        self.gradient_calculated = False
        if len(self.convolution_shape) == 3:
            self.pad_top = math.ceil((self.convolution_shape[1] - 1) / 2)
            self.pad_bottom = math.floor((self.convolution_shape[1] - 1) / 2)
            self.pad_left = math.ceil((self.convolution_shape[2] - 1) / 2)
            self.pad_right = math.floor((self.convolution_shape[2] - 1) / 2)
            output_tensor = np.zeros((input_tensor.shape[0], self.num_kernels,
                                    math.ceil(input_tensor.shape[2] / self.stride_shape[0]),
                                    math.ceil(input_tensor.shape[3] / self.stride_shape[1])))
            self.padded_input_tensor = np.pad(input_tensor,
                                              [(0, 0), (0, 0), (self.pad_bottom, self.pad_top),
                                               (self.pad_left, self.pad_right)],
                                              mode='constant')
        else:
            self.pad_left = math.ceil((self.convolution_shape[1] - 1) / 2)
            self.pad_right = math.floor((self.convolution_shape[1] - 1) / 2)
            output_tensor = np.zeros(
                (input_tensor.shape[0], self.num_kernels, math.ceil(input_tensor.shape[2] / self.stride_shape[0])))
            self.padded_input_tensor = np.pad(input_tensor, [(0, 0), (0, 0), (self.pad_left, self.pad_right)],
                                              mode='constant')
            #use correlate in forward pass
        for i in range(len(input_tensor)):
            for j in range(int(self.num_kernels)):
                image = signal.correlate(self.padded_input_tensor[i], self.weights[j], mode='valid')
                if len(self.convolution_shape) == 3:
                    image = image.reshape(image.shape[1], image.shape[2])
                    output_tensor[i, j] = image[::self.stride_shape[0], ::self.stride_shape[1]]
                else:
                    image = image.reshape(image.shape[1])
                    output_tensor[i, j] = image[::self.stride_shape[0]]

                output_tensor[i, j] = output_tensor[i, j] + self.bias[j]

        return output_tensor

    def backward(self, error_tensor):
        weights_flipped = np.stack((self.weights[0:self.num_kernels]), axis=1)

        if len(self.convolution_shape) == 3:
            weights_flipped = np.flip(weights_flipped, axis=1)

        if len(self.convolution_shape) == 3:
            error_tensor_sampled = np.zeros(
                (self.batch_size, self.num_kernels, self.input_image[0], self.input_image[1]))
            error_tensor_sampled[:, :, ::self.stride_shape[0], ::self.stride_shape[1]] = error_tensor
            # pad as done in forward pass
            error_tensor_sampled = np.pad(error_tensor_sampled,
                                            [(0, 0), (0, 0), (self.pad_bottom, self.pad_top),
                                             (self.pad_left, self.pad_right)], mode='constant')

            prev_error_tensor = np.zeros(
                (self.batch_size, self.convolution_shape[0], self.input_image[0], self.input_image[1]))
        else:
            error_tensor_sampled = np.zeros((self.batch_size, self.num_kernels, self.input_image[0]))
            # adding strides
            error_tensor_sampled[:, :, ::self.stride_shape[0]] = error_tensor
            # same padding for same convolution
            error_tensor_sampled = np.pad(error_tensor_sampled,
                                            [(0, 0), (0, 0), (self.pad_left, self.pad_right)], mode='constant')

            prev_error_tensor = np.zeros((self.batch_size, self.convolution_shape[0], self.input_image[0]))

        # use convolve in forward pass
        for i in range(self.batch_size):
            for j in range(len(weights_flipped)):
                image = signal.convolve(error_tensor_sampled[i], weights_flipped[j], mode='valid')
                image = image.reshape(image.shape[1:])

                prev_error_tensor[i, j] = image

        w_gradient_shape = list((self.num_kernels,) + self.convolution_shape).copy()
        w_gradient_shape.insert(0, self.batch_size)
        self.gradient_tensor = np.zeros(w_gradient_shape)
        self.bias_gradient = np.zeros((self.batch_size, self.num_kernels))
        if len(self.convolution_shape) == 3:
            self.bias_gradient = np.sum(error_tensor_sampled, axis=(2, 3))
            self.padded_input_tensor = np.pad(self.padded_input_tensor,
                                              [(0, 0), (0, 0), (self.pad_top, self.pad_bottom),
                                               (self.pad_left, self.pad_right)],
                                              mode='constant')
        else:
            self.bias_gradient = np.sum(error_tensor_sampled, axis=2)
            self.padded_input_tensor = np.pad(self.padded_input_tensor,
                                              [(0, 0), (0, 0), (self.pad_left, self.pad_right),
                                               ],
                                              mode='constant')

        # get gradient tensor
        for i in range(self.batch_size):
            for j in range(self.num_kernels):
                filter = error_tensor_sampled[i, j]
                filter = np.expand_dims(filter, axis = 0)
                self.gradient_tensor[i, j] = signal.correlate(self.padded_input_tensor[i], filter, mode='valid')


        self.gradient_calculated = True
        self._weights_gradient = np.sum(self.gradient_tensor, axis=0)
        self._bias_gradient = np.sum(self.bias_gradient, axis=0)
        if self._weights_optimizer != None:
            self.weights = self._weights_optimizer.calculate_update(self.weights, self._weights_gradient)
        if self._bias_optimizer != None:
            self.bias = self._bias_optimizer.calculate_update(self.bias, self._bias_gradient)
        return prev_error_tensor


    def initialize(self, weights_initializer, bias_initializer):
        self.weights = weights_initializer.initialize(self.weights.shape, self.fan_in, self.fan_out)
        self.bias = bias_initializer.initialize(self.bias.shape, self.fan_in, self.fan_out)

        return None

    @property
    def optimizer(self):
        return self._weights_optimizer, self._bias_optimizer

    @optimizer.setter
    def optimizer(self, val):
        self._weights_optimizer = copy.deepcopy(val)
        self._bias_optimizer = copy.deepcopy(val)


    @property
    def gradient_weights(self):
        if self.gradient_calculated:
            return self._weights_gradient

    @property
    def gradient_bias(self):
        if self.gradient_calculated:
            return self._bias_gradient
