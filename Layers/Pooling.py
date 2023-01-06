from Layers.Base import Base
import numpy as np
import math

class Pooling(Base):
    def __init__(self, stride_shape, pooling_shape):
        Base.__init__(self)
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape
    def forward(self, input_tensor):
        self.batch_size = input_tensor.shape[0]
        self.num_channels = input_tensor.shape[1]
        self.input_image = input_tensor.shape[2:]
        #formula to get output shape
        self.output_height = math.floor((self.input_image[0] - self.pooling_shape[0]) / self.stride_shape[0]) + 1
        self.output_width = math.floor((self.input_image[1] - self.pooling_shape[1]) / self.stride_shape[1]) + 1

        output_tensor = np.zeros((input_tensor.shape[0], self.num_channels, self.output_height, self.output_width))
        self.max_location_tensor = np.zeros(output_tensor.shape, dtype='int')
        for a in range(input_tensor.shape[0]):
            for b in range(self.num_channels):
                for c in range(0, self.stride_shape[0] * self.output_height, self.stride_shape[0]):
                    for d in range(0, self.stride_shape[1] * self.output_width, self.stride_shape[1]):
                        pool = input_tensor[a, b, c:c + self.pooling_shape[0], d:d + self.pooling_shape[1]]

                        y = int(c / self.stride_shape[0])
                        x = int(d / self.stride_shape[1])
                        output_tensor[a, b, y, x] = np.amax(pool)
                        max_loc_index = np.unravel_index(np.argmax(pool, axis=None), pool.shape)
                        max_loc_index = (max_loc_index[0] + c, max_loc_index[1] + d)
                        # get max on original tensor
                        self.max_location_tensor[a, b, y, x] = max_loc_index[0] * self.input_image[1] + max_loc_index[1]
        return output_tensor
    def backward(self, error_tensor):
        prev_error_tensor = np.zeros((error_tensor.shape[0], self.num_channels, self.input_image[0], self.input_image[1]))
        for a in range(error_tensor.shape[0]):
            for b in range(self.num_channels):
                for c in range(self.output_height):
                    for d in range(self.output_width):
                        max_loc_index = np.unravel_index(self.max_location_tensor[a, b, c, d], self.input_image)
                        prev_error_tensor[a, b, max_loc_index[0], max_loc_index[1]] += error_tensor[a, b, c, d]
        return prev_error_tensor