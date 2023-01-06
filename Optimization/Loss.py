import numpy as np

class CrossEntropyLoss:
    def __init__(self):
        self.prediction_tensor = None
    def forward(self, prediction_tensor, label_tensor):
        eps = np.finfo(float).eps
        prediction_tensor = prediction_tensor + eps
        self.prediction_tensor = prediction_tensor
        return np.sum(np.multiply(-np.log(prediction_tensor),label_tensor)).astype(float)

    def backward(self, label_tensor):
        return -(label_tensor/self.prediction_tensor).astype(float)
