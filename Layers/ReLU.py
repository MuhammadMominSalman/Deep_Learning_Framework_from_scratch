from Layers.Base import Base
class ReLU(Base):
    def __init__(self):
        Base.__init__(self)
        self.input_tensor = None
    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        return input_tensor * (input_tensor > 0).astype(float)
    def backward(self, error_tensor):
        return error_tensor * (self.input_tensor > 0).astype(float)

