import numpy as np
from numpy.typing import NDArray
from .Module import Module
from myframe.MyTensor import MyTensor

def batchnorm(x: MyTensor,
              gamma: MyTensor,
              beta: MyTensor,
              mean: NDArray,
              var: NDArray,
              eps: float):
    N = x.data.shape[0]
    x_norm = MyTensor((x.data - mean) / np.sqrt(var + eps), (x,), 'normalize') 
    output = gamma * x_norm + beta
    def _backward():
        dx_norm = output.grad * gamma.data
        dvar = np.sum(dx_norm * (x.data - mean) * -0.5 * (var + eps)**(-1.5), axis=0)
        dmean  = np.sum(dx_norm * -1/np.sqrt(var + eps), axis=0) \
            + dvar * np.mean(-2*(x.data - mean), axis=0)
        x.grad += dx_norm / np.sqrt(var + eps) + dvar*2*(x.data - mean) / N + dmean / N
    output._backward = _backward
    return output

class BatchNorm(Module):
    def __init__(self, num_features: int, momentum: float = 0.1, eps: float = 1e-5):
        super().__init__()
        self.gamma = MyTensor(np.ones((1, num_features)))
        self.beta = MyTensor(np.zeros((1, num_features)))
        self.momentum = momentum
        self.eps = eps
        self.running_mean = np.zeros((1, num_features))
        self.running_var = np.ones((1, num_features))

    def forward(self, x: MyTensor) -> MyTensor:
        if self.is_train:
            mean = np.mean(x.data, axis=0, keepdims=True)
            var = np.var(x.data, axis=0, keepdims=True)

            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var  = (1 - self.momentum) * self.running_var  + self.momentum * var
        else:
            mean = self.running_mean
            var  = self.running_var

        return batchnorm(x, self.gamma, self.beta, mean, var, self.eps)
    
    def update(self, alpha: float = 1.0) -> None:
        self.gamma.data -= alpha * self.gamma.grad
        self.beta.data -= alpha * self.beta.grad