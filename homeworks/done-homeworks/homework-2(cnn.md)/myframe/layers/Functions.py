import numpy as np
from myframe.MyTensor import MyTensor
from .Module import Module


def relu(x: MyTensor) -> MyTensor:
    data = np.maximum(x.data, 0)
    output = MyTensor(data, (x,), 'relu')
    def _backward():
        x.grad += (data > 0.0) * output.grad
    output._backward = _backward
    return output

def sigmoid(x: MyTensor) -> MyTensor:
    sigma = lambda z: 1 / (1 + np.exp(-z))
    data = sigma(x.data)
    output = MyTensor(data, (x,), 'sigmoid')
    def _backward():
        x.grad += data * (1 - data)
    output._backward = _backward
    return output

def softmax(x: MyTensor, dim: int) -> MyTensor:
    exps = np.exp(x.data - np.max(x.data, axis=dim, keepdims=True))
    data = exps / np.sum(exps, axis=dim, keepdims=True)
    output = MyTensor(data, (x,), 'softmax')
    def _backward():
        s = np.sum(output.grad * data, axis=dim, keepdims=True)
        x.grad += data * (output.grad - s)
    output._backward = _backward
    return output

def dropout(x: MyTensor, p: float) -> MyTensor:
    mask = (np.random.rand(*x.data.shape) < p)
    data = x.data * mask / p
    output = MyTensor(data, (x,), 'dropout')
    def _backward():
        x.grad += output.grad * mask / p
    output._backward = _backward
    return output

class ReLU(Module):
    def forward(self, x: MyTensor) -> MyTensor:
        return relu(x)
    
class Sigmoid(Module):
    def forward(self, x: MyTensor) -> MyTensor:
        return sigmoid(x)
    
class Softmax(Module):
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim
        
    def forward(self, x: MyTensor) -> MyTensor:
        return softmax(x, self.dim)
    
class Dropout(Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p
        
    def forward(self, x: MyTensor) -> MyTensor:
        if self.is_train:
            return dropout(x, self.p)
        return x
        
    
    