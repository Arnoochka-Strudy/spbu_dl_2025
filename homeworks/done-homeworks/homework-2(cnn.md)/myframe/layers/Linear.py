import numpy as np
from .Module import Module
from myframe.MyTensor import MyTensor
from typing import Optional


class Linear(Module):
    def __init__(self,
                 weights: MyTensor,
                 bias: Optional[MyTensor] = None):
        super().__init__()
        self.weights = weights
        self.bias = bias
        
    def forward(self, x: MyTensor) -> MyTensor:
        logits = x @ self.weights
        if self.bias is not None:
            return  logits + self.bias
        return logits
    
    def update(self, alpha: float = 1.0) -> None:
        self.weights.data -= self.weights.grad * alpha
        if self.bias is not None:
            self.bias.data -= self.bias.grad * alpha
        
        
    





