from myframe.MyTensor import MyTensor
from typing import List

class Module:
    def __init__(self, modules: List["Module"]  = []):
        self.is_train = False
        self.modules = modules
    
    def __call__(self, *args, **kwargs) -> MyTensor:
        return self.forward(*args, **kwargs)
    
    def forward(self, x: MyTensor) -> MyTensor:
        for module in self.modules:
            x = module.forward(x)
        return x
    
    def backward(self, logits: MyTensor) -> None:
        logits.backward()
        
    def add_module(self, module: "Module") -> None:
        self.modules.append(module)
        
    def train(self) -> None:
        self.is_train = True
        for module in self.modules:
            module.train()
    
    def eval(self) -> None:
        self.is_train = False
        for module in self.modules:
            module.eval()
            
    def update(self, alpha: float = 1.0) -> None:
        for module in self.modules:
            module.update(alpha)
        
        
    