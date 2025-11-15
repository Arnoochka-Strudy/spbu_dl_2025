from numpy.typing import NDArray
import numpy as np
from typing import Optional

class Stack:
    def __init__(self):
        self.data = [None]
        self.size = 0
        self.capacity = 1
        
    def empty(self):
        return self.size == 0
    
    def put(self, item):
        if self.size == self.capacity:
            self.data += [None] * self.capacity
            self.capacity *= 2
        self.data[self.size] = item
        self.size += 1
        
    def get(self):
        if self.empty():
            return None
        self.size -= 1
        item = self.data[self.size]
        self.data[self.size] = None
        return item
    
    def __repr__(self):
        return f"{self.data}"

class MyTensor:
    def __init__(self, data: NDArray, _children=(), _op='input'):
        self.data = data
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"MyTensor(data={self.data}, grad={self.grad})"

    def __add__(self, other: "MyTensor") -> "MyTensor":
        other = other if isinstance(other, MyTensor) else MyTensor(other)
        output = MyTensor(self.data + other.data, (self, other), 'add')

        def _backward():
            self.grad += self.backward_broadcast(output.grad, len(self.grad.shape))
            other.grad += self.backward_broadcast(output.grad, len(other.grad.shape))
        output._backward = _backward

        return output

    def __mul__(self, other: "MyTensor") -> "MyTensor":
        output = MyTensor(self.data * other.data, (self, other), 'mul')
        def _backward():
            self.grad += self.backward_broadcast(other.data * output.grad, len(self.grad.shape))
            other.grad += self.backward_broadcast(self.data * output.grad, len(other.grad.shape))
        output._backward = _backward
        
        return output
    
    def __matmul__(self, other: "MyTensor") -> "MyTensor":
        output = MyTensor(self.data @ other.data, (self, other), 'matmul')
        def _backward():
            self.grad += self.backward_broadcast(output.grad @ np.swapaxes(other.data, -1, -2), len(self.grad.shape))
            other.grad += self.backward_broadcast(np.swapaxes(self.data, -1, -2) @ output.grad, len(other.grad.shape))
        output._backward = _backward
        
        return output

    def backward_broadcast(self, grad: NDArray, shape: int) -> NDArray:
        while len(grad.shape) > shape:
            grad = grad.sum(axis=0)
            
        for i in range(shape):
            if self.grad.shape[i] == 1 and grad.shape[i] > 1:
                grad = grad.sum(axis=i, keepdims=True)
            
        while len(grad.shape) < shape:
            grad = np.expand_dims(grad, axis=0)
            
        return grad    
    
    def backward(self):
        topo = list()
        visited = set()
        stack = Stack()
        stack.put(self)
        temp_mark = set()
        while not stack.empty():
            node = stack.get()

            if node in visited:
                continue

            if node in temp_mark:
                visited.add(node)
                topo.append(node)
            else:
                temp_mark.add(node)
                stack.put(node)
                for child in node._prev:
                    if child not in visited:
                        stack.put(child)
                        
        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()
                        
        