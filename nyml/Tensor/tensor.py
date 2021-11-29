import numpy as np
from typing import List
from nyml.Tensor.tensor_dependency import Tensor_Dependency

class Tensor():
    
    def __init__(self, data: np.ndarray, requires_grad: bool = False, is_leaf = True, depends_on: List[Tensor_Dependency] = None) -> None:
        """[summary]
        Constructor for the Tensor class.

        Args:
            data (np.ndarray): data of the numpy array
            requires_grad (bool, optional): gradient of the tensor if requires_grad is true. Defaults to False.
            depends_on (List[Tensor_Dependency], optional): a list of tensors that were used to create this tensor. Defaults to None.
            is_leaf (bool, optional): if the tensor is a leaf or not. Defaults to True.
        
        Constructor additional notes:    
        self.shape = data.shape # shape of the tensor based off the shape of the numpy array
        self.grad = None # gradient of the tensor
        
        """
        self.data = data 
        self.grad = None
        self.requires_grad = requires_grad 
        self.depends_on = depends_on or []
        self.is_leaf = is_leaf
        self.shape = data.shape 
        self._version = None
        
        
    def __repr__(self) -> str:
        return f"Tensor({self.data}, requires_grad = {self.requires_grad}, is_leaf = {self.is_leaf})"
    
    def backward(self, grad: 'Tensor' = None) -> None:
        """[summary]
        Backpropagation of the tensor.

        Args:
        """
        pass
            
    def __add__(self):
        return add(self)
    
    def __radd__(self):
        return add(self)
    
    def __sub__(self):
        return subtract(self)
    
    def __rsub__(self):
        return subtract(self)
    
    def __mul__(self):
        return multiplication(self)
    
    def __rmul__(self):
        return multiplication(self)
    
    def __div_(self):
        return divide(self)
    
    def __rdiv__(self):
        return divide(self)
    
    def __matmul__(self):
        return matmul(self)
    
    def __rmatmul__(self):
        return mat_mul(self)
    
    def __pos__(self):
        TODO
        pass
    
    def __neg__(self):
        TODO
        pass
    
    def __sum__(self):
        return sum(self)
    
    def __rsum__(self):
        return sum(self)
    
    def __pow__(self):
        return pow(self)
    
    def __rpow__(self):
        return pow(self)
    
    def __exp__(self):
        return exp(self)
    
    def __rexp__(self):
        return exp(self)
    
    def __sqrt__(self):
        return sqrt(self)
    
    def __rsqrt__(self):
        return sqrt(self)
    
    def __abs__(self):
        return abs(self)
    
    def __rabs__(self):
        return abs(self)
    
    def __log__(self):
        return log(self)
    
    def __rlog__(self):
        return log(self)
    
    def __max__(self):
        return max(self)
    
    def __min__(self):
        return min(self)
    
    def __sin__(self):
        return sin(self)
    
    def __rsin__(self):
        return sin(self)
    
    def __cos__(self):
        return cos(self)
    
    def __tan__(self):
        return tan(self)
    
    def __rtan__(self):
        return tan(self)
    
    def __sinh__(self):
        return sinh(self)
    
    def __rsinh__(self):
        return sinh(self)
    
    def __cosh__(self):
        return cosh(self)
    
    def __rcosh__(self):
        return cosh(self)
    
    def __tanh__(self):
        return tanh(self)
    
    def __rtanh__(self):
        return tanh(self)
    
    def __mean__(self):
        return mean(self)
    
    def __median__(self):
        return median(self)
    
    def __mode__(self):
        return _mode(self)
    
    def __cov__(self, x1, x2):
        return _cov(self, x1, x2)
    
    def __std_dev__(self):
        return _std_dev(self)
    
    def __maxpool__(self):
        return maxpool(self)
    
    def __cross_correlation__(self):
        return cross_correlation(self)
    
    def __concat__(self):
        return concatenate(self)
    
    def __expand_dims__(self):
        return expand_dims(self)
    
    def __slice__(self):
        return slice(self)
    
    
    
        
    
    
    