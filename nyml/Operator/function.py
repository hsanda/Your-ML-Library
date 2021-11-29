from abc import abstractmethod
import abs as ABS

class Function(ABS):
    
    @abstractmethod
    def __init__(self, *tensors) -> None:
        self.parents = tensors
        self.saved_tensors = []
        
    def save_tensor(self, tensor):
        self.saved_tensors.extend(tensor)
        
    @abstractmethod
    @staticmethod
    def forward(self):
        pass
        
    @abstractmethod
    @staticmethod
    def backward(self):
        pass
        
        
    
    def __add__(self):
        raise NotImplementedError
    
    def __radd__(self):
        raise NotImplementedError
    
    def __sub__(self):
        raise NotImplementedError
    
    def __rsub__(self):
        raise NotImplementedError
    
    def __mul__(self):
        raise NotImplementedError
    
    def __rmul__(self):
        raise NotImplementedError
    
    def __div_(self):
        raise NotImplementedError
    
    def __rdiv__(self):
        raise NotImplementedError
    
    def __matmul__(self):
        raise NotImplementedError
    
    def __rmatmul__(self):
        raise NotImplementedError
    
    def __pos__(self):
        raise NotImplementedError
    
    def __neg__(self):
        raise NotImplementedError
    
    def __sum__(self):
        raise NotImplementedError
    
    def __rsum__(self):
        raise NotImplementedError
    
    def __pow__(self):
        raise NotImplementedError
    
    def __rpow__(self):
        raise NotImplementedError
    
    def __exp__(self):
        raise NotImplementedError
    
    def __rexp__(self):
        raise NotImplementedError
    
    def __sqrt__(self):
        raise NotImplementedError
    
    def __rsqrt__(self):
        raise NotImplementedError
    
    def __abs__(self):
        raise NotImplementedError
    
    def __rabs__(self):
        raise NotImplementedError
    
    def __log__(self):
        raise NotImplementedError
    
    def __rlog__(self):
        raise NotImplementedError
    
    def __max__(self):
        raise NotImplementedError
    
    def __min__(self):
        raise NotImplementedError
    
    def __sin__(self):
        raise NotImplementedError
    
    def __rsin__(self):
        raise NotImplementedError
    
    def __cos__(self):
        raise NotImplementedError
    
    def __tan__(self):
        raise NotImplementedError
    
    def __rtan__(self):
        raise NotImplementedError
    
    def __sinh__(self):
        raise NotImplementedError
    
    def __rsinh__(self):
        raise NotImplementedError
    
    def __cosh__(self):
        raise NotImplementedError
    
    def __rcosh__(self):
        raise NotImplementedError
    
    def __tanh__(self):
        raise NotImplementedError
    
    def __rtanh__(self):
        raise NotImplementedError
    
    def __mean__(self):
        raise NotImplementedError
    
    def __median__(self):
        raise NotImplementedError
    
    def __mode__(self):
        raise NotImplementedError
    
    def __cov__(self, x1, x2):
        raise NotImplementedError
    
    def __std_dev__(self):
        raise NotImplementedError
    
    def __maxpool__(self):
        raise NotImplementedError
    
    def __cross_correlation__(self):
        raise NotImplementedError
    
    def __concat__(self):
        raise NotImplementedError
    
    def __expand_dims__(self):
        raise NotImplementedError
    
    def __slice__(self):
        raise NotImplementedError