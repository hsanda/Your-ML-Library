import numpy as np
from nyml.Tensor.tensor import Tensor
from nyml.Tensor.tensor_dependency import Tensor_Dependency
from nyml.Operator.Function import Function

class Add(Function):
    
    def forward(A: Tensor, B: Tensor) -> Tensor:
        c_data = np.add(A.data, B.data)
        requires_grad = A.requires_grad or B.requires_grad
        depends_on = []
        
        if A.requires_grad:
            grad_fn = backward(A)
            depends_on.append(Tensor_Dependency(A, grad_fn))
            
        if B.requires_grad:
            grad_fn = self.forward(B)
            depends_on.append(Tensor_Dependency(B, grad_fn))
        
        return Tensor(c_data, requires_grad, depends_on, is_leaf=False)
    
    def backward(self, tensor: Tensor, grad: np.ndarray) -> np.ndarray:
        
        return grad
        
    
class Subtract(Function):

    def subtract(self, A: Tensor, B: Tensor):
        return A - B
    
    def forward(self):
        pass
    
    def backward(self):
        pass
    
class Matrix_Multiplication(Function):
    
    def matmul(self, A: Tensor, B: Tensor):
        return A @ B
    
    def forward(self):
        pass
    
    def backward(self):
        pass
    
class Element_Wise_Matrix_Multiplication(Function):
    
    def multiplication(self, A: Tensor, B: Tensor):
        return A * B
    
    def forward(self):
        pass
    
    def backward(self):
        pass
    
class Inner_Product_Matrix_Multiplication(Function):
    
    def multiplication(self, A: Tensor, B: Tensor):
        return np.inner(A, B)
    
    def forward(self):
        pass
    
    def backward(self):
        pass
    
class Divide(Function):
    
    def divide(self, A: Tensor, B: Tensor):
        return A / B
    
    def forward(self):
        pass
    
    def backward(self):
        pass
    
class Sum(Function):
    
    def sum(self, A: Tensor) -> Tensor:
        return np.sum(A)
    
    def forward(self):
        pass
    
    def backward(self):
        pass
    
class Power(Function):
    
    def pow(self, tensor: Tensor, n: int) -> Tensor:
        return np.pow(tensor, n)
    
    def forward(self):
        pass
    
    def backward(self):
        return (1/(n-1)) * np.pow(self.tensor_a, n-1)
    
class Exponential(Function):
    
    def exp(self, A: Tensor) -> Tensor:
        return np.exp(A)
    
    def forward(self):
        pass
    
    def backward(self):
        pass
    
class Square_Root(Function):
    
    def sqrt(self, A: Tensor) -> Tensor:
        return np.sqrt(A)
    
    def forward(self):
        pass
    
    def backward(self):
        pass
    
class Absolute_Value(Function):
    
    def abs(self, A: Tensor) -> Tensor:
        return np.abs(A)
    
    def forward(self):
        pass
    
    def backward(self):
        pass
    
class Logarithm(Function):
    
    def log(self, A: Tensor) -> Tensor:
        return np.log(A)
    
    def forward(self):
        pass
    
    def backward(self):
        pass
    
class Max(Function):
    
    def max(self, A: Tensor, B: Tensor) -> Tensor:
        return np.maximum(A, B)
    
    def forward(self):
        pass
    
    def backward(self):
        pass
    
class Min(Function):
    
    def min(self, A: Tensor, B: Tensor) -> Tensor:
        return np.minimum(A, B)
    
    def forward(self):
        pass
    
    def backward(self):
        pass
    
class Sine(Function):
    
    def sin(self, A: Tensor) -> Tensor:
        return np.sin(A)
    
    def forward(self):
        pass
    
    def backward(self):
        pass
    
class Cosine(Function):
    
    def cos(self, A: Tensor) -> Tensor:
        return np.cos(A)
    
    def forward(self):
        pass
    
    def backward(self):
        pass
    
class Tangent(Function):
    
    def tan(self, A: Tensor) -> Tensor:
        return np.tan(A)
    
    def forward(self):
        pass
    
    def backward(self):
        pass
    
class Hyperbolic_Sine(Function):
    
    def sinh(self, A: Tensor) -> Tensor:
        return np.sinh(A)
    
    def forward(self):
        pass
    
    def backward(self):
        pass
    
class Hyperbolic_Cosine(Function):
    
    def cosh(self, A: Tensor) -> Tensor:
        return np.cosh(A)
    
    def forward(self):
        pass
    
    def backward(self):
        pass
    
class Hyperbolic_Tangent(Function): 
    
    def tanh(self, A: Tensor) -> Tensor:
        return np.tanh(A)
    
    def forward(self):
        pass
    
    def backward(self):
        pass
    
class Mean(Function):
    
    def mean(self, A: Tensor) -> Tensor:
        return np.mean(A)
    
    def forward(self):
        pass
    
    def backward(self):
        pass
    
class Median(Function):
    
    def median(self, A: Tensor) -> Tensor:
        return np.median(A)
    
    def forward(self):
        pass
    
    def backward(self):
        pass
    
class Mode(Function):
    
    def mode(self, A: Tensor) -> Tensor:
        return np.bincount(A).argmax()
    
    def forward(self):
        pass
    
    def backward(self):
        pass
    
class Covariance(Function):
    def __init__(self) -> None:
        pass
    
    def cov(self, A: Tensor, B: Tensor) -> Tensor:
        return np.cov(A, B) 
    
class Standard_Deviation(Function):
    
    def std_dev(self, A: Tensor) -> Tensor:
        return np.std(A)
    
    def forward(self):
        pass
    
    def backward(self):
        pass
    
class Maxpool(Function):
    
    def maxpool(self, A: Tensor, kernel_size: tuple) -> Tensor:
        stride = np.max(A)
        #TODO: Implement Maxpool
        # refer to 
        # https://wiseodd.github.io/techblog/2016/07/18/convnet-maxpool-layer/
        pass
    
    def forward(self):
        pass
    
    def backward(self):
        pass
    
class Cross_Correlation(Function):
    
    def cross_correlation(self):
        pass
    
    def forward(self):
        pass
    
    def backward(self):
        pass
    
class Concatenate(Function):
    
    def concatenate(self):
        pass
    
    def forward(self):
        pass
    
    def backward(self):
        pass
    
class Expand_Dimensions(Function):
    
    def expand_dims(self):
        pass
    
    def forward(self):
        pass
    
    def backward(self):
        pass
    
class Slice(Function):
    
    def slice(self):
        pass
    
    def forward(self):
        pass
    
    def backward(self):
        pass
    
    