from nyml.Tensor import tensor
import numpy as np
from typing import NamedTuple, Callable
from nyml.Tensor.tensor import Tensor

class Tensor_Dependency(NamedTuple):
    """
    A dependency is defining a tuple with specific inputs. These inputs contain the tensor label and the gradient of the tensor.
    """
    tensor: Tensor
    grad_fn: np.ndarray = Callable[[np.ndarray], np.ndarray] # ex. Callable[[int], str] is a function of (int) -> str.