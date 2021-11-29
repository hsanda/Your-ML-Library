import pytest
import numpy as np
from nyml.Tensor.tensor import Tensor


def test_tensor():
    a_array = np.ndarray([1, 2, 3])
    a_tensor = Tensor(a_array)
    
    b_array = np.ndarray([2, 4, 6])
    b_tensor = Tensor(b_array)
    
    assert a_tensor.shape == (3,), "shape for tensor_a is not 3x1"
    assert b_tensor.shape == (3,), "shape for tensor_b is not 3x1"
    assert a_tensor + b_tensor == Tensor(np.ndarray([3, 6, 9])), "addition of tensors resulted in the incorrect solution"
    
def test_tensor_add_scalar():
    a_array = np.ndarray([1, 2, 3])
    a_tensor = Tensor(a_array)
    
    b_array = np.ndarray([2, 4, 6])
    b_tensor = Tensor(b_array)
    
    assert a_tensor + 2 == Tensor(np.ndarray([3, 4, 5])), "addition of tensors and scalar resulted in the incorrect solution"
    assert 2 + a_tensor == Tensor(np.ndarray([3, 4, 5])), "addition of scalar and tensors resulted in the incorrect solution"
    
def test_tensor_subtract_scalar():
    a_array = np.ndarray([1, 2, 3])
    a_tensor = Tensor(a_array)
    
    b_array = np.ndarray([2, 4, 6])
    b_tensor = Tensor(b_array)
    
    assert a_tensor - 2 == Tensor(np.ndarray([-1, 0, 1])), "subtraction of tensors and scalar resulted in the incorrect solution"
    assert 2 - a_tensor == Tensor(np.ndarray([1, 0, -1])), "subtraction of scalar and tensors resulted in the incorrect solution"
    
def test_tensor_multiply_scalar():
    a_array = np.ndarray([1, 2, 3])
    a_tensor = Tensor(a_array)
    
    b_array = np.ndarray([2, 4, 6])
    b_tensor = Tensor(b_array)
    
    assert a_tensor * 2 == Tensor(np.ndarray([2, 4, 6])), "multiplication of tensors and scalar resulted in the incorrect solution"
    assert 2 * a_tensor == Tensor(np.ndarray([2, 4, 6])), "multiplication of scalar and tensors resulted in the incorrect solution"
    
def test_tensor_divide_scalar():
    a_array = np.ndarray([1, 2, 3])
    a_tensor = Tensor(a_array)
    
    b_array = np.ndarray([2, 4, 6])
    b_tensor = Tensor(b_array)
    
    assert a_tensor / 2 == Tensor(np.ndarray([0.5, 1, 1.5])), "division of tensors and scalar resulted in the incorrect solution"
    assert 2 / a_tensor == Tensor(np.ndarray([2, 2, 2])), "division of scalar and tensors resulted in the incorrect solution"
    assert b_tensor / 2 == Tensor(np.ndarray([1, 2, 3])), "division of tensors and scalar resulted in the incorrect solution"
    assert 2 / b_tensor == Tensor(np.ndarray([2, 4, 6])), "division of scalar and tensors resulted in the incorrect solution"

def test_tensor_matmul():
    a_array = np.ndarray([[1, 2], [3, 4]])
    a_tensor = Tensor(a_array)
    
    b_array = np.ndarray([[5, 6], [7, 8]])
    b_tensor = Tensor(b_array)
    
    assert a_tensor.matmul(b_tensor) == Tensor(np.ndarray([[19, 22], [43, 50]])), "matmul of tensor_a and tensor_b is not 19x2"

def test_tensor_sum():
    a_array = np.ndarray([1, 2, 3])
    a_tensor = Tensor(a_array)
    
    b_array = np.ndarray([2, 4, 6])
    b_tensor = Tensor(b_array)
    
    assert a_tensor.sum() == Tensor(np.ndarray([6])), "sum of tensor_a is not 6x1"
    assert b_tensor.sum() == Tensor(np.ndarray([12])), "sum of tensor_b is not 12x1"
        
def test_tensor_abs():
    a_array = np.ndarray([1, -2, 3])
    a_tensor = Tensor(a_array)
    
    assert a_tensor.abs() == Tensor(np.ndarray([1, 2, 3])), "abs of tensor_a is not 1x1"
    
def test_tensor_negative():
    a_array = np.ndarray([1, 2, 3])
    a_tensor = Tensor(a_array)
    
    b_array = np.ndarray([2, 4, 6])
    b_tensor = Tensor(b_array)
    
    assert -a_tensor == Tensor(np.ndarray([-1, -2, -3])), "negative of tensor_a is not -1x1"
    assert -b_tensor == Tensor(np.ndarray([-2, -4, -6])), "negative of tensor_b is not -2x1"


def test_tensor_a_grad():
    a_array = np.ndarray([1, 2, 3])
    a_tensor = Tensor(a_array, requires_grad=True)
    
    b_array = np.ndarray([2, 4, 6])
    b_tensor = Tensor(b_array)
    
    assert a_tensor.requires_grad == True, "requires_grad is not True"
    assert b_tensor.requires_grad == False, "requires_grad is not False"
    assert a_tensor.grad == np.ndarray([1, 2, 3]), "gradient of tensor_a is not 1x1"
    assert b_tensor.grad == np.ndarray([2, 4, 6]), "gradient of tensor_b is not 2x1"
    assert a_tensor + b_tensor == Tensor(np.ndarray([3, 6, 9])), "addition of tensors resulted in the incorrect solution"
    
