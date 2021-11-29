import numpy as np

class Activation():
    def __init__(self, activation_type):
        self.activation_type = activation_type

    def __call__(self, x, alpha=None, x_min=None, x_max=None):
        if self.activation_type == 'relu':
            return self.rectified_linear_unit(x)
        elif self.activation_type == 'leaky_relu':
            return self.leaky_relu()
        elif self.activation_type == 'prelu':
            return self.parameterized_relu(x, alpha)
        elif self.activation_type == 'smooth_relu':
            return self.smooth_relu(x)
        elif self.activation_type == 'elu':
            return self.exponential_linear_unit(x, alpha)
        elif self.activation_type == 'sigmoid':
            return self.sigmoid(x)
        elif self.activation_type == 'tanh':
            return self.tanh(x)
        elif self.activation_type == 'logit':
            return self.logit(x)
        elif self.activation_type == 'cosine':
            return self.cosine(x)
        elif self.activation_type == 'sine':
            return self.sine(x)
        elif self.activation_type == 'linear':
            return self.linear(x)
        elif self.activation_type == 'piecewise_linear':
            return self.piecewise_linear(x, x_min, x_max)
        elif self.activation_type == 'binary_step':
            return self.binary_step(x)
        elif self.activation_type == 'bipolar':
            return self.bipolar(x)
        elif self.activation_type == 'lecun_tanh':
            return self.lecun_tanh(x)   
        elif self.activation_type == 'hard_tanh':
            return self.hard_tanh(x)
        elif self.activation_type == 'abs':
            return self.abs(x)
        else:
            raise Exception('Activation type not found')

    def rectified_linear_unit(self, x): # ReLU (Rectified Linear Unit)
        return np.max(x, 0)
    
    def leaky_relu(self, x):
        return np.where(x >= 0, x, 0.01 * x)
    
    def parameterized_relu(self, x, alpha): # PReLU (Parameterized Rectified Linear Unit)   
        return np.where(x >= 0, x, alpha * x)
    
    def smooth_relu(self, x):
        return log(1 + np.exp(x))
    
    def exponential_linear_unit(self, x, alpha): # ELU (Exponential Linear Unit)
        return np.where(x < 0, alpha * (np.exp(x) - 1), x)
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        return np.tanh(x)

    def logit(self, x):
        return np.log(x / (1 - x))
    
    def cosine(self, x):
        return np.cos(x)
    
    def sine(self, x):
        return np.sin(x)

    def linear(self, x):
        return x
    
    def piecewise_linear(self, x, x_min=0, x_max=1):
        ret_val = None
        if x < x_min:
            ret_val = 0
        elif x > x_max:
            ret_val = 1
        elif x_min <= x <= x_max:
            m = 1 / (x_max - x_min)
            b = 1 - m * x_max
            ret_val = m * x + b   
        else:
            raise Exception('Error in piecewise_linear')
        
        return ret_val
        
    def binary_step(self, x):
        return np.where(x >= 0, 1, 0)
    
    def bipolar(self, x):
        ret_val = None
        if x > 0:
            ret_val = 1
        elif x < 0:
            ret_val = -1
        else:
            ret_val = 0
            
        return ret_val
    
    def lecun_tanh(self, x):
        return 1.7159 * np.tanh(2/3 * x)
    
    def hard_tanh(self, x):
        return np.max(np.min(x, 1), -1)
    
    def abs(self, x):
        return np.abs(x)
        