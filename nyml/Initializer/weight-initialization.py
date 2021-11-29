import numpy as np

class Initializer():
    def __init__(self, initializer):
        self.initializer = initializer
        
    def __call__(self, x):
        if self.initializer == 'relu':
            return self.rectified_linear_unit(x)
        elif self.initializer == 'leaky_relu':
            return self.leaky_relu()
        elif self.initializer == 'prelu':
            return self.parameterized_relu(x, alpha)
        elif self.initializer == 'smooth_relu':
            return self.smooth_relu(x)
        elif self.initializer == 'elu':
            return self.exponential_linear_unit(x, alpha)
        elif self.initializer == 'sigmoid':
            return self.sigmoid(x)
        elif self.initializer == 'tanh':
            return self.tanh(x)
        elif self.initializer == 'logit':
            return self.logit(x)
        elif self.initializer == 'cosine':
            return self.cosine(x)
        elif self.initializer == 'sine':
            return self.sine(x)
        elif self.initializer == 'linear':
            return self.linear(x)
        elif self.initializer == 'piecewise_linear':
            return self.piecewise_linear(x, x_min, x_max)
        elif self.initializer == 'binary_step':
            return self.binary_step(x)
        elif self.initializer == 'bipolar':
            return self.bipolar(x)
        elif self.initializer == 'lecun_tanh':
            return self.lecun_tanh(x)   
        elif self.initializer == 'hard_tanh':
            return self.hard_tanh(x)
        elif self.initializer == 'abs':
            return self.abs(x)
        else:
            raise Exception('Initialization type not found')
        
        
    """
    ################################################################################
    ################################# Normals ######################################
    ################################################################################
    """
        
    def random_normal(self, x):
        pass
    
    def xavier_normal(self, x): # Xavier Glorot normal initialization
        pass
    
    def he_normal(self, x): # Kaiming He normal initialization
        pass
    
    def lecun_normal(self, x):
        pass
    
    def truncated_normal(self, x):
        pass
    

    """
    ################################################################################
    ############################# Uniforms and Gaussians ###########################
    ################################################################################
    """
    
    def random_uniform(self, x):
        pass
    
    def xavier_uniform(self, x): # Xavier Glorot uniform initialization
        pass
    
    def he_uniform(self, x): # Kaiming He uniform initialization
        pass
    
    def lecun_uniform(self, x):
        pass
    
    """
    ################################################################################
    ######################### Additional initialization ############################
    ################################################################################
    """
    
    def constant(self, x):
        pass
    
    def zeros(self, x):
        pass
    
    def ones(self, x):
        pass
    
    def orthogonal(self, x):
        pass
    
    def identity(self, x):
        pass
    
    def variance_scaling(self, x):
        pass
            
        