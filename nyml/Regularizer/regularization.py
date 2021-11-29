import numpy as np

class Regularization():
    def __init__(self):
        pass
    
    def l1_norm(self, x):
        """
        inputs:          
            x -> np.array:
        
        returns:
            
            
        short description:
        
        
        description:
            function in latex
            
        """
        return np.sum(np.abs(x))
    
    def l2_norm(self, x):
        """
        inputs:          
            x -> np.array:
        
        returns:
            
            
        short description:
        
        
        description:
            function in latex
            
        """
        return np.sqrt(np.sum(np.square(x)))
    
    def p_norm(self, x, p):
        """
        inputs:          
            x -> np.array:
            p -> float:
        
        returns:
            
            
        short description:
        
        
        description:
            function in latex
            
        """
        return np.sum(np.power(np.abs(x), p))
    
    def l1_regularizer(self, x, lambda_):
        """
        inputs:            
            x -> np.array: 
        
        returns:
            
            
        short description:
        
        
        description:
            function in latex
            \lambda * \sum_{i=1}^{N} \left | x_{i} \right |
        """
        return lambda_ * self.l1_norm(x)
    
    def l2_regularizer(self, x, lambda_):
        """
        inputs:            
            x -> np.array: 
            lambda_ -> float:
            
        returns:

            
        short description:
        
        
        description:
            function in latex
            \lambda * \sum_{i=1}^{N} (x_{i})^2
        """
        return lambda_ * self.l2_norm(x)
    
    def lp_regularizer(self, x, lambda_):
        """
        inputs:          
            x -> np.array:  
            lambda_ -> float:
        
        returns:
            
            
        short description:
        
        
        description:
            function in latex
            \sum_{i=1}^{N} \left | x_{i} \right |^p
        """
        return lambda_ * np.power(self.p_norm(), 1/p)