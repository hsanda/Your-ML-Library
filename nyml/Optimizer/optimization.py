from typing import Callable
import numpy as np

class optimization():
    def __init__(self) -> None:
        pass

    def batch_iterator(self, data, size_of_batch):
            p = np.random.permutation(data.shape[0])
            data_rand = data[p]
            for i in np.arange(0, data.shape[0], size_of_batch):
                yield data_rand[i:i + size_of_batch]

    # ----------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------- First Order Algorithms -----------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------
    
    # --------------------------------------
    # ------ Gradient Descent Methods ------
    # --------------------------------------
    """
    By taking the gradients of all the points within the landscape, iteratively update parameters in order to minimize (descend) the loss function (within the loss function's landscape).  

    Args:
        params (dict): contains the parameters of the model
        lr (float): learning rate
        epochs (int): number of epochs
        loss_fun (Callable): loss function to be optimized
        data (np.ndarray): data to be used for the optimization
        
        var (<insert type>): description of var

    Returns:
        <dict>: The optimized parameters
    """    
    def gradient_descent(self, params:dict, lr:float, epochs:int, loss_fun:Callable, data:np.ndarray) -> dict:
        size_of_batch = len(data) # batch size is equal to the size of the dataset

        for i in range(epochs):
            for scrambled_dataset in self.batch_iterator(data, size_of_batch):
                d_params = {}
                d_params = self.eval_grads(loss_fun, params, scrambled_dataset)
                for param in params:
                    params[param] = params[param] - (lr * d_params[param]) # e.g. w = w - lr * d_w
                
        return params

    
    """
    By taking the gradient (only one gradient) at the current point within the landscape, iteratively update parameters in order to minimize (descend) the loss function (within the loss function's landscape).  

    Args:
        params (dict): contains the parameters of the model
        lr (float): learning rate
        epochs (int): number of epochs
        loss_fun (Callable): loss function to be optimized
        data (np.ndarray): data to be used for the optimization
        
        var (<insert type>): description of var

    Returns:
        <dict>: The optimized parameters
    """ 
    def stochastic_gradient_descent(self, params:dict, lr:float, epochs:int, loss_fun:Callable, data:np.ndarray) -> dict:
        size_of_batch = 1

        for i in range(epochs):
            for random_data_point in self.batch_iterator(data, size_of_batch):
                d_params = {}
                d_params = self.eval_grads(loss_fun, params, random_data_point)
                for param in params:
                    params[param] = params[param] - (lr * d_params[param]) # e.g. w = w - lr * d_w
                
        return params
    
    """
    By taking the gradient of all the points within a (batch) portion within the landscape, iteratively update parameters in order to minimize (descend) the loss function (within the loss function's landscape).  

    Args:
        params (dict): contains the parameters of the model
        lr (float): learning rate
        size_of_batch (int): size of the batch
        epochs (int): number of epochs
        loss_fun (Callable): loss function to be optimized
        data (np.ndarray): data to be used for the optimization
        
        var (<insert type>): description of var

    Returns:
        <dict>: The optimized parameters
    """ 
    def mini_batch_gradient_descent(self, params:dict, lr:float, size_of_batch:int, epochs:int, loss_fun:Callable, data:np.ndarray) -> dict:
        for i in range(epochs):
            for random_mini_batch in self.batch_iterator(data, size_of_batch):
                d_params = {}
                d_params = self.eval_grads(loss_fun, params, random_mini_batch)
                for param in params:
                    params[param] = params[param] - (lr * d_params[param]) # e.g. w = w - lr * d_w
                
        return params
    
    # -------------------------------------------
    # ----- Additional First Order Methods ------
    # ------------------------------------------- 
    
    """
    By taking the gradient of all the points within a (batch) portion within the landscape, iteratively update parameters in order to minimize (descend) the loss function (within the loss function's landscape).  

    Args:
        params (dict): contains the parameters of the model
        lr (float): learning rate
        size_of_batch (int): size of the batch
        epochs (int): number of epochs
        loss_fun (Callable): loss function to be optimized
        data (np.ndarray): data to be used for the optimization
        
        var (<insert type>): description of var

    Returns:
        <dict>: The optimized parameters
    """ 
    def momentum(self, params:dict, lr:float, gamma:float, size_of_batch:int, epochs:int, loss_fun:Callable, data:np.ndarray) -> dict:
        v_t = {}
        for param in params:
            v_t[param] = 0 # initialize v_t

        for i in range(epochs):
            for random_mini_batch in self.batch_iterator(data, size_of_batch):
                d_params = {}
                d_params = self.eval_grads(loss_fun, params, random_mini_batch)
                for param in params:
                    v_t[param] = (lr * d_params[param]) + (gamma * v_t[param]) # (gamma * v_t) is the momentum
                    params[param] = params[param] - v_t[param] # e.g. w = w - ((lr * d_w) + (gamma * v_{t-1}))
                
        return params
    
    def nesterov_gradient_acceleration(self, params:dict, lr:float, gamma:float, size_of_batch:int, epochs:int, loss_fun:Callable, data:np.ndarray) -> dict:
        v_t = {}
        for param in params:
            v_t[param] = 0 # initialize v_t

        for i in range(epochs):
            for random_mini_batch in self.batch_iterator(data, size_of_batch):
                d_params = {}
                params_copy = params.copy()
                for param in params_copy:
                    params_copy[param] = params_copy[param] - (gamma * v_t)

                d_params = self.eval_grads(loss_fun, params_copy, random_mini_batch)
                for param in params:
                    v_t[param] = (lr * d_params[param]) + (gamma * v_t[param]) # (gamma * v_t) is the momentum
                    params[param] = params[param] + v_t[param] # e.g. w = w - ((lr * d_w) + (gamma * v_{t-1}))
                
        return params
    
    def adaptive_gradient(self, params:dict, lr:float, epsilon:float, size_of_batch:int, epochs:int, loss_fun:Callable, data:np.ndarray) -> dict: #adagrad
        assert(epsilon > 0)
        g_t = {}
        for param in params:
            g_t[param] = 0 # initialize g_t

        for i in range(epochs):
            for random_mini_batch in self.batch_iterator(data, size_of_batch):
                d_params = {}
                d_params = self.eval_grads(loss_fun, params, random_mini_batch)
                for param in params:
                    g_t[param] = g_t[param] + np.pow(d_params[param], 2)
                    epsilon = np.matmul(epsilon, np.eye(params[param].shape[0]))
                    eta_t = lr / np.sqrt(g_t[param] + epsilon)
                    params[param] = params[param] - eta_t * d_params[param] # e.g. param_{t+1} = param_{t} - \frac{\eta}{\sqrt{v^{param}_{t} + (\epsilon * I)}} * \nabla_{param_{t}}
                
        return params

    def ada_delta(self, params:dict, lr:float, epsilon:float, gamma:float, size_of_batch:int, epochs:int, loss_fun:Callable, data:np.ndarray) -> dict:
        assert(epsilon > 0)
        g_t = {}
        delta_param_t = {}
        for param in params:
            g_t[param] = 0 # initialize g_t
            delta_param_t[param] = 0  # initialize delta_{param_{t}}

        for i in range(epochs):
            for random_mini_batch in self.batch_iterator(data, size_of_batch):
                d_params = {}
                d_params = self.eval_grads(loss_fun, params, random_mini_batch)
                for param in params:
                    g_t[param] = (gamma * g_t[param]) + ((1 - gamma) * np.pow(d_params[param], 2))
                    delta_param_t[param] = (gamma * delta_param_t[param]) + ((1 - gamma) * np.pow(d_params[param], 2))
                    epsilon = np.matmul(epsilon, np.eye(params[param].shape[0]))
                    eta_t = np.sqrt(delta_param_t[param] + epsilon) / np.sqrt(g_t[param] + epsilon)
                    params[param] = params[param] - eta_t * d_params[param] # e.g. param_{t+1} = param_{t} - \frac{\eta}{\sqrt{v^{param}_{t} + (\epsilon * I)}} * \nabla_{param_{t}}
                
        return params

    def root_mean_square_propagation(self, params:dict, lr:float, epsilon:float, gamma:float, size_of_batch:int, epochs:int, loss_fun:Callable, data:np.ndarray) -> dict: # rms_prop
        assert(epsilon > 0)
        g_t = {}
        for param in params:
            g_t[param] = 0 # initialize g_t

        for i in range(epochs):
            for random_mini_batch in self.batch_iterator(data, size_of_batch):
                d_params = {}
                d_params = self.eval_grads(loss_fun, params, random_mini_batch)
                for param in params:
                    g_t[param] = (gamma * g_t[param]) + ((1 - gamma) * np.pow(d_params[param], 2))
                    epsilon = np.matmul(epsilon, np.eye(params[param].shape[0]))
                    eta_t = lr / np.sqrt(g_t[param] + epsilon)
                    params[param] = params[param] - eta_t * d_params[param] # e.g. param_{t+1} = param_{t} - \frac{\eta}{\sqrt{v^{param}_{t} + (\epsilon * I)}} * \nabla_{param_{t}}
                
        return params

    def adaptive_moment_estimation(self, params:dict, lr:float, epsilon:float, beta:float, size_of_batch:int, epochs:int, loss_fun:Callable, data:np.ndarray) -> dict: # Adam
        assert(epsilon > 0)
        v_t = {}
        g_t = {}
        for param in params:
            v_t[param] = 0 # initialize v_t
            g_t[param] = 0 # initialize g_t

        beta_v, beta_g = beta

        for i in range(epochs):
            for random_mini_batch in self.batch_iterator(data, size_of_batch):
                d_params = {}
                d_params = self.eval_grads(loss_fun, params, random_mini_batch)
                for param in params:
                    v_t[param] = (beta_v * v_t[param]) + ((1 - beta_v) * np.pow(d_params[param], 2))
                    g_t[param] = (beta_g * v_t[param]) + ((1 - beta_g) * d_params[param])
                    g_hat = g_t[param] / (1 - (beta_g**(i+1)))
                    v_hat = v_t[param] / (1 - (beta_v**(i+1)))
                    epsilon = np.matmul(epsilon, np.eye(params[param].shape[0]))
                    eta_t = lr / np.sqrt(v_hat + epsilon)
                    params[param] = params[param] - eta_t * g_hat # e.g. param_{t+1} = param_{t} - \frac{\eta}{\sqrt{\hat{v^{param}_{t}} + (\epsilon * I)}} * \hat{g_{t}}
                
        return params

    def adamw(self, params:dict, lr:float, wd:float, epsilon:float, beta:float, size_of_batch:int, epochs:int, loss_fun:Callable, data:np.ndarray) -> dict: # Adam
        assert(epsilon > 0)
        v_t = {}
        g_t = {}
        for param in params:
            v_t[param] = 0 # initialize v_t
            g_t[param] = 0 # initialize g_t

        beta_v, beta_g = beta

        for i in range(epochs):
            for random_mini_batch in self.batch_iterator(data, size_of_batch):
                d_params = {}
                d_params = self.eval_grads(loss_fun, params, random_mini_batch)
                for param in params:
                    v_t[param] = (beta_v * v_t[param]) + ((1 - beta_v) * np.pow(d_params[param], 2))
                    g_t[param] = (beta_g * v_t[param]) + ((1 - beta_g) * d_params[param])
                    g_hat = g_t[param] / (1 - (beta_g**(i+1)))
                    v_hat = v_t[param] / (1 - (beta_v**(i+1)))
                    epsilon = np.matmul(epsilon, np.eye(params[param].shape[0]))
                    eta_t = lr / np.sqrt(v_hat + epsilon)
                    params[param] = params[param] - (eta_t * g_hat) - (lr * wd * params[param]) # e.g. param_{t+1} = param_{t} - \frac{\eta}{\sqrt{\hat{v^{param}_{t}} + (\epsilon * I)}} * \hat{g_{t}}
                
        return params, loss_fun

    # ----------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------- Second Order Algorithms ----------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------
    
    # --------------------------------------
    # ----------- Newton Methods -----------
    # -------------------------------------- 
    
    def newton_method(self, params:dict, size_of_batch:int, epochs:int, loss_fun:Callable, data:np.ndarray):
        for i in range(epochs):
            for random_mini_batch in self.batch_iterator(data, size_of_batch):
                d_params = {}
                d_params = self.eval_grads(loss_fun, params, random_mini_batch) # first derivative
                sd_params = self.eval_grads(loss_fun, params, random_mini_batch) # second derivative, TODO: implement second derivative 
                for param in params:
                    params[param] = params[param] - (d_params[param] / sd_params[param])
                    
    # --------------------------------------
    # -------- Quasi-Newton Methods --------
    # -------------------------------------- 

    def secant_method(self, params:dict, size_of_batch:int, epochs:int, loss_fun:Callable, data:np.ndarray):
        for i in range(epochs):
            prev_params = {}
            pre_d_params = {}
            for param in params:
                prev_params[param] = 0 # initialize v_t
                pre_d_params[param] = 0 # initialize g_t
            for random_mini_batch in self.batch_iterator(data, size_of_batch):
                d_params = {}
                sd_params = {}
                d_params = self.eval_grads(loss_fun, params, random_mini_batch) # first derivative 
                for param in params:
                    sd_params[param] = (d_params[param] - pre_d_params[param]) / (params[param] - prev_params[param])
                    params[param] = params[param] - (d_params[param] * (1 / sd_params[param]))
                prev_params = params.copy()
                pre_d_params = d_params.copy()

        return params

    def dfp(self): # davidson_fletcher_powell
        pass

    def bfgs(self): # Broyden-Fletcher-Goldfarb-Shanno 
        pass

    def l_bfgs(self): # Limited-memory BFGS
        pass

    def newton_raphson(self):
        pass

    def levenberg_marquardt(self):
        pass

    def powell_method(self):
        pass

    def steepest_descent(self):
        pass

    def truncated_newton(self):
        pass

    def fletcher_reeves(self):
        pass

    
    # ----------------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------- Overloaded functions -------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------
    
    """
    By taking the gradient (only one gradient) at the current point within the landscape, iteratively update parameters in order to minimize (descend) the loss function (within the loss function's landscape).  

    Args:
        params (dict): contains the parameters of the model
        lr (float): learning rate
        epochs (int): number of epochs
        loss_fun (Callable): loss function to be optimized
        data (np.ndarray): data to be used for the optimization
        
        var (<insert type>): description of var

    Returns:
        <dict>: The optimized parameters
    """ 
    def SGD(self, params:dict, lr:float, epochs:int, loss_fun:Callable, data:np.ndarray) -> dict:
        size_of_batch = 1

        for i in range(epochs):
            for random_data_point in self.batch_iterator(data, size_of_batch):
                d_params = {}
                d_params = self.eval_grads(loss_fun, params, random_data_point)
                for param in params:
                    params[param] = params[param] - (lr * d_params[param]) # e.g. w = w - lr * d_w
                
        return params
    
    """
    By taking the gradient of all the points within a (batch) portion within the landscape, iteratively update parameters in order to minimize (descend) the loss function (within the loss function's landscape).  

    Args:
        params (dict): contains the parameters of the model
        lr (float): learning rate
        size_of_batch (int): size of the batch
        epochs (int): number of epochs
        loss_fun (Callable): loss function to be optimized
        data (np.ndarray): data to be used for the optimization
        
        var (<insert type>): description of var

    Returns:
        <dict>: The optimized parameters
    """ 
    def mini_batch_SGD(self, params:dict, lr:float, size_of_batch:int, epochs:int, loss_fun:Callable, data:np.ndarray) -> dict:
        for i in range(epochs):
            for random_mini_batch in self.batch_iterator(data, size_of_batch):
                d_params = {}
                d_params = self.eval_grads(loss_fun, params, random_mini_batch)
                for param in params:
                    params[param] = params[param] - (lr * d_params[param]) # e.g. w = w - lr * d_w
                
        return params
    
    def NAG(self, params:dict, lr:float, gamma:float, size_of_batch:int, epochs:int, loss_fun:Callable, data:np.ndarray) -> dict:
        v_t = 0
        for i in range(epochs):
            for random_mini_batch in self.batch_iterator(data, size_of_batch):
                d_params = {}
                params_copy = params.copy()
                for param in params_copy:
                    params_copy[param] = params_copy[param] - (gamma * v_t)

                d_params = self.eval_grads(loss_fun, params_copy, random_mini_batch)
                for param in params:
                    vt_1 = (lr * d_params[param]) + (gamma * v_t) # (gamma * v_t) is the momentum
                    params[param] = params[param] + vt_1 # e.g. w = w - ((lr * d_w) + (gamma * vt_1))
                    v_t = vt_1 # everything before was done for readability of the math. This line is to update the momentum var but isnt true to form for the math.  
                
        return params

    def adagrad(self, params:dict, lr:float, epsilon:float, size_of_batch:int, epochs:int, loss_fun:Callable, data:np.ndarray) -> dict: 
        assert(epsilon > 0)
        g_t = {}
        for param in params:
            g_t[param] = 0 # initialize g_t

        for i in range(epochs):
            for random_mini_batch in self.batch_iterator(data, size_of_batch):
                d_params = {}
                d_params = self.eval_grads(loss_fun, params, random_mini_batch)
                for param in params:
                    g_t[param] = g_t[param] + np.pow(d_params[param], 2)
                    epsilon = np.matmul(epsilon, np.eye(params[param].shape[0]))
                    eta_t = lr / np.sqrt(g_t[param] + epsilon)
                    params[param] = params[param] - eta_t * d_params[param] # e.g. param_{t+1} = param_{t} - \frac{\eta}{\sqrt{v^{param}_{t} + (\epsilon * I)}} * \nabla_{param_{t}}
                
        return params

    def rms_prop(self, params:dict, lr:float, epsilon:float, gamma:float, size_of_batch:int, epochs:int, loss_fun:Callable, data:np.ndarray) -> dict: 
        assert(epsilon > 0)
        g_t = {}
        for param in params:
            g_t[param] = 0 # initialize g_t

        for i in range(epochs):
            for random_mini_batch in self.batch_iterator(data, size_of_batch):
                d_params = {}
                d_params = self.eval_grads(loss_fun, params, random_mini_batch)
                for param in params:
                    g_t[param] = (gamma * g_t[param]) + ((1 - gamma) * np.pow(d_params[param], 2))
                    epsilon = np.matmul(epsilon, np.eye(params[param].shape[0]))
                    eta_t = lr / np.sqrt(g_t[param] + epsilon)
                    params[param] = params[param] - eta_t * d_params[param] # e.g. param_{t+1} = param_{t} - \frac{\eta}{\sqrt{v^{param}_{t} + (\epsilon * I)}} * \nabla_{param_{t}}
                
        return params
    
    def adam(self, params:dict, lr:float, epsilon:float, beta:float, size_of_batch:int, epochs:int, loss_fun:Callable, data:np.ndarray) -> dict: 
        assert(epsilon > 0)
        v_t = {}
        g_t = {}
        for param in params:
            v_t[param] = 0 # initialize v_t
            g_t[param] = 0 # initialize g_t

        beta_v, beta_g = beta

        for i in range(epochs):
            for random_mini_batch in self.batch_iterator(data, size_of_batch):
                d_params = {}
                d_params = self.eval_grads(loss_fun, params, random_mini_batch)
                for param in params:
                    v_t[param] = (beta_v * v_t[param]) + ((1 - beta_v) * np.pow(d_params[param], 2))
                    g_t[param] = (beta_g * v_t[param]) + ((1 - beta_g) * d_params[param])
                    g_hat = g_t[param] / (1 - (beta_g**(i+1)))
                    v_hat = v_t[param] / (1 - (beta_v**(i+1)))
                    epsilon = np.matmul(epsilon, np.eye(params[param].shape[0]))
                    eta_t = lr / np.sqrt(v_hat + epsilon)
                    params[param] = params[param] - eta_t * g_hat # e.g. param_{t+1} = param_{t} - \frac{\eta}{\sqrt{\hat{v^{param}_{t}} + (\epsilon * I)}} * \hat{g_{t}}
                
        return params
