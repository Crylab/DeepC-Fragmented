import numpy as np
import source.deepcf as deepcf
import DeePC.source.deepc as deepc

class Pendulum:
    def set_default_parameters(self, dict_in: dict, name: str, value) -> None:
        if name in dict_in.keys():
                self.parameters[name] = dict_in[name]
        else:
            self.parameters[name] = value
    
    def __init__(self, parameters: dict = {}) -> None:
        
        self.parameters = {}
        
        self.set_default_parameters(parameters, "length", 0.5) # m
        self.set_default_parameters(parameters, "mass", 1.0) # kg
        self.set_default_parameters(parameters, "dt", 0.1) # s
        self.set_default_parameters(parameters, "damping", 0.5) # Ns/m
        self.set_default_parameters(parameters, "gravity", 9.81) # m/s^2
        self.set_default_parameters(parameters, "max_input", 20.0) # N
        self.set_default_parameters(parameters, "E", 0.5)
        
        self.dt = self.parameters["dt"]
        
        self.x = 0.0
        self.x_d = 0.0
        
    def Step(self, input: float) -> None:
        term1 = (input-self.parameters["damping"]*self.x_d)/self.parameters["mass"]
        term2 = self.parameters["gravity"]*np.sin(self.x)/self.parameters["length"]
        x_d_p = self.x_d + ((term1-term2))*self.dt
        x_p = self.x + self.x_d*self.dt + np.random.normal(0, self.parameters["E"])*self.dt
         
        self.x = x_p
        self.x_d = x_d_p
        return self.x
    
class Pendulum_tracking:
    def set_default_parameters(self, dict_in: dict, name: str, value) -> None:
        if name in dict_in.keys():
                self.parameters[name] = dict_in[name]
        else:
            self.parameters[name] = value
    
    def __init__(self, parameters: dict = {}) -> None:
        
        self.parameters = {}
        
        self.set_default_parameters(parameters, "N", 20)
        self.set_default_parameters(parameters, "R", [10^(-3)])
        self.set_default_parameters(parameters, "Q", [1.0])
        self.set_default_parameters(parameters, "lambda_y", [10^(2)])
        self.set_default_parameters(parameters, "lambda_g", 1.0)
        self.set_default_parameters(parameters, "lambda_y", [10^(2)])
        self.set_default_parameters(parameters, "lambda_g", 1.0)
        self.set_default_parameters(parameters, 'initial_horizon', 1)
        self.set_default_parameters(parameters, 'prediction_horizon', 4)
        self.set_default_parameters(parameters, 'dt', 0.1)
        self.set_default_parameters(parameters, 'algorithm', "deepc")
    
        self.parameters["n_inputs"] = 1
        self.parameters["n_outputs"] = 1
            
        self.model = Pendulum(parameters)
        if self.parameters['algorithm'] == "deepcf":
            self.solver = deepcf.DeepC_Fragment(parameters)
        elif self.parameters['algorithm'] == "deepc":
            self.solver = deepc.DeepC(parameters)
        else:
            raise ValueError("Invalid algorithm")
        
    