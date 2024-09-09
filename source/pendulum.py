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
        
    def Step(self, input: float) -> float:
        if input > self.parameters["max_input"]:
            input = self.parameters["max_input"]
        elif input < -self.parameters["max_input"]:
            input = -self.parameters["max_input"]
        
        term1 = (input-self.parameters["damping"]*self.x_d)/self.parameters["mass"]
        term2 = self.parameters["gravity"]*np.sin(self.x)/self.parameters["length"]
        x_d_p = self.x_d + ((term1-term2))*self.dt
        x_p = self.x + self.x_d*self.dt + np.random.normal(0, self.parameters["E"])*self.dt
         
        self.x = x_p
        self.x_d = x_d_p
        return self.x
    
    def Initialization(self, x: float = 0.0, x_d: float = 0.0) -> None:
        self.x = x
        self.x_d = x_d
    
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
        self.set_default_parameters(parameters, 'seed', 1)
        
        np.random.seed(self.parameters['seed'])
    
        self.parameters["n_inputs"] = 1
        self.parameters["n_outputs"] = 1
            
        self.model = Pendulum(self.parameters)
        self.parameters.update(self.model.parameters)
        
        
        if self.parameters['algorithm'] == "deepcf":
            self.solver = deepcf.DeepC_Fragment(self.parameters)
        elif self.parameters['algorithm'] == "deepc":
            self.solver = deepc.DeepC(self.parameters)
        else:
            raise ValueError("Invalid algorithm")
        
    def trajectory_generation(self, x: float, x_d: float, l: int):
        
        self.model.Initialization(x, x_d)
        max_input = self.parameters["max_input"]
        tau = 2 * max_input * np.random.random(l) - max_input
        out = np.zeros(l)
        out[0] = x
        for t in range(1, l):
            x_p = self.model.Step(tau[t-1])
            out[t] = x_p
        return tau, out
        
    def dataset_generation(self) -> None:
        dataset_inputs = []
        dataset_outputs = []
        for i in range(0, self.parameters["N"]):    
            total_length = self.parameters["initial_horizon"] + self.parameters["prediction_horizon"]
            init_x = (2*np.random.random()-1) * np.pi
            init_x_d = (2*np.random.random()-1) * np.pi
            tau, out = self.trajectory_generation(init_x, init_x_d, total_length)
            dataset_inputs.append([tau])
            dataset_outputs.append([out])
        self.solver.set_data(dataset_inputs, dataset_outputs)
        
    