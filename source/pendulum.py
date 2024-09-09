import numpy as np
import source.deepcf as deepcf
import DeePC.source.deepc as deepc
from termcolor import colored
import time
import DeePC.source.track as track

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
        x_d_p = self.x_d + (term1-term2)*self.dt
        x_p = self.x + self.x_d*self.dt
         
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
        self.set_default_parameters(parameters, "R", [0.001])
        self.set_default_parameters(parameters, "Q", [1.0])
        self.set_default_parameters(parameters, "lambda_y", [100])
        self.set_default_parameters(parameters, "lambda_g", 1.0)
        self.set_default_parameters(parameters, 'initial_horizon', 1)
        self.set_default_parameters(parameters, 'prediction_horizon', 4)
        self.set_default_parameters(parameters, 'dt', 0.1)
        self.set_default_parameters(parameters, 'algorithm', "deepc")
        self.set_default_parameters(parameters, 'seed', 1)
        self.set_default_parameters(parameters, 'tracking_time', 100)
        
        # Horizon parameters
        self.INITIAL_HORIZON = self.parameters['initial_horizon']
        self.PREDICTION_HORIZON = self.parameters['prediction_horizon']
        
        self.past_states = []
        self.past_actions = []
        self.reference_states = []
        
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
        
        self.set_random_trajectory(self.parameters['tracking_time'])
        self.solver.set_opt_criteria(self.parameters.copy())
        
    def __trajectory_generation(self, x: float, x_d: float, l: int):
        
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
            tau, out = self.__trajectory_generation(init_x, init_x_d, total_length)
            dataset_inputs.append([tau])
            dataset_outputs.append([out])
        self.solver.set_data(dataset_inputs, dataset_outputs)
        
    def set_random_trajectory(self, l: int) -> None:
        _, self.trajectory = self.__trajectory_generation(
            (2*np.random.random()-1) * np.pi, 
            (2*np.random.random()-1) * np.pi, l)
        
    def set_trajectory(self, trajectory: np.ndarray) -> None:
        if trajectory.size == self.parameters['tracking_time']:
            self.trajectory = trajectory
        else:
            raise ValueError("Invalid trajectory size")
        
    def control_step(self) -> float:
        """
        Perform a control step based on the current state and reference state.

        Returns:
            model.Racecar_Action: An object containing the steering and throttle values.
        """
        inputs = [self.past_actions]
        outputs = [self.past_states]
        self.solver.set_init_cond(inputs, outputs)
        
        self.solver.set_reference([self.reference_states])

        # Solve the optimization problem
        result = self.solver.solve().T
        if result is None:
            raise ValueError("Optimization failed")
        action = result[0][0]
        return action
           
    def __winshift(self, vector: list, new_val) -> list:
        new_vector = vector[1:].copy()
        new_vector.append(new_val)
        return new_vector
        
    def __tracking_error(self, result: list) -> float:
        bank = 0.0
        shift = self.INITIAL_HORIZON + 1
        for i in range(0, len(result)):
            dist = np.abs(result[i]-self.trajectory[i+shift])
            bank += dist
        return bank/len(result)
        
    def __save_result(self, result: list, error: float, exec_time: float) -> None:
        pass
        
    def trajectory_tracking(self):
                
        # Check if simulation already exists
        
        
        self.past_states = self.trajectory[:self.INITIAL_HORIZON].copy().tolist()
        for _ in range(self.INITIAL_HORIZON):
            self.past_actions.append(0.0)
            
        self.reference_states = self.trajectory[self.INITIAL_HORIZON:self.INITIAL_HORIZON + self.PREDICTION_HORIZON].copy().tolist()
        
        self.dataset_generation()
        self.solver.dataset_reformulation(self.solver.dataset)
                
        self.model.Initialization(x=self.past_states[-1])    
                
        result = []
        
        total_time = len(self.trajectory) - self.INITIAL_HORIZON - self.PREDICTION_HORIZON
        
        percent_counter = 1
        
        start_time = time.time()
                
        for t in range(self.INITIAL_HORIZON, len(self.trajectory) - self.PREDICTION_HORIZON):            
            
            percent = 100.0 * (t - self.INITIAL_HORIZON) / total_time
            if percent > percent_counter:
                percent_counter += 1
                print(colored('Trajectory tracking: ', 'yellow'), colored(percent_counter - 1, 'white'), colored('%', 'yellow'))
                current_time = (101 - percent_counter) * (time.time() - start_time) / percent_counter
                print(colored('Estimated time: ', 'red'), colored(current_time, 'white'), colored(' sec. left', 'red'))
                        
            action = self.control_step()
                        
            step_after = self.model.Step(action)
                        
            result.append(step_after)
            self.past_states = self.__winshift(self.past_states, step_after)
            self.past_actions = self.__winshift(self.past_actions, action)
            self.reference_states = self.__winshift(self.reference_states, self.trajectory[t+self.PREDICTION_HORIZON])
                                
        exec_time = time.time()-start_time
        
        error_per = self.__tracking_error(result)
        
        self.rss = error_per
                    
        print(colored('Trajectory tracking: Succeded', 'green'))
        print(colored('Accumulated tracking error per point:', 'green'), colored('{:.2f} m.'.format(error_per), 'white'))
        print(colored('Execution time:', 'green'), colored('{:.2f} s.'.format(exec_time), 'white'))
        print(colored('Averege responce time:', 'green'), colored('{:.2f} s.'.format(exec_time/total_time), 'white'))
                        
        self.__save_result(result, error_per, exec_time)
            
        return result
        
    