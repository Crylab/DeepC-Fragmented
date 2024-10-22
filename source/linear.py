import numpy as np
import source.deepcf as deepcf
import DeePC.source.deepc as deepc
from termcolor import colored
import time
import DeePC.source.track as track

class LinearSystem:
    def set_default_parameters(self, dict_in: dict, name: str, value) -> None:
        if name in dict_in.keys():
                self.parameters[name] = dict_in[name]
        else:
            self.parameters[name] = value
    
    def __init__(self, parameters: dict = {}) -> None:
        
        self.parameters = {}
        
        self.set_default_parameters(parameters, "mass_one", 0.5)
        self.set_default_parameters(parameters, "mass_two", 1.5)
        self.set_default_parameters(parameters, "spring_one", 2.0)
        self.set_default_parameters(parameters, "spring_two", 2.0)
        self.set_default_parameters(parameters, "damping_one", 1.0)
        self.set_default_parameters(parameters, "damping_two", 1.0)
        self.set_default_parameters(parameters, "dt", 1.0) # s
        self.set_default_parameters(parameters, "max_input", 1.0)
        self.set_default_parameters(parameters, "sinusoid_amplitude", 0.2)
        self.set_default_parameters(parameters, "sinusoid_bias", 0.2)
        self.set_default_parameters(parameters, "sinusoid_freq", 0.01)
        self.set_default_parameters(parameters, "uniform_amplitude", 0.15)
        self.set_default_parameters(parameters, "measurement_std_dev", 0.1)
        self.set_default_parameters(parameters, "seed", 1)
        
        self.dt = self.parameters["dt"]        
        self.time = 0.0
        self.x = np.zeros(4)
        np.random.seed(self.parameters["seed"])
        
    def Step(self, input: float) -> float:
        # Input saturation
        if input > self.parameters["max_input"]:
            input = self.parameters["max_input"]
        elif input < -self.parameters["max_input"]:
            input = -self.parameters["max_input"]
        
        # Noise generation
        measurement_noise = 0.0#np.random.normal(0.0, self.parameters["measurement_std_dev"])

        sinusoidal_noise = np.sin(2*np.pi*self.parameters["sinusoid_freq"]*self.time) * self.parameters["sinusoid_amplitude"] + self.parameters["sinusoid_bias"]

        uniform_noise = 2 * self.parameters["uniform_amplitude"] * np.random.random() - self.parameters["uniform_amplitude"]

        # System dynamics

        k1 = self.parameters["spring_one"]
        k2 = self.parameters["spring_two"]
        c1 = self.parameters["damping_one"]
        c2 = self.parameters["damping_two"]
        m1 = self.parameters["mass_one"]
        m2 = self.parameters["mass_two"]

        ### State-space representation
        # x = [x1, x2, x1_dot, x2_dot]
        # x_dot = Ax + Bu
        # y = Cx + Du
        ### State-space matrices
        
        internal_dt = 0.01

        A = np.array([
                [0.0, 0.0, 1.0, 0.0], 
                [0.0, 0.0, 0.0, 1.0], 
                [-(k1+k2)/m1, k2/m1, -(c1+c2)/m1, c2/m1],
                [k2/m2, -k2/m2, c2/m2, -c2/m2],
            ])
        
        B = np.array([0.0, 0.0, 1.0/m1, 0.0])
        
        for _ in range(int(self.dt/internal_dt)):
            x_dot = np.dot(A, self.x) + np.dot(B, input+sinusoidal_noise+uniform_noise)
            self.x += x_dot * internal_dt
            self.time += internal_dt
            
        y = self.x[1] + measurement_noise 

        return y
    
    def Initialization(self) -> None:
        self.x = np.zeros(4)
        self.time = 0.0
    
class Linear_tracking:
    def set_default_parameters(self, dict_in: dict, name: str, value) -> None:
        if name in dict_in.keys():
                self.parameters[name] = dict_in[name]
        else:
            self.parameters[name] = value
    
    def __init__(self, parameters: dict = {}) -> None:
        
        self.parameters = {}
        
        
        self.set_default_parameters(parameters, "R", [1.0])
        self.set_default_parameters(parameters, "Q", [1.0])
        self.set_default_parameters(parameters, "lambda_y", [1])
        self.set_default_parameters(parameters, "lambda_g", 0.5)
        self.set_default_parameters(parameters, 'initial_horizon', 5)
        self.set_default_parameters(parameters, 'prediction_horizon', 10)
        self.set_default_parameters(parameters, 'dt', 1.0)
        self.set_default_parameters(parameters, 'algorithm', "deepcf")
        self.set_default_parameters(parameters, 'seed', 1)
        self.set_default_parameters(parameters, 'tracking_time', 100)

        dataset_size = 300 - self.parameters['initial_horizon'] - self.parameters['prediction_horizon']
        self.parameters['N'] = dataset_size
        
        # Horizon parameters
        self.INITIAL_HORIZON = self.parameters['initial_horizon']
        self.PREDICTION_HORIZON = self.parameters['prediction_horizon']
        
        self.past_states = []
        self.past_actions = []
        self.reference_states = []
        
        np.random.seed(self.parameters['seed'])
    
        self.parameters["n_inputs"] = 1
        self.parameters["n_outputs"] = 1
            
        self.model = LinearSystem(self.parameters)
        self.parameters.update(self.model.parameters)
        
        
        if self.parameters['algorithm'] == "deepcf":
            self.solver = deepcf.DeepC_Fragment(self.parameters)
        elif self.parameters['algorithm'] == "deepc":
            self.solver = deepc.DeepC(self.parameters)
        else:
            raise ValueError("Invalid algorithm")
        
        meander = np.array([0.4]*(25+self.INITIAL_HORIZON) + [-0.24]*20 + [-0.08]*25 + [-0.4]*20 + [0.4]*(10+self.PREDICTION_HORIZON))

        self.set_trajectory(meander)
        self.solver.set_opt_criteria(self.parameters.copy())
                
    def dataset_generation(self) -> None:
        dataset_inputs = []
        dataset_outputs = []
        self.model.Initialization()
        max_input = self.parameters["max_input"]
        tau = []
        out = []
        for i in range(30):
            val = 4 * max_input * np.random.random() - (2 * max_input)
            for _ in range(10):
                tau.append(val)
        
        for i in range(300):
            out.append(self.model.Step(tau[i]))

        total_length = self.parameters["initial_horizon"] + self.parameters["prediction_horizon"]
        for i in range(300-total_length):
            dataset_inputs.append([tau[i:i+total_length]])
            dataset_outputs.append([out[i:i+total_length]])
        self.solver.set_data(dataset_inputs, dataset_outputs)
                
    def set_trajectory(self, trajectory: np.ndarray) -> None:
        if trajectory.size == self.parameters['tracking_time']+self.INITIAL_HORIZON+self.PREDICTION_HORIZON:
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
    
    def __abs_tracking_error(self, result: list) -> float:
        bank = 0.0
        shift = self.INITIAL_HORIZON + 1
        for i in range(0, len(result)):
            dist = np.abs(result[i]-self.trajectory[i+shift])
            bank += dist
        return bank
        
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
                
        self.model.Initialization()    
                
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

        self.error = self.__abs_tracking_error(result)
        
        self.rss = error_per
                    
        print(colored('Trajectory tracking: Succeded', 'green'))
        print(colored('Accumulated tracking error per point:', 'green'), colored('{:.2f} m.'.format(error_per), 'white'))
        print(colored('Accumulated tracking error:', 'green'), colored('{:.2f} m.'.format(self.error), 'white'))
        print(colored('Execution time:', 'green'), colored('{:.2f} s.'.format(exec_time), 'white'))
        print(colored('Averege responce time:', 'green'), colored('{:.2f} s.'.format(exec_time/total_time), 'white'))
                        
        self.__save_result(result, error_per, exec_time)
            
        return result
        
    