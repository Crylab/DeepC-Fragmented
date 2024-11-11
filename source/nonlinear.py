import numpy as np
import source.deepcf as deepcf
import DeePC.source.deepc as deepc
import source.deepce as deepce
from termcolor import colored
import time
import DeePC.source.track as track
import source.deepcfe as deepcfe
import source.deepcfg as deepcfg

import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib

viridis = matplotlib.colormaps['viridis']

class DoublePendulum:
    def set_default_parameters(self, dict_in: dict, name: str, value) -> None:
        if name in dict_in.keys():
                self.parameters[name] = dict_in[name]
        else:
            self.parameters[name] = value

    def __init__(self, parameters: dict = {}) -> None:
        # Parameters
        self.parameters = {}

        self.set_default_parameters(parameters, "m1", 1.0)
        self.set_default_parameters(parameters, "m2", 1.0)
        self.set_default_parameters(parameters, "l1", 1.0)
        self.set_default_parameters(parameters, "l2", 1.0)
        self.set_default_parameters(parameters, "g", 9.81)
        self.set_default_parameters(parameters, "k", 100.0)
        self.set_default_parameters(parameters, "c", 1.0)
        self.set_default_parameters(parameters, "dt", 0.01)
        self.set_default_parameters(parameters, "max_input", 20)

        self.dt = self.parameters["dt"]
                
        # State variables: [theta1, omega1, theta2, omega2]
        self.state = np.zeros(4)
        self.visual_storage = []

    def OptimalStep(self, target):
        theta1, velocity, theta2, _ = self.state

        m1 = self.parameters["m1"]
        m2 = self.parameters["m2"]
        l1 = self.parameters["l1"]
        l2 = self.parameters["l2"]
        g = self.parameters["g"]

        # Calculate weight compensation
        component1 = l1*m1*g*np.sin(target)
        component2 = m2 * g * (l1*np.sin(target) + l2*np.sin(theta2))

        # Calculate the feedback and velocity
        feedback = target - theta1
        component3 = 20 * feedback
        component4 = -10 * velocity

        return np.clip(component1 + component2 + component3 + component4, -self.parameters["max_input"], self.parameters["max_input"])

    def equations_of_motion(self, t, y, torque):
        # Unpack state variables
        theta1, omega1, theta2, omega2 = y

        # Unpack parameters
        m1 = self.parameters["m1"]
        m2 = self.parameters["m2"]
        l1 = self.parameters["l1"]
        l2 = self.parameters["l2"]
        g = self.parameters["g"]
        k = self.parameters["k"]
        damper = self.parameters["c"]
        

        # Pre-compute terms for the equations
        delta = theta2 - theta1
        M = m1 + m2
        m2l1l2_cos_delta = m2 * l1 * l2 * np.cos(delta)
        m2l1l2_sin_delta = m2 * l1 * l2 * np.sin(delta)
        
        # Equations derived from Lagrangian mechanics
        a = (M * l1**2)
        b = m2l1l2_cos_delta
        c = m2l1l2_cos_delta
        d = m2 * l2**2

        f1 = -m2l1l2_sin_delta * omega2**2 - M * g * l1 * np.sin(theta1) + torque
        f2 = m2l1l2_sin_delta * omega1**2 - m2 * g * l2 * np.sin(theta2)

        # Spring and damper forces
        spring_force = k * (delta)  # Spring force proportional to angle difference
        damper_force = damper * (omega2 - omega1)       # Damping force proportional to angular velocity difference

        # Add spring and damper forces to equations of motion
        f1 += spring_force + damper_force
        f2 -= spring_force + damper_force

        # Solving linear system for accelerations
        accel1, accel2 = np.linalg.solve(
            [[a, b], [c, d]], 
            [f1, f2]
        )

        return [omega1, accel1, omega2, accel2]

    def Step(self, input: float) -> float:
        """Simulate one time step forward using the current state and input torque."""

        torque = np.clip(input, -self.parameters["max_input"], self.parameters["max_input"])

        sol = solve_ivp(
            self.equations_of_motion, 
            [0, self.dt], 
            self.state, 
            args=(torque,), 
            t_eval=[self.dt]
        )
        self.state = sol.y[:, -1]
        self.visual_storage.append(self.get_positions())
        return self.state[0]

    def get_positions(self):
        """Compute the (x, y) positions of the links' ends based on angles."""
        theta1, _, theta2, _ = self.state
        x1 = self.parameters["l1"] * np.sin(theta1)
        y1 = -self.parameters["l1"] * np.cos(theta1)
        x2 = x1 + self.parameters["l2"] * np.sin(theta2)
        y2 = y1 - self.parameters["l2"] * np.cos(theta2)
        return (0, 0), (x1, y1), (x2, y2)

    def visualize(self):
        """Draw the double pendulum using matplotlib."""
        (x0, y0), (x1, y1), (x2, y2) = self.get_positions()
        
        plt.figure(figsize=(5, 5))
        plt.plot([x0, x1], [y0, y1], 'o-', lw=2, markersize=3, color='blue')
        plt.plot([x1, x2], [y1, y2], 'o-', lw=2, markersize=3, color='blue')
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.title("Learning Double Pendulum ...")
        plt.pause(self.dt)

    def post_visualization(self, ax, compress = 1):
        """Draw the double pendulum using matplotlib."""      
        for i, each in enumerate(self.visual_storage):
            if i % compress != 0:
                continue
            (x0, y0), (x1, y1), (x2, y2) = each
            color = viridis(float(i / len(self.visual_storage)))
            ax.plot([x0, x1], [y0, y1], 'o-', lw=2, markersize=2, color=color, alpha=0.5)
            ax.plot([x1, x2], [y1, y2], 'o-', lw=2, markersize=2, color=color, alpha=0.5)
        ax.plot([x0, x1], [y0, y1], 'o-', lw=3, markersize=5, color='blue')
        ax.plot([x1, x2], [y1, y2], 'o-', lw=3, markersize=5, color='blue')

    def Initialization(self, state=np.zeros(4)) -> None:
        self.state = state
        self.visual_storage.clear()

class Nonlinear_tracking:
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
        self.set_default_parameters(parameters, 'N', 300)
        self.set_default_parameters(parameters, 'control_horizon', 1)
        
        # Horizon parameters
        self.INITIAL_HORIZON = self.parameters['initial_horizon']
        self.PREDICTION_HORIZON = self.parameters['prediction_horizon']
        
        self.past_states = []
        self.past_actions = []
        self.reference_states = []
        
        np.random.seed(self.parameters['seed'])
    
        self.parameters["n_inputs"] = 1
        self.parameters["n_outputs"] = 1
            
        self.model = DoublePendulum(parameters)
        self.parameters.update(self.model.parameters)
        
        
        if self.parameters['algorithm'] == "deepcf":
            self.solver = deepcf.DeepC_Fragment(self.parameters)
        elif self.parameters['algorithm'] == "deepc":
            self.solver = deepc.DeepC(self.parameters)
        elif self.parameters['algorithm'] == "deepce":
            self.solver = deepce.DeepCe(self.parameters)
        elif self.parameters['algorithm'] == "deepcfe":
            self.solver = deepcfe.DeepCeF(self.parameters)
        elif self.parameters['algorithm'] == "deepcgf":
            n_sub = int(self.parameters["prediction_horizon"] / self.parameters["control_horizon"])
            self.PREDICTION_HORIZON = n_sub * self.parameters["control_horizon"]
            self.parameters["prediction_horizon"] = self.PREDICTION_HORIZON
            self.solver = deepcfg.DeepCgF(self.parameters)

        else:
            raise ValueError("Invalid algorithm")
        
        expand = self.parameters["tracking_time"]/100

        #meander = np.array([np.pi/2]*(self.parameters['tracking_time']+self.INITIAL_HORIZON+self.PREDICTION_HORIZON))
        meander = np.array([0.4]*(int(25*expand)+self.INITIAL_HORIZON) + [-0.24]*int(20*expand) + [-0.08]*int(25*expand) + [-0.4]*int(20*expand) + [0.4]*(int(10*expand)+self.PREDICTION_HORIZON)) * np.pi

        self.set_trajectory(meander)
        self.solver.set_opt_criteria(self.parameters.copy())
                
    def dataset_generation(self) -> None:
        dataset_inputs = []
        dataset_outputs = []
        self.model.Initialization()
        max_input = self.parameters["max_input"]
        total_length = self.parameters["initial_horizon"] + self.parameters["prediction_horizon"]
        tau = []
        targets = []
        out = []
        stable = 10
        for i in range(int((self.parameters["N"]+total_length))):
            desire = np.pi * np.random.random() - 0.5*np.pi
            #val = np.clip(val, -1, 1)
            for _ in range(stable):
                targets.append(desire)
        

        for i in range(self.parameters["N"]+total_length):
            torque = self.model.OptimalStep(targets[i])
            tau.append(torque)
            out.append(self.model.Step(torque))
            #self.model.visualize()

        for i in range(self.parameters["N"]):
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
        self.past_states = [0.0]*self.INITIAL_HORIZON
        for _ in range(self.INITIAL_HORIZON):
            self.past_actions.append(0.0)
            
        self.reference_states = self.trajectory[self.INITIAL_HORIZON:self.INITIAL_HORIZON + self.PREDICTION_HORIZON].copy().tolist()
        
        self.dataset_generation()
        self.solver.dataset_reformulation(self.solver.dataset)
                
        self.model.Initialization(np.array([0.0, 0.0, 0.0, 0.0]))    
                
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