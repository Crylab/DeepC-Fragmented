#!/usr/bin/env python
# This line indicates that the script should be run using the Python interpreter.

import sys

# Importing the sys module to manipulate the Python runtime environment.

sys.path.insert(1, "source/")
# Inserting 'source/' at position 1 in the sys.path list, to allow importing modules from this directory.

import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
from matplotlib import cm
import source.deepcf as deepcf
from source.pendulum import Pendulum_tracking

# Define a function that prints "Hello, World!"
def print_hello_world():
    print("Hello, World!")
        
def test_MIMO():
    print("Test")
    params = {
        "N": 2,
        "initial_horizon": 2,
        "prediction_horizon": 3,
        "n_inputs": 2,
        "n_outputs": 2,
    }
    tot = params["initial_horizon"] + params["prediction_horizon"]
    obj = deepcf.DeepC_Fragment(params)
    
    # Setup the dataset
    dataset_inputs = []
    dataset_outputs = []
    for i in range(0, params["N"]):
        realization_outputs = []
        realization_actions = []

        for t in range(0, tot):
            realization_actions.append(np.array([0.1+t/100+i/1000, -0.1+t/100+i/1000]))
            realization_outputs.append(np.array([0.2+t/100+i/1000, -0.2+t/100+i/1000]))
        dataset_inputs.append(np.array(realization_actions).T)
        dataset_outputs.append(np.array(realization_outputs).T)
    obj.set_data(dataset_inputs, dataset_outputs)
    
    # Setup the initial conditions
    inputs = np.array([np.array([0.35, -0.35]) for each in range(0, params["initial_horizon"])]).T
    outputs = np.array([np.array([0.45, -0.45]) for each in range(0, params["initial_horizon"])]).T
    obj.set_init_cond(inputs, outputs)
    
    # Setup the reference
    inputs = np.array([np.array([0.55, -0.55]) for each in range(0, params["prediction_horizon"])]).T
    obj.set_reference(inputs)
    
    # Setup the criteria
    criteria = {
        "Q": [1.0, 1.1],
        "R": [2.0, 2.1],
        "lambda_y": [3.0, 3.1],
        "lambda_g": 4.0,
    }
    obj.set_opt_criteria(criteria)
    
    # Solve the optimization problem
    obj.dataset_reformulation(obj.dataset)
    obj.solve_raw()
    
def test_SISO():
    print("Test")
    params = {
        "N": 2,
        "initial_horizon": 2,
        "prediction_horizon": 8,
        "n_inputs": 1,
        "n_outputs": 1,
    }
    tot = params["initial_horizon"] + params["prediction_horizon"]
    obj = deepcf.DeepC_Fragment(params)
    
    # Setup the dataset
    dataset_inputs = []
    dataset_outputs = []
    for i in range(0, params["N"]):
        realization_outputs = []
        realization_actions = []

        for t in range(0, tot):
            realization_actions.append(np.array([0.1+t/100+i/1000]))
            realization_outputs.append(np.array([0.2+t/100+i/1000]))
        dataset_inputs.append(np.array(realization_actions).T)
        dataset_outputs.append(np.array(realization_outputs).T)
    obj.set_data(dataset_inputs, dataset_outputs)
    
    # Setup the initial conditions
    inputs = np.array([np.array([0.35]) for each in range(0, params["initial_horizon"])]).T
    outputs = np.array([np.array([0.45]) for each in range(0, params["initial_horizon"])]).T
    obj.set_init_cond(inputs, outputs)
    
    # Setup the reference
    inputs = np.array([np.array([0.55]) for each in range(0, params["prediction_horizon"])]).T
    obj.set_reference(inputs)
    
    # Setup the criteria
    criteria = {
        "Q": [1.0],
        "R": [2.0],
        "lambda_y": [3.0],
        "lambda_g": 4.0,
    }
    obj.set_opt_criteria(criteria)
    
    # Solve the optimization problem
    obj.dataset_reformulation(obj.dataset)
    result = obj.solve() 
    print(result)
        
def test_pendulum():
    params = {
        "dt": 0.1,
        "tracking_time": 50,
    }
    obj = Pendulum_tracking(params)
    obj.trajectory_tracking()
    print("Dataset generated")
    
        
# Check if the script is being run as the main module
if __name__ == "__main__":
    test_pendulum()
    exit()
    
    test_MIMO()
    test_SISO()
    exit()
    
    # Call the function to print "Hello, World!"
    print_hello_world()
    
    params = {
        "N": 100,
        "initial_horizon": 3,
        "prediction_horizon": 8,
        "n_inputs": 2,
        "n_outputs": 4,
    }
    
    obj = deepcf.local_deepc_tracking({})
    obj.trajectory_tracking()
    
    print("End")
    
