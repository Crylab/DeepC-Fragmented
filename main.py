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
        
def experiment_pendulum(params = {}):
    params["algorithm"] = "deepc"
    obj = Pendulum_tracking(params)
    obj.trajectory_tracking()
    deepc_rss = obj.rss
    
    #########
    
    params["algorithm"] = "deepcf"
    obj = Pendulum_tracking(params)
    obj.trajectory_tracking()
    deepcf_rss = obj.rss
    
    print("RSS DeepC: ", deepc_rss)
    print("RSS DeepCF: ", deepcf_rss)
    
def experiment_pendulum_stable(params = {}):
    traj = np.ones(100) * np.pi / 8
    params["algorithm"] = "deepc"
    obj = Pendulum_tracking(params)
    obj.set_trajectory(traj)
    deepc_track = obj.trajectory_tracking()
    deepc_rss = obj.rss
    
    #########
    
    params["algorithm"] = "deepcf"
    obj = Pendulum_tracking(params)
    obj.set_trajectory(traj)
    deepcf_track = obj.trajectory_tracking()
    deepcf_rss = obj.rss
    
    print("RSS DeepC: ", deepc_rss)
    print("RSS DeepCF: ", deepcf_rss)
    
    # Plot the chart
    plt.figure()
    plt.plot(deepc_track, label="DeepC")
    plt.plot(deepcf_track, label="DeepCF")
    plt.plot(traj, label="Reference", linestyle="--")
    plt.xlabel("Time")
    plt.ylabel("Angle")
    plt.title("Pendulum Tracking")
    plt.legend()
    plt.show()
    
def path():
    # Generate values from 0 to pi/2
    x = np.linspace(-np.pi / 2, np.pi / 2, 100)
    y = (np.sin(x)+1)/2.6
    temp = np.ones(100)/1.3
    y_comb = np.concatenate((y, temp))
    return y_comb

def path2():
    temp = np.ones(200)/1.3
    return temp

def experiment_pendulum_transit(params):
    obj = Pendulum_tracking(params)
    obj.set_trajectory(path2())
    deepcf_track = obj.trajectory_tracking()
    rss = obj.rss
    return rss
    
def old_main():
       
    params = {
        "lambda_g": 10.0,
        "lambda_y": [1e2],
        "Q": [400],
        "R": [0.00001],
        "dt": 0.01,
        "seed": 1,
        "N": 20,
        "max_input": 20.0,
        "initial_horizon": 2,
        "prediction_horizon": 10,
        "tracking_time": 200,
    }
    N_seeds = 2
    deepc_dataset = []
    deepc_var = []
    deepcf_dataset = []
    deepcf_var = []
    #param_range = range(1, 10)
    param_range = np.logspace(-4, 2, num=20)
    for j in param_range:
        deepc_seeds = []
        params["algorithm"] = "deepc"
        params["lambda_g"] = j
        for i in range(0, N_seeds):
            params["seed"] = i
            try:
                rss = experiment_pendulum_transit(params)
                deepc_seeds.append(rss)
            except:
                pass
        deepc_dataset.append(np.mean(deepc_seeds))
        deepc_var.append(np.var(deepc_seeds))
        
        deepcf_seeds = []
        params["algorithm"] = "deepcf"
        for i in range(0, N_seeds):
            params["seed"] = i
            try:
                rss = experiment_pendulum_transit(params)
                deepcf_seeds.append(rss)
            except:
                pass
        deepcf_dataset.append(np.mean(deepcf_seeds))
        deepcf_var.append(np.var(deepcf_seeds))
        
    # Plot the chart
    plt.figure()
    plt.plot(param_range, deepc_dataset, label="DeepC", color="red")
    plt.fill_between(param_range, np.array(deepc_dataset) - np.sqrt(deepc_var),
                     np.array(deepc_dataset) + np.sqrt(deepc_var), alpha=0.2, color="red")
    plt.plot(param_range, deepcf_dataset, label="DeepCF", color="blue")
    plt.fill_between(param_range, np.array(deepcf_dataset) - np.sqrt(deepcf_var), 
                     np.array(deepcf_dataset) + np.sqrt(deepcf_var), alpha=0.2, color="blue")
    plt.xlabel("Lambda_g")
    plt.ylabel("RSS")
    plt.yscale("log")
    plt.xscale("log")
    plt.title("RSS Comparison between DeepC and DeepCF")
    plt.legend()
    plt.savefig("img/rss_comparison.pdf")
    plt.show()
        
    
        
    exit()
    
    test_MIMO()
    test_SISO()
    
    # Plot the chart
    if False:
        plt.figure()
        if is_deepc:
            plt.plot(deepc_track, label="DeepC")
        plt.plot(deepcf_track, label="DeepCF")
        plt.plot(path(), label="Reference", linestyle="--")
        plt.xlabel("Time")
        plt.ylabel("Angle")
        plt.title("Pendulum Tracking")
        plt.legend()
        plt.show()
    
    
    exit()

# Check if the script is being run as the main module
if __name__ == "__main__": 
    print("Main")