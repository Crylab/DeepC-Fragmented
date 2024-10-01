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
import DeePC.source.deepc_tracking as deepc_tracking
from DeePC.source import graph
import os

def run_experiment(dict_in: dict):
    """
    Run a single experiment based on the specified algorithm.

    Arguments:
    dict_in -- dictionary containing configuration parameters, including the algorithm to use.

    Returns:
    rss -- Residual Sum of Squares (RSS) of the tracking deviation.
    """
    if dict_in["algorithm"] == "deepcf":
        obj = deepcf.DEEPCF_Tracking(dict_in)
    elif dict_in["algorithm"] == "deepc":
        obj = deepc_tracking.DEEPC_Tracking(dict_in)
    obj.trajectory_tracking()
    rss = float(obj.rss)
    return rss
    
def run_30_exp(dict_in: dict, n_seeds: int = 30):
    """
    Run 30 experiments based on the specified algorithm.

    Arguments:
    dict_in -- dictionary containing configuration parameters, including the algorithm to use.

    Returns:
    rss -- Residual Sum of Squares (RSS) of the tracking deviation.
    """
    rss = 0
    count = 0
    for i in range(n_seeds):
        local_dict = dict_in.copy()
        local_dict["seed"] = i
        count += 1
        try:
            rss += run_experiment(local_dict)
        except:
            pass
    return rss / count
    
def plot_lambda_g(N: int = 25):
    """
    Plot the effect of lambda_y on the tracking deviation.
    """
    # Define the parameters for the experiment
    range_g = np.logspace(-3, 0, 30)
    deepc_list = []
    deepcf_list = []
    for lambda_g in range_g:
        parameters = {
            "algorithm": "deepc",
            "prediction_horizon": 8,
            "Q": [1, 1, 1, 100],
            "R": [0.1, 0.1],
            "N": N,
            "lambda_g": float(lambda_g),
            "lambda_y": [float(50000)] * 4,
            "Lissajous_circle_time": 3700,
            "seed": 4,
            #"print_out": "Nothing",
        }
        result = run_experiment(parameters)
        deepc_list.append(result)
        parameters["algorithm"] = "deepcf"
        result = run_experiment(parameters)
        deepcf_list.append(result)
        
    # Plot the effect of lambda_y on the tracking deviation
    fig = plt.figure()
    plt.plot(range_g, deepc_list, label='DeepC')
    plt.plot(range_g, deepcf_list, label='DeepCF')
    plt.xlabel('lambda_g')
    plt.ylabel('RSS')
    plt.xscale('log')
    plt.grid()
    plt.legend()
    plt.title('Effect of lambda_g on RSS')
    plt.savefig(f'img/DeePCN{N}-lambda_g.pdf')
    plt.show()
    
def plot_lambda_y():
    """
    Plot the effect of lambda_y on the tracking deviation.
    """
    n_seeds = 3
    # Define the parameters for the experiment
    range_g = [0.57]#np.logspace(-1, 1, 30)
    range_y = np.logspace(0, 6, 30)
    deepc_list = []
    deepc_var = []
    for lambda_g in range_g:
        for lambda_y in range_y:
            seed_list = []
            for seed in range(n_seeds):
                parameters = {
                    "algorithm": "deepc",
                    "prediction_horizon": 8,
                    "Q": [1, 1, 1, 100],
                    "R": [0.1, 0.1],
                    "N": 50,
                    "lambda_g": float(lambda_g),
                    "lambda_y": [float(lambda_y)] * 4,
                    "Lissajous_circle_time": 3700,
                    "seed": seed,
                    #"print_out": "Nothing",
                }
                result = run_experiment(parameters)
                seed_list.append(result)
            deepc_list.append(np.mean(seed_list))
        deepc_var.append(np.var(seed_list))
        
    # Define the parameters for the experiment
    range_g = [0.13]#np.logspace(-1, 1, 30)
    deepcf_list = []
    deepcf_var = []
    for lambda_g in range_g:
        for lambda_y in range_y:
            seed_list = []
            for seed in range(n_seeds):
                parameters = {
                    "algorithm": "deepcf",
                    "prediction_horizon": 8,
                    "Q": [1, 1, 1, 100],
                    "R": [0.1, 0.1],
                    "N": 50,
                    "lambda_g": float(lambda_g),
                    "lambda_y": [float(lambda_y)] * 4,
                    "Lissajous_circle_time": 3700,
                    "seed": seed,
                    #"print_out": "Nothing",
                }
                result = run_experiment(parameters)
                seed_list.append(result)
            deepcf_list.append(np.mean(seed_list))
        deepcf_var.append(np.var(seed_list))
           
    # Plot the effect of lambda_y on the tracking deviation
    fig = plt.figure()
    plt.plot(range_y, deepc_list, label='DeepC')
    plt.plot(range_y, deepcf_list, label='DeepCF')
    #plt.fill_between(range_y, np.array(deepc_list) - np.array(deepc_var),
    #                 np.array(deepc_list) + np.array(deepc_var), alpha=0.2)
    plt.xlabel('lambda_y')
    plt.ylabel('RSS')
    plt.xscale('log')
    plt.grid()
    plt.legend()
    plt.title('Effect of lambda_g on RSS')
    plt.savefig("img/DeePCN50-lambda_y.pdf")
    
def dataset_variation():
    """
    Plot the effect of dataset size (N) on the tracking deviation.
    """
    
    deepc_list = []
    R = 0.1
    n_seeds = 30
    lambda_y = 50000
    dataset_list = range(2, 25)
    deepcf_list = []
    deepcf_var = []
    deepc_list = []
    deepc_var = []
    for N in dataset_list:
        seed_list = []
        for seed in range(n_seeds):
            
            parameters = {
                "algorithm": "deepcf",
                "prediction_horizon": 8,
                "Q": [1, 1, 1, 100],
                "R": [R] * 2,
                "N": N,
                "lambda_g": 0.06,
                "lambda_y": [lambda_y] * 4,
                "Lissajous_circle_time": 3700,
                "seed": seed,
            }
            try:
                result = run_experiment(parameters)
                seed_list.append(result)
            except:
                pass
        deepcf_list.append(np.mean(seed_list))
        deepcf_var.append(np.var(seed_list))
        seed_list = []
        for seed in range(n_seeds):
            
            parameters = {
                "algorithm": "deepc",
                "prediction_horizon": 8,
                "Q": [1, 1, 1, 100],
                "R": [R] * 2,
                "N": N,
                "lambda_g": 0.2,
                "lambda_y": [lambda_y] * 4,
                "Lissajous_circle_time": 3700,
                "seed": seed,
            }
            try:
                result = run_experiment(parameters)
                seed_list.append(result)
            except:
                pass
        deepc_list.append(np.mean(seed_list))
        deepc_var.append(np.var(seed_list))
    
    if True:
        if not os.path.exists('img'):
            os.makedirs('img')
        fig = plt.figure()
        plt.plot(dataset_list, deepc_list, label='DeepC')
        plt.plot(dataset_list, deepcf_list, label='DeepCF')
        plt.fill_between(dataset_list, 
                         np.array(deepc_list) - np.array(deepc_var), 
                         np.array(deepc_list) + np.array(deepc_var), 
                         alpha=0.2)
        print(deepc_var)
        plt.fill_between(dataset_list, 
                         np.array(deepcf_list) - np.array(deepcf_var), 
                         np.array(deepcf_list) + np.array(deepcf_var), 
                         alpha=0.2)
        plt.xlabel('N')
        plt.ylabel('RSS')
        plt.yscale('log')
        plt.grid()
        plt.legend()
        plt.title('Effect of Dataset size (N) on RSS of tracking deviation')
        plt.savefig(f'img/Dataset-variation.pdf')
        
    
def dataset_variation_high():
    """
    Plot the effect of dataset size (N) on the tracking deviation.
    """
    
    deepc_list = []
    R = 0.1
    n_seeds = 30
    lambda_y = 50000
    dataset_list = range(25, 125, 25)
    deepcf_list = []
    deepcf_var = []
    deepc_list = []
    deepc_var = []
    for N in dataset_list:
        seed_list = []
        for seed in range(n_seeds):
            
            parameters = {
                "algorithm": "deepcf",
                "prediction_horizon": 8,
                "Q": [1, 1, 1, 100],
                "R": [R] * 2,
                "N": N,
                "lambda_g": float(N)/500.0,
                "lambda_y": [lambda_y] * 4,
                "Lissajous_circle_time": 3700,
                "seed": seed,
            }
            try:
                result = run_experiment(parameters)
                seed_list.append(result)
            except:
                pass
        deepcf_list.append(np.mean(seed_list))
        deepcf_var.append(np.var(seed_list))
        seed_list = []
        for seed in range(n_seeds):
            
            parameters = {
                "algorithm": "deepc",
                "prediction_horizon": 8,
                "Q": [1, 1, 1, 100],
                "R": [R] * 2,
                "N": N,
                "lambda_g": float(N)/250.0,
                "lambda_y": [lambda_y] * 4,
                "Lissajous_circle_time": 3700,
                "seed": seed,
            }
            try:
                result = run_experiment(parameters)
                seed_list.append(result)
            except:
                pass
        deepc_list.append(np.mean(seed_list))
        deepc_var.append(np.var(seed_list))
    
    if True:
        fig = plt.figure()
        plt.plot(dataset_list, deepc_list, label='DeepC')
        plt.plot(dataset_list, deepcf_list, label='DeepCF')
        plt.fill_between(dataset_list, 
                         np.array(deepc_list) - np.array(deepc_var), 
                         np.array(deepc_list) + np.array(deepc_var), 
                         alpha=0.2)
        print(deepc_var)
        plt.fill_between(dataset_list, 
                         np.array(deepcf_list) - np.array(deepcf_var), 
                         np.array(deepcf_list) + np.array(deepcf_var), 
                         alpha=0.2)
        plt.xlabel('N')
        plt.ylabel('RSS')
        plt.grid()
        plt.legend()
        plt.title('Effect of Dataset size (N) on RSS of tracking deviation')
        plt.savefig(f'img/Dataset-variation-big.pdf')
        
        
def animation_of_deepcf():
    seed = 10 # 7 -> 3.44
    
    parameters_deepc = {
        "algorithm": "deepc",
        "prediction_horizon": 8,
        "Q": [1, 1, 1, 100],
        "R": [0.1] * 2,
        "N": 19,
        "lambda_g": 0.2,
        "lambda_y": [50000] * 4,
        "Lissajous_circle_time": 3700,
        "seed": seed,
    }
    obj = deepc_tracking.DEEPC_Tracking(parameters_deepc)
    traj_deepc = obj.trajectory_tracking()
    print("DeepC: ", obj.rss)
    
    parameters_deepc = {
        "algorithm": "deepcf",
        "prediction_horizon": 8,
        "Q": [1, 1, 1, 100],
        "R": [0.1] * 2,
        "N": 19,
        "lambda_g": 0.06,
        "lambda_y": [50000] * 4,
        "Lissajous_circle_time": 3700,
        "seed": seed,
    }
    obj = deepcf.DEEPCF_Tracking(parameters_deepc)
    traj_deepcf = obj.trajectory_tracking()
    print("DeepCF: ", obj.rss)
        
    # Visual parameters
    visual_params = {
        "name": "∞-shape tracking with the same dataset N=19",
        "xmin": -110,
        "ymin": -110,
        "xmax": 110,
        "ymax": 110,
        "vehicle_length": 5,
        "vehicle_width": 2,
    }
    animation_obj = graph.graph_compete(visual_params)
    animation_obj.add_state_path(traj_deepc, 'r', name='DeepC')
    animation_obj.add_state_path(traj_deepcf, 'b', name='DeepCF')
    animation_obj.add_state_landscape(obj.trajectory)
    animation_obj.compression(10)
    animation_obj.generate_gif(name='img/DeePC-Fragmentation.gif')
    
def try_run_experiment(parameters):
        try:
            return run_experiment(parameters)
        except:
            return np.nan
    
def parallel_dataset_variaion(n_seeds = 30):
    """
    Plot the effect of dataset size (N) on the tracking deviation.
    """
    deepc_list = []
    dataset_list = range(2, 25)
    execution_list = []
    for alg in ["deepc", "deepcf"]:
        for N in dataset_list:
            for seed in range(n_seeds):            
                parameters = {
                    "algorithm": alg,
                    "prediction_horizon": 8,
                    "Q": [1, 1, 1, 100],
                    "R": [0.1] * 2,
                    "N": N,
                    "lambda_g": 0.06,
                    "lambda_y": [50000] * 4,
                    "Lissajous_circle_time": 3700,
                    "seed": seed,
                }
                execution_list.append(parameters)
    
    pool = multiprocessing.Pool(processes=90)
    results = pool.map(try_run_experiment, execution_list)
    data = {
        "deepc": [],
        "deepcf": [],
        "deepc_var": [],
        "deepcf_var": [],
    }
    for alg in ["deepc", "deepcf"]:
        for N in dataset_list:
            seed_list = []
            for seed in range(n_seeds):
                seed_list.append(results.pop(0))
            data[alg].append(np.nanmean(seed_list))
            data[alg + "_var"].append(np.nanvar(seed_list))   
    
    if True:
        fig = plt.figure()
        plt.plot(dataset_list, data["deepc"], label='DeepC')
        plt.plot(dataset_list, data["deepcf"], label='DeepCF')
        plt.fill_between(dataset_list, 
                         np.array(data["deepc"]) - np.array(data["deepc_var"]), 
                         np.array(data["deepc"]) + np.array(data["deepc_var"]), 
                         alpha=0.2)
        plt.fill_between(dataset_list, 
                         np.array(data["deepcf"]) - np.array(data["deepcf_var"]), 
                         np.array(data["deepcf"]) + np.array(data["deepcf_var"]), 
                         alpha=0.2)
        plt.xlabel('N')
        plt.ylabel('RSS')
        plt.yscale('log')
        plt.grid()
        plt.legend()
        plt.title('Effect of Dataset size (N) on RSS of tracking deviation')
        plt.savefig(f'img/Dataset-variation-parallel.pdf')
    
# Check if the script is being run as the main module
if __name__ == "__main__":
    #plot_lambda_y()
    #dataset_variation()
    #dataset_variation_high()
    #plot_lambda_g(N=25)
    #animation_of_deepcf()
    parallel_dataset_variaion(5)
    exit()
    