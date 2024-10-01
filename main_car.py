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
import DeepC.source.deepc_tracking as deepc_tracking
from DeepC.source import graph
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
                    "lambda_g": 0.03 if alg == "deepcf" else 0.1,
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
        fig = plt.figure(figsize=(7, 4))
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
        plt.xlabel('Dataset size, N')
        plt.ylabel('Residual sum of squares (RSS) per step, m')
        plt.yscale('symlog', linthresh=1)
        plt.xlim([5, 24])
        plt.ylim([0, 1000])
        plt.xticks(range(5, 25, 5))
        plt.grid()
        plt.legend()
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        plt.title('Effect of Dataset size (N) on RSS of tracking deviation')
        plt.savefig(f'img/Dataset-variation-parallel.pdf')
    
def plot_lambda_g_parallel(n_seeds = 30):
    """
    Plot the effect of dataset size (N) on the tracking deviation.
    """
    range_g = np.logspace(-3, 0, 30)
    execution_list = []
    for alg in ["deepc", "deepcf"]:
        for lambda_g in range_g:
            for seed in range(n_seeds):            
                parameters = {
                    "algorithm": alg,
                    "prediction_horizon": 8,
                    "Q": [1, 1, 1, 100],
                    "R": [0.1] * 2,
                    "N": 25,
                    "lambda_g": lambda_g,
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
        for N in range_g:
            seed_list = []
            for seed in range(n_seeds):
                if results[0] > 10:
                    _ = results.pop(0)
                else:
                    seed_list.append(results.pop(0))
            data[alg].append(np.nanmean(seed_list))
            data[alg + "_var"].append(np.nanvar(seed_list))   
    
    if True:
        fig = plt.figure(figsize=(7, 4))
        plt.plot(range_g.tolist(), data["deepc"], label='DeepC')
        plt.plot(range_g.tolist(), data["deepcf"], label='DeepCF')
        #plt.fill_between(range_g.tolist(), 
        #                 np.array(data["deepc"]) - np.array(data["deepc_var"]), 
        #                 np.array(data["deepc"]) + np.array(data["deepc_var"]), 
        #                 alpha=0.2)
        #plt.fill_between(range_g.tolist(), 
        #                 np.array(data["deepcf"]) - np.array(data["deepcf_var"]), 
        #                 np.array(data["deepcf"]) + np.array(data["deepcf_var"]), 
        #                 alpha=0.2)
        plt.plot(0.15, 1.15,'*', color='b')
        plt.plot(0.06, 1.1,'*', color='orange')
        plt.xlabel('λ coefficient')
        plt.ylabel('Residual sum of squares (RSS) per step, m')
        plt.xscale('log')
        plt.xlim([0.001, 1])
        #plt.yscale('log')
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        plt.grid()
        plt.legend()
        plt.title('Effect of λ on RSS of tracking deviation for N=25')
        plt.savefig(f'img/Lambda-g-variation-parallel.pdf')
    
def Lissague_tracking_parallel(n_seeds = 30):
    """
    Plot the effect of Circle time on the tracking deviation.
    """
    circle_time = range(3500, 4200, 50)
    execution_list = []
    for N in [19, 25]:
        for alg in ["deepc", "deepcf"]:
            for circle in circle_time:
                for seed in range(n_seeds):            
                    parameters = {
                        "algorithm": alg,
                        "prediction_horizon": 8,
                        "Q": [1, 1, 1, 100],
                        "R": [0.1] * 2,
                        "N": N,
                        "lambda_g": 0.03 if alg == "deepcf" else 0.1,
                        "lambda_y": [50000] * 4,
                        "Lissajous_circle_time": circle,
                        "seed": seed,
                    }
                    execution_list.append(parameters)
    
    pool = multiprocessing.Pool(processes=90)
    results = pool.map(try_run_experiment, execution_list)
    data = {
        "deepc25": [],
        "deepcf25": [],
        "deepc19": [],
        "deepcf19": [],
        "deepc_var": [],
        "deepcf_var": [],
    }
    for N in [19, 25]:
        for alg in ["deepc", "deepcf"]:
            for _ in circle_time:
                seed_list = []
                for seed in range(n_seeds):
                    seed_list.append(results.pop(0))
                data[alg+str(N)].append(np.nanmean(seed_list))
                if N == 19:
                    data[alg + "_var"].append(np.nanvar(seed_list))   
    
    if True:
        argument = (np.array(circle_time)/100).tolist()
        fig = plt.figure(figsize=(7, 4))
        plt.plot(argument, data["deepc25"], label='DeepC, N=25', color='tab:blue', linestyle='--')
        plt.plot(argument, data["deepcf25"], label='DeepCF, N=25', color='tab:orange', linestyle='--')
        plt.plot(argument, data["deepc19"], label='DeepC, N=19', color='tab:blue')
        plt.plot(argument, data["deepcf19"], label='DeepCF, N=19', color='tab:orange')
        plt.fill_between(argument, 
                         np.array(data["deepc19"]) - np.array(data["deepc_var"]), 
                         np.array(data["deepc19"]) + np.array(data["deepc_var"]), 
                         alpha=0.2, color='tab:blue')
        plt.fill_between(argument, 
                         np.array(data["deepcf19"]) - np.array(data["deepcf_var"]), 
                         np.array(data["deepcf19"]) + np.array(data["deepcf_var"]), 
                         alpha=0.2, color='tab:orange')
        plt.xlabel('Total time to complete the whole lap, s')
        plt.ylabel('Residual sum of squares (RSS) per step, m')
        plt.yscale('symlog', linthresh=1)
        plt.xlim([35, 41.5])
        plt.ylim([0.3, 5])
        #plt.xticks(range(5, 25, 5))
        plt.grid()
        plt.legend()
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        plt.title('Eight-shape trajectory tracking')
        plt.savefig(f'img/Circle-time-parallel.pdf')
        
        
# Check if the script is being run as the main module
if __name__ == "__main__":
    #plot_lambda_y()
    #dataset_variation()
    #dataset_variation_high()
    #plot_lambda_g(N=25)
    #animation_of_deepcf()
    #parallel_dataset_variaion(100)
    plot_lambda_g_parallel(50)
    #Lissague_tracking_parallel(30)
    exit()
    