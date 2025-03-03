#!/usr/bin/env python
# This line indicates that the script should be run using the Python interpreter.

import sys

# Importing the sys module to manipulate the Python runtime environment.

sys.path.insert(1, "source/")
# Inserting 'source/' at position 1 in the sys.path list, to allow importing modules from this directory.

import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib import cm
import source.deepcf as deepcf
import source.deepce as deepce
from source.pendulum import Pendulum_tracking
from source.linear import LinearSystem, Linear_tracking
import json
from source.nonlinear import DoublePendulum
from source.nonlinear import Nonlinear_tracking
import scipy.io
import pandas as pd
import seaborn as sns
import seaborn.objects as so
import matplotlib.cm as cm

def hyperparameter_tuning():
    n_seeds = 30
    fig, ax = plt.subplots(2, 1, figsize=(5, 6))

    control_list = range(10)
    plot_list = []
    shadow_list = []
    lambda_list = [2.0, 5.0, 8.0, 11.0, 14.0]
    alg = "deepcgf"

    if True:
        for control in control_list:
            error_list = []
            
            for seed in range(0, n_seeds):
                # Define the parameters for the experiment
                params = {
                    "Q": [1.0],
                    "R": [0.0],
                    "lambda_g": 7.5,
                    "N": 300,
                    "lambda_y": [0.5],
                    "algorithm": alg,
                    "initial_horizon": 5,
                    "control_horizon": control,
                    "prediction_horizon": 10,
                    "seed": seed,
                    "noise": False,
                }
                try:
                    obj = Linear_tracking(params)
                    _ = obj.trajectory_tracking()
                    error_list.append(float(obj.error))
                except:
                    continue
                

            plot_list.append(np.mean(error_list))
            shadow_list.append(np.std(error_list))

        ax[0].plot(control_list, plot_list, label="Fragmented", color="blue")
        ax[0].fill_between(control_list, np.array(plot_list) - np.array(shadow_list), np.array(plot_list) + np.array(shadow_list), color="blue", alpha=0.1)

    ax[0].set_xlabel("Fragmented prediction horizon, steps")
    ax[0].set_ylabel("Sum of Set-point errors (m)")
    ax[0].legend()
    ax[0].grid(True)

    if True:
        plot_list = []
        shadow_list = []
        for lambda_g in lambda_list:
            error_list = []
            
            for seed in range(0, n_seeds):
                # Define the parameters for the experiment
                params = {
                    "Q": [1.0],
                    "R": [0.0],
                    "lambda_g": lambda_g,
                    "N": 200,
                    "lambda_y": [0.5],
                    "algorithm": alg,
                    "initial_horizon": 5,
                    "control_horizon": 3,
                    "prediction_horizon": 10,
                    "seed": seed,
                    "noise": False,
                }
                try:
                    obj = Linear_tracking(params)
                    _ = obj.trajectory_tracking()
                    error_list.append(float(obj.error))
                except:
                    continue
                

            plot_list.append(np.mean(error_list))
            shadow_list.append(np.std(error_list))

        ax[1].plot(lambda_list, plot_list, label="Fragmented", color="blue")
        ax[1].fill_between(lambda_list, np.array(plot_list) - np.array(shadow_list), np.array(plot_list) + np.array(shadow_list), color="blue", alpha=0.1)
    ax[1].set_xlabel(r"Regularization weight $\lambda_g$")
    ax[1].set_ylabel("Sum of Set-point errors (m)")
    ax[1].legend()
    ax[1].grid(True)
    plt.tight_layout()
    plt.savefig("img/hyperparam.pdf")

def nonlinear_chart():
    # Example of simulation loop
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    for i in range(3):
        parameters = {
            "k": 0.0,
            "c": 10.0**((i-2)/1),
            "m2": 0.5,
            "l2": 0.5,
        }
        pendulum = DoublePendulum(parameters)
        pendulum.Initialization()
        
        for j in range(100):
            
            torque = 10 if j < 50 else 0
            pendulum.Step(torque)
            
            #pendulum.visualize()
        pendulum.post_visualization(ax[2-i])
        ax[2-i].set_title(f"Damping coefficient = {parameters['c']}")
    fig.set_tight_layout(True)
    fig.suptitle("Step Response of Double Pendulum for Varying Damping Coefficients", fontsize=16)
    fig.savefig("img/double_pendulum.pdf")

def try_violin():

    data = scipy.io.loadmat("data/pendulum_violin.mat")  
    trajectories = data["trajectories"]  

    data = []
    Damping_list = [1.0, 0.5, 0.25, 0.125]
    for i in range(4):
        for j in range(100):
            data.append({"Algorithm": "Original", "Damping factor": Damping_list[i], "Sum of Set-point Errors": trajectories[i, j]})
            data.append({"Algorithm": "Segmented", "Damping factor": Damping_list[i], "Sum of Set-point Errors": trajectories[i+4, j]})

    # Generate some data
    if True:
        for damping in [1.0, 0.5, 0.25, 0.125]:
            for seed in range(100):
                parameters = {
                    "k": 0.0,
                    "c": damping,
                    "m2": 0.5,
                    "l2": 0.5,
                    "dt": 0.2,
                    "N": 100-8,
                    "tracking_time": 100,
                    "algorithm": "deepcgf",
                    "lambda_g": 7.5,
                    "control_horizon": 3,
                    "prediction_horizon": 10,
                    "seed": seed,
                }
                obj = Nonlinear_tracking(parameters)
                _ = obj.trajectory_tracking()
                data.append({"Algorithm": "Fragmented", "Damping factor": damping, "Sum of Set-point Errors": obj.error})

    df = pd.DataFrame(data)

    # Create the violin plot
    plt.figure(figsize=(8, 5))
    #sns.violinplot(x="Damping factor", y="Value", hue="Algorithm", data=df, palette="Set2", inner="quart")
    sns.boxplot(x="Damping factor", y="Sum of Set-point Errors", hue="Algorithm", data=df, palette="viridis", gap=0.1, hue_order=["Fragmented", "Segmented", "Original"])

    # Add title and labels
    #plt.title("Violin Plot with Damping factor and Category")
    plt.xlabel("Damping factor")
    plt.ylabel("Sum of Set-point Errors")
    plt.ylim(5, 10**4)
    plt.yscale('log')
    plt.tight_layout()
    plt.grid(True)
    plt.savefig("img/violin_plot.pdf")
    
def try_dataset():

    data = scipy.io.loadmat("data/pendulum_dataset.mat")  
    trajectories = data["trajectories"]  
    data = []
    dataset = [30, 50, 100, 200]
    for i, dataset_size in enumerate(dataset):
        for j in range(100):
            data.append({"Algorithm": "Segmented", "Dataset size": dataset_size, "Sum of Set-point Errors": trajectories[i+len(dataset), j]})
            data.append({"Algorithm": "Original", "Dataset size": dataset_size, "Sum of Set-point Errors": trajectories[i, j]})
            

    # Generate some data
    if True:
        for N in dataset:
            for seed in range(100):
                parameters = {
                    "k": 0.0,
                    "c": 0.25,
                    "m2": 0.5,
                    "l2": 0.5,
                    "dt": 0.2,
                    "N": N-8,
                    "tracking_time": 100,
                    "algorithm": "deepcgf",
                    "lambda_g": 7.5,
                    "control_horizon": 3,
                    "prediction_horizon": 10,
                    "seed": seed,
                }
                obj = Nonlinear_tracking(parameters)
                _ = obj.trajectory_tracking()
                data.append({"Algorithm": "Fragmented", "Dataset size": N, "Sum of Set-point Errors": obj.error})

    df = pd.DataFrame(data)


    # Create the violin plot
    plt.figure(figsize=(8, 5))

    # Count the number of points higher than 100 for each dataset size and algorithm
    count_data = df[df["Sum of Set-point Errors"] > 100].groupby(["Dataset size", "Algorithm"]).size().reset_index(name='Count')
    count_data.loc[len(count_data)] = [200, "Fragmented", 0]
    
    #sns.violinplot(x="Damping factor", y="Value", hue="Algorithm", data=df, palette="Set2", inner="quart")
    sns.boxplot(x="Dataset size", y="Sum of Set-point Errors", hue="Algorithm", data=df, palette="viridis", gap=0.1, hue_order=["Fragmented", "Segmented", "Original"])
    viridis = plt.get_cmap()
    
    # Add title and labels
    #plt.title("Violin Plot with Damping factor and Category")
    plt.xlabel("Dataset size")
    plt.ylabel("Sum of Set-point Errors")
    #plt.ylim(0, 99)
    plt.tight_layout()
    plt.yscale('log')
    
    plt.grid(True)
    plt.savefig("img/dataset.pdf")

def try_linear():
    data = scipy.io.loadmat("data/linear.mat")  
    trajectories = data["trajectories"]  
    data = []
    dataset = [30, 40, 50, 60, 100]
    for i in dataset:
        for j in range(100):
            value = trajectories[int(i/10)-1, 100+j]
            if value > 100:
                value = 100
            data.append(
                {
                    "Algorithm": "Segmented", 
                    "Dataset size": i, 
                    "Sum of Set-point Errors": value
                }
            )
            value = trajectories[int(i/10)-1, j]
            if value > 100:
                value = 100
            data.append(
                {
                    "Algorithm": "Original", 
                    "Dataset size": i, 
                    "Sum of Set-point Errors": value
                }
            )
            

    # Generate some data
    alg = "deepcgf"
    if True:
        for N in dataset:
            for seed in range(100):                
                params = {
                    "Q": [1.0],
                    "R": [0.0],
                    "lambda_g": 7.5,
                    "N": N-8,
                    "lambda_y": [0.5],
                    "algorithm": alg,
                    "initial_horizon": 5,
                    "control_horizon": 3,
                    "prediction_horizon": 10,
                    "seed": seed,
                    "noise": True,
                }
                obj = Linear_tracking(params)
                _ = obj.trajectory_tracking()
                data.append({"Algorithm": "Fragmented", "Dataset size": N, "Sum of Set-point Errors": obj.error})

    df = pd.DataFrame(data)


    # Create the violin plot
    fig, ax = plt.subplots(figsize=(8, 5))

    # Count the number of points higher than 100 for each dataset size and algorithm
    count_data = df[df["Sum of Set-point Errors"] > 100].groupby(["Dataset size", "Algorithm"]).size().reset_index(name='Count')
    count_data.loc[len(count_data)] = [200, "Fragmented", 0]

    #sns.violinplot(x="Damping factor", y="Value", hue="Algorithm", data=df, palette="Set2", inner="quart")
    sns.boxplot(x="Dataset size", y="Sum of Set-point Errors", hue="Algorithm", data=df, palette="viridis", gap=0.1, hue_order=["Fragmented", "Segmented", "Original"])
    viridis = plt.get_cmap()
    ax.legend(loc='upper right')

    # Add title and labels
    #plt.title("Violin Plot with Damping factor and Category")
    plt.xlabel("Amount of data, L")
    plt.ylabel("Sum of Set-point Errors")
    plt.ylim(0, 105)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.02)
    #plt.yscale('log')
    
    plt.grid(True)
    plt.savefig("img/linearDopoTh.pdf")

def chart_example():
    plt.figure(figsize=(8, 5))
    # Plot the chart
    prediction_list = [10]
    error_dict = {}
    alg = "deepcgf"

    if True:
        result_list = []
        for prediction_horizon in prediction_list:
            error_list = []
            
            for seed in range(0, 100):
                # Define the parameters for the experiment
                params = {
                    "Q": [1.0],
                    "R": [0.0],
                    "lambda_g": 7.5,
                    "N": 50-8,
                    "lambda_y": [0.5],
                    "algorithm": alg,
                    "initial_horizon": 5,
                    "control_horizon": 3,
                    "prediction_horizon": prediction_horizon,
                    "seed": seed,
                    "noise": True,
                }
                obj = Linear_tracking(params)
                result = obj.trajectory_tracking()
                if prediction_horizon == 10:
                    result_list.append(result)
                error_list.append(float(obj.error))
                #if prediction_horizon == 10:
                #    if seed == 0:
                #        ax[0].plot(result, color='blue', alpha=0.5, linewidth=0.5, label="Fragmented")
                #    elif seed < 20:
                #        ax[0].plot(result, color='blue', alpha=0.5, linewidth=0.5)

            error_dict[str(prediction_horizon)] = {
                "mean": np.mean(error_list),
                "std": np.std(error_list),
                "min": np.min(error_list),
                "max": np.max(error_list),
            }

        # Plot average trajectory
        avg_frag_trajectory = np.mean(result_list, axis=0)
        std_frag_trajectory = np.std(result_list, axis=0)
        plt.plot(obj.trajectory.tolist()[params["initial_horizon"]:105], label="Reference", linestyle="--", color="black")
        plt.plot(avg_frag_trajectory, color=cm.viridis(0.0), label="Fragmented")
        plt.fill_between(range(len(avg_frag_trajectory)), avg_frag_trajectory - std_frag_trajectory, avg_frag_trajectory + std_frag_trajectory, color=cm.viridis(0.0), alpha=0.15)
        
    

    # Matlab data
    data = scipy.io.loadmat("data/trajectories.mat")
    variable_name = "trajectories"
    trajectories = data[variable_name]

    # Calculate the average trajectory for the specified range
    avg_seg_trajectory = np.mean(trajectories[:, 200:300], axis=1)
    std_seg_trajectory = np.std(trajectories[:, 200:300], axis=1)
    
    # Plot the average trajectory with standard deviation
    plt.plot(avg_seg_trajectory, color=cm.viridis(0.4), label="Segmented")
    plt.fill_between(range(len(avg_seg_trajectory)), avg_seg_trajectory - std_seg_trajectory, avg_seg_trajectory + std_seg_trajectory, color=cm.viridis(0.4), alpha=0.15)
    
    # Calculate the average trajectory for the specified range
    avg_orig_trajectory = np.mean(trajectories[:, 0:100], axis=1)
    std_orig_trajectory = np.std(trajectories[:, 0:100], axis=1)
    
    # Plot the average trajectory with standard deviation
    plt.plot(avg_orig_trajectory, color=cm.viridis(0.9), label="Original")
    plt.fill_between(range(len(avg_orig_trajectory)), avg_orig_trajectory - std_orig_trajectory, avg_orig_trajectory + std_orig_trajectory, color=cm.viridis(0.9), alpha=0.15)

    
    plt.xlabel("Time (s)")
    plt.ylabel("Displacement $x_2$, (m)")
    plt.legend()
    plt.grid(True)

    # Save the plot
    plt.tight_layout()
    plt.savefig("img/LTI50.pdf")

def chart_example2():
    plt.figure(figsize=(8, 5))
    # Plot the chart
    prediction_list = [10]
    error_dict = {}
    alg = "deepcgf"

    if True:
        result_list = []
        for prediction_horizon in prediction_list:
            error_list = []
            
            for seed in range(0, 100):
                # Define the parameters for the experiment
                params = {
                    "Q": [1.0],
                    "R": [0.0],
                    "lambda_g": 7.5,
                    "N": 40-8,
                    "lambda_y": [0.5],
                    "algorithm": alg,
                    "initial_horizon": 5,
                    "control_horizon": 3,
                    "prediction_horizon": prediction_horizon,
                    "seed": seed,
                    "noise": True,
                }
                obj = Linear_tracking(params)
                result = obj.trajectory_tracking()
                if prediction_horizon == 10:
                    result_list.append(result)
                error_list.append(float(obj.error))
                #if prediction_horizon == 10:
                #    if seed == 0:
                #        ax[0].plot(result, color='blue', alpha=0.5, linewidth=0.5, label="Fragmented")
                #    elif seed < 20:
                #        ax[0].plot(result, color='blue', alpha=0.5, linewidth=0.5)

            error_dict[str(prediction_horizon)] = {
                "mean": np.mean(error_list),
                "std": np.std(error_list),
                "min": np.min(error_list),
                "max": np.max(error_list),
            }

        # Plot average trajectory
        avg_frag_trajectory = np.mean(result_list, axis=0)
        std_frag_trajectory = np.std(result_list, axis=0)
        plt.plot(obj.trajectory.tolist()[params["initial_horizon"]:105], label="Reference", linestyle="--", color="black")
        plt.plot(avg_frag_trajectory, color=cm.viridis(0.0), label="Fragmented")
        plt.fill_between(range(len(avg_frag_trajectory)), avg_frag_trajectory - std_frag_trajectory, avg_frag_trajectory + std_frag_trajectory, color=cm.viridis(0.0), alpha=0.15)
        
    

    # Matlab data
    data = scipy.io.loadmat("data/trajectories2.mat")
    variable_name = "trajectories"
    trajectories = data[variable_name]

    # Calculate the average trajectory for the specified range
    avg_seg_trajectory = np.mean(trajectories[:, 200:300], axis=1)
    std_seg_trajectory = np.std(trajectories[:, 200:300], axis=1)
    
    # Plot the average trajectory with standard deviation
    plt.plot(avg_seg_trajectory, color=cm.viridis(0.4), label="Segmented")
    plt.fill_between(range(len(avg_seg_trajectory)), avg_seg_trajectory - std_seg_trajectory, avg_seg_trajectory + std_seg_trajectory, color=cm.viridis(0.4), alpha=0.15)
    
    # Calculate the average trajectory for the specified range
    avg_orig_trajectory = np.mean(trajectories[:, 0:100], axis=1)
    std_orig_trajectory = np.std(trajectories[:, 0:100], axis=1)
    
    # Plot the average trajectory with standard deviation
    plt.plot(avg_orig_trajectory, color=cm.viridis(0.9), label="Original")
    plt.fill_between(range(len(avg_orig_trajectory)), avg_orig_trajectory - std_orig_trajectory, avg_orig_trajectory + std_orig_trajectory, color=cm.viridis(0.9), alpha=0.15)

    
    plt.xlabel("Time (s)")
    plt.ylabel("Displacement $x_2$, (m)")
    plt.legend()
    plt.grid(True)

    # Save the plot
    plt.tight_layout()
    plt.savefig("img/LTI40.pdf")

if __name__ == "__main__":    
    #nonlinear_chart()
    #hyperparameter_tuning()
    #try_dataset()
    #try_violin()
    #try_linear()
    chart_example2()
