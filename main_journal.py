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
    n_seeds = 1000
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))

    control_list = range(10)
    plot_list = []
    shadow_list = []
    lambda_list = []#[2.0, 5.0, 8.0, 11.0, 14.0]#range(3, 15) #[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    alg = "deepcgf"

    for control in control_list:
        error_list = []
        
        for seed in range(0, n_seeds):
            # Define the parameters for the experiment
            params = {
                "Q": [1.0],
                "R": [0.0],
                "lambda_g": 0.5,
                "N": 20,
                "lambda_y": [0.5],
                "algorithm": alg,
                "initial_horizon": 5,
                "control_horizon": control,
                "prediction_horizon": 10,
                "seed": seed,
                "noise": True,
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
    #ax[0].set_title("Control horizon optimization")
    ax[0].legend()
    ax[0].grid(True)

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
    #ax[1].set_title("Regularization optimization")
    ax[1].legend()
    ax[1].grid(True)
    plt.tight_layout()
    plt.savefig("img/hyperparam.pdf")

def varyingN():
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), width_ratios=[3, 1])
    fig_traj,ax_traj = plt.subplots(figsize=(10, 5))
    fig_bar,ax_bar = plt.subplots(figsize=(5, 5))
    # Plot the chart
    prediction_list = [10, 20, 40]
    error_dict = {}
    alg = "deepcgf"

    result_list = []
    for prediction_horizon in prediction_list:
        error_list = []
        
        for seed in range(0, 100):
            # Define the parameters for the experiment
            params = {
                "Q": [1.0],
                "R": [0.0],
                "lambda_g": 7.5,
                "N": 200,
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
    ax[0].plot(avg_frag_trajectory, color='blue')
    ax[0].fill_between(range(len(avg_frag_trajectory)), avg_frag_trajectory - std_frag_trajectory, avg_frag_trajectory + std_frag_trajectory, color='blue', alpha=0.15)
    ax_traj.plot(avg_frag_trajectory, color='blue')
    ax_traj.fill_between(range(len(avg_frag_trajectory)), avg_frag_trajectory - std_frag_trajectory, avg_frag_trajectory + std_frag_trajectory, color='blue', alpha=0.15)
    

    # Matlab data
    data = scipy.io.loadmat("data/linear.mat")
    variable_name = "trajectories"    
    # Extract the trajectories variable
    if variable_name not in data:
        raise KeyError(f"Variable '{variable_name}' not found in the .mat file.")    
    trajectories = data[variable_name]    
    # Ensure trajectories is 2D
    if trajectories.ndim != 2:
        raise ValueError("Trajectories data must be a 2D array.")
    # Calculate the average trajectory for the specified range
    avg_seg_trajectory = np.mean(trajectories[:, 100:200], axis=1)
    std_seg_trajectory = np.std(trajectories[:, 100:200], axis=1)
    
    # Plot the average trajectory with standard deviation
    ax[0].plot(avg_seg_trajectory, color='orange')
    ax[0].fill_between(range(len(avg_seg_trajectory)), avg_seg_trajectory - std_seg_trajectory, avg_seg_trajectory + std_seg_trajectory, color='orange', alpha=0.15)
    ax_traj.plot(avg_seg_trajectory, color='orange')
    ax_traj.fill_between(range(len(avg_seg_trajectory)), avg_seg_trajectory - std_seg_trajectory, avg_seg_trajectory + std_seg_trajectory, color='orange', alpha=0.15)
    
    # Calculate the average trajectory for the specified range
    avg_orig_trajectory = np.mean(trajectories[:, 0:100], axis=1)
    std_orig_trajectory = np.std(trajectories[:, 0:100], axis=1)
    
    # Plot the average trajectory with standard deviation
    ax[0].plot(avg_orig_trajectory, color='green')
    ax[0].fill_between(range(len(avg_orig_trajectory)), avg_orig_trajectory - std_orig_trajectory, avg_orig_trajectory + std_orig_trajectory, color='green', alpha=0.15)
    ax_traj.plot(avg_orig_trajectory, color='green')
    ax_traj.fill_between(range(len(avg_orig_trajectory)), avg_orig_trajectory - std_orig_trajectory, avg_orig_trajectory + std_orig_trajectory, color='green', alpha=0.15)

    ax[0].plot(obj.trajectory.tolist()[params["initial_horizon"]:105], label="Reference", linestyle="--", color="black")
    ax[0].set_xlabel("Time (s)")
    ax[0].set_ylabel("Displacement $x_2$, (m)")
    ax[0].set_title("DeePC with linear function: Linear Tracking")
    ax[0].legend()
    ax[0].grid(True)

    ax_traj.plot(obj.trajectory.tolist()[params["initial_horizon"]:105], linestyle="--", color="black")
    ax_traj.set_xlabel("Time (s)")
    ax_traj.set_ylabel("Displacement $x_2$, (m)")
    #ax_traj.set_title("DeePC with linear function: Linear Tracking")
    #ax_traj.legend()
    ax_traj.grid(True)
    ax_traj.set_xlim(0, 100)

    # Read from the JSON file
    with open("data/matlab_outcomes.json", "r") as file:
        data = json.load(file)
        data["Fragmented"] = error_dict

    # Create the candlestick plot
    alg_list = ["Fragmented", "Segmented", "Original"]
    color_dict = {"Fragmented": "blue", "Segmented": "orange", "Original": "green"}
    for zu, prediction_horizon in enumerate(prediction_list):
        for i, alg in enumerate(alg_list):
            idx = i + zu * len(alg_list)
            alg_obj = data[alg][str(prediction_horizon)]
            ax[1].plot([idx, idx], [alg_obj["min"], alg_obj["max"]], color='black')
            ax[1].plot([idx - 0.1, idx + 0.1], [alg_obj["min"], alg_obj["min"]], color='black')
            ax[1].plot([idx - 0.1, idx + 0.1], [alg_obj["max"], alg_obj["max"]], color='black')
            ax[1].plot([idx - 0.2, idx + 0.2], [alg_obj["mean"], alg_obj["mean"]], color='red')
            ax[1].add_patch(plt.Rectangle((idx - 0.4, alg_obj["mean"] - alg_obj["std"]), 0.8, 2 * alg_obj["std"], 
                                        color=color_dict[alg], alpha=0.3))
            ax_bar.plot([idx, idx], [alg_obj["min"], alg_obj["max"]], color='black')
            ax_bar.plot([idx - 0.1, idx + 0.1], [alg_obj["min"], alg_obj["min"]], color='black')
            ax_bar.plot([idx - 0.1, idx + 0.1], [alg_obj["max"], alg_obj["max"]], color='black')
            ax_bar.plot([idx - 0.2, idx + 0.2], [alg_obj["mean"], alg_obj["mean"]], color='red')
            ax_bar.add_patch(plt.Rectangle((idx - 0.4, alg_obj["mean"] - alg_obj["std"]), 0.8, 2 * alg_obj["std"], 
                                        color=color_dict[alg], alpha=0.3))

    # Set the x-axis labels
    x_len = len(prediction_list) * len(alg_list)
    ax[1].set_xticks(range(x_len))
    ax_bar.set_xticks(range(x_len))
    x_label = ["", "10", "", "", "20", "", "", "40", ""]
    print(x_label[0:x_len])
    ax[1].set_xticklabels(x_label[0:x_len])
    ax_bar.set_xticklabels(x_label[0:x_len])
    # Set the y-axis label
    ax[1].set_ylabel('Sum of set-point errors')
    ax_bar.set_ylabel('Sum of set-point errors')
    # Set the x-axis label
    ax[1].set_xlabel('Prediction horizon')
    ax_bar.set_xlabel('Prediction horizon')
    # Set the title
    ax[1].set_title('Tracking errors')
    #ax_bar.set_title('Tracking errors')
    patch_list = []
    for alg in alg_list:
        patch = mpatches.Patch(color=color_dict[alg], label=alg, alpha=0.3)
        patch_list.append(patch)
    ax[1].legend(handles=patch_list, loc=2)
    ax_bar.legend(handles=patch_list, loc=1)
    ax[0].legend(handles=patch_list, loc=3)
    ax_traj.legend(handles=patch_list, loc=3)
    ax_bar.grid(True)
    ax[1].grid(True)
    # Save the plot
    fig.tight_layout()
    fig.savefig("img/linear.pdf")
    fig_traj.tight_layout()
    fig_traj.savefig("img/LTI.pdf")
    fig_bar.tight_layout()
    fig_bar.savefig("img/linear_bar.pdf")

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

def nonlinear_track():
    time = np.arange(0.0, 20.0, 0.2)
    fig, ax = plt.subplots(2, 1, figsize=(8, 8))
    arguments = np.arange(0.2, 1.0, 0.01)
    values = []
    for seed, c in enumerate(arguments):
        parameters = {
            "k": 0.0,
            "c": 0.01,
            "m2": 0.5,
            "l2": 0.5,
            "dt": 0.2,
            "N": 200,
            "tracking_time": 100,
            "algorithm": "deepcgf",
            "lambda_g": 0.5,
            "control_horizon": 3,
            "prediction_horizon": 10,
            "seed": seed,
        }
        obj = Nonlinear_tracking(parameters)
        results = obj.trajectory_tracking()
        error = obj.error
        print(f"Error: {error}")
        return
        values.append(error)

    ax[0].plot(arguments, values, 'bo', label="Fragmented", mfc='none', markersize=5) #color=cm.viridis(1.0-c), label="Fragmented" if seed == 0 else "")
    ax[1].plot(arguments, values, 'bo', label="Fragmented", mfc='none', markersize=5) #color=cm.viridis(1.0-c), label="Fragmented" if seed == 0 else "")

    
    #ax[0].plot(time, obj.trajectory.tolist()[5:105], label="Reference", linestyle="--", color="black")
    ax[0].set_xlabel("Factor damping")
    ax[0].set_ylabel("Sum of Set-point Errors")
    
    ax[0].set_yscale('log')
    #ax[0].set_ylim(-np.pi/2, np.pi/2)
    #ax[0].set_xlim(0, 20)
    ax[0].grid(True)


    ### MATLAB PART
    lower_bound = 110
    upper_bound = 190
    arguments = np.arange(0.2, 1.0, 0.01)
    # Load the .mat file
    data = scipy.io.loadmat("data/pendulum.mat")
    variable_name = "trajectories"    
    # Extract the trajectories variable
    if variable_name not in data:
        raise KeyError(f"Variable '{variable_name}' not found in the .mat file.")    
    trajectories = data[variable_name]    
    # Ensure trajectories is 2D
    if trajectories.ndim != 2:
        raise ValueError("Trajectories data must be a 2D array.")    
    # Plot each trajectory
    #for i in range(lower_bound, upper_bound):  # Loop over columns
    #    ax[1].plot(time, trajectories[:, i], color=cm.viridis(1.0-((i-lower_bound)/100)), label="Segmented" if i == upper_bound-1 else "")
    
    #ax[1].plot(time, obj.trajectory.tolist()[5:105], label="Reference", linestyle="--", color="black")

    error_list = []
    for i in range(lower_bound, upper_bound):
        matlab_trajectory = trajectories[:, i]
        fragmented_trajectory = obj.trajectory.tolist()[5:105]
        error = np.sum(np.abs(matlab_trajectory - fragmented_trajectory))
        error_list.append(error)
    
    ax[0].plot(arguments, error_list, 'rx', label="Segmented", markersize=5)


    error_list_orig = []
    for i in range(lower_bound-100, upper_bound-100):
        matlab_trajectory = trajectories[:, i]
        fragmented_trajectory = obj.trajectory.tolist()[5:105]
        error = np.sum(np.abs(matlab_trajectory - fragmented_trajectory))
        error_list_orig.append(error)
    
    ax[1].plot(arguments, error_list_orig, 'gx', label="Original", markersize=5)

    # Calculate the moving average of error_list
    window_size = 9
    moving_avg = np.convolve(error_list, np.ones(window_size)/window_size, mode='valid')
    moving_avg_orig = np.convolve(error_list_orig, np.ones(window_size)/window_size, mode='valid')
    moving_avg_frag = np.convolve(values, np.ones(window_size)/window_size, mode='valid')
    
    # Plot the moving average
    ax[0].plot(arguments[window_size//2:len(moving_avg)+window_size//2], moving_avg, 'r-', markersize=5)
    ax[0].plot(arguments[window_size//2:len(moving_avg_frag)+window_size//2], moving_avg_frag, 'b-', markersize=5)
    ax[1].plot(arguments[window_size//2:len(moving_avg_frag)+window_size//2], moving_avg_frag, 'b-', markersize=5)
    ax[1].plot(arguments[window_size//2:len(moving_avg_orig)+window_size//2], moving_avg_orig, 'g-', markersize=5)

    ax[0].legend()
    # Customize the plot
    ax[1].set_xlabel("Factor damping")
    ax[1].set_ylabel("Sum of Set-point Errors")
    ax[1].legend()
    ax[1].set_yscale('log')

    #ax[1].set_xlabel("Time (s)")
    #ax[1].set_ylabel("Angle $\theta$, (rad)")
    #ax[1].legend()
    #ax[1].set_ylim(-np.pi/2, np.pi/2)
    #ax[1].set_xlim(0, 20)
    ax[1].grid(True)
    
    # Show the plot
    plt.tight_layout()
    plt.savefig("img/pendulum_tracking.pdf")

def nonlinear_track_deleteme():
    fig = plt.figure(figsize=(8, 4))
    for seed, c in enumerate(np.arange(0.1, 1.0, 0.01)):
        parameters = {
            "k": 0.0,
            "c": 1.0,
            "m2": 0.5,
            "l2": 0.5,
            "dt": 0.2,
            "N": 300,
            "tracking_time": 100,
            "algorithm": "deepcgf",
            "lambda_g": 0.5,
            "control_horizon": 3,
            "prediction_horizon": 10,
            "seed": seed,
        }
        obj = Nonlinear_tracking(parameters)
        error = obj.trajectory_tracking()
        plt.plot(error, color=cm.viridis(1.0-c))

    
    plt.plot(obj.trajectory.tolist()[5:105], label="Reference", linestyle="--", color="black")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle $\theta$, (rad)")
    plt.title("Fragmented DeePC with nonlinear function: Nonlinear Tracking")
    plt.legend()
    plt.ylim(-np.pi/2, np.pi/2)
    plt.xlim(0, 100)
    plt.grid(True)
    plt.savefig("img/pendulum_tracking.pdf")

def pendulum_hyperparam(ax, parameter = "initial_horizon", parameter_range = range(2, 10)):
    n_seeds = 20
    plot_list = []
    std_list = []
    for param_value in parameter_range:
        seed_list = []
        for seed in range(n_seeds):
            parameters = {
                "k": 0.0,
                "c": 1.0,
                "m2": 0.5,
                "l2": 0.5,
                "dt": 0.2,
                "N": 300,
                "tracking_time": 100,
                "algorithm": "deepcgf",
                "lambda_g": 10,
                "control_horizon": 3,
                "prediction_horizon": 10,
                "initial_horizon": 5,
                "seed": seed,
            }
            parameters[parameter] = param_value
            obj = Nonlinear_tracking(parameters)
            obj.trajectory_tracking()
            seed_list.append(obj.error)
        plot_list.append(np.mean(seed_list))
        std_list.append(np.std(seed_list))
    ax.plot(parameter_range, plot_list, label="Fragmented")
    ax.fill_between(parameter_range, np.array(plot_list) - np.array(std_list), np.array(plot_list) + np.array(std_list), alpha=0.1)
    ax.set_xlabel(f"Parameter {parameter}")
    ax.set_ylabel("Angle deviation of tracking, (rad)")
    ax.legend()
    ax.grid(True)

def pendulum_growing():
    plt.figure(figsize=(8, 4))
    for c in np.arange(0.2, 1.0, 0.1):
        seed_list = []
        for horizon in range(2, 10):
            parameters = {
                "k": 0.0,
                "c": c,
                "m2": 0.5,
                "l2": 0.5,
                "dt": 0.2,
                "N": 300,
                "tracking_time": 100,
                "algorithm": "deepcgf",
                "lambda_g": 10,
                "control_horizon": 3,
                "prediction_horizon": 10,
                "initial_horizon": horizon,
                "seed": 1,
            }
            obj = Nonlinear_tracking(parameters)
            obj.trajectory_tracking()
            seed_list.append(obj.error)
        plt.plot(range(2, 10), seed_list, color=cm.viridis(1.0-c))
    plt.xlabel("Initial horizon")
    plt.ylabel("Angle deviation of tracking, (rad)")
    plt.legend()
    plt.yscale('log')
    plt.grid(True)
    plt.savefig("img/pendulum_growing.pdf")

def nonlinear_hyperparams():
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    pendulum_hyperparam(ax[0], "initial_horizon", range(2, 10))
    pendulum_hyperparam(ax[1], "lambda_g", [0.5, 1, 5, 10, 15])
    pendulum_hyperparam(ax[2], "control_horizon", [1, 2, 3, 5, 10])
    plt.tight_layout()
    plt.savefig("img/nonlinear_hyperparameter.pdf")

def plot_matlabs(mat_file_path, lower_bound, upper_bound, name = "Original",):
    """
    Plots all trajectories stored in a .mat file.

    Parameters:
        mat_file_path (str): Path to the .mat file.
        variable_name (str): Name of the variable storing trajectories.
    """
    # Load the .mat file
    data = scipy.io.loadmat(mat_file_path)

    variable_name = "trajectories"
    
    # Extract the trajectories variable
    if variable_name not in data:
        raise KeyError(f"Variable '{variable_name}' not found in the .mat file.")
    
    trajectories = data[variable_name]
    
    # Ensure trajectories is 2D
    if trajectories.ndim != 2:
        raise ValueError("Trajectories data must be a 2D array.")
    
    # Plot each trajectory
    plt.figure(figsize=(8, 4))
    for i in range(lower_bound, upper_bound):  # Loop over columns
        plt.plot(trajectories[:, i], color=cm.viridis(1.0-((i-lower_bound)/100)))
    
    # Customize the plot
    plt.xlabel("Time (s)")
    plt.ylabel("Angle $\theta$, (rad)")
    plt.title(f'{name} DeePC with nonlinear function: Nonlinear Tracking')
    plt.ylim(-np.pi/2, np.pi/2)
    plt.grid(True)
    
    # Show the plot
    plt.tight_layout()
    plt.savefig(f'img/matlab_tracking_{name}.pdf')

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
                    "N": 200,
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
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.grid(True)
    plt.savefig("img/violin_plot.pdf")
    
def try_dataset():

    data = scipy.io.loadmat("data/pendulum_dataset.mat")  
    trajectories = data["trajectories"]  
    data = []
    dataset = [30, 50, 100, 200]
    for i in range(4):
        for j in range(100):
            data.append({"Algorithm": "Segmented", "Dataset size": dataset[i], "Sum of Set-point Errors": trajectories[i+4, j]})
            data.append({"Algorithm": "Original", "Dataset size": dataset[i], "Sum of Set-point Errors": trajectories[i, j]})
            

    # Generate some data
    if True:
        for N in [20, 30, 50, 100, 200]:
            for seed in range(100):
                parameters = {
                    "k": 0.0,
                    "c": 0.25,
                    "m2": 0.5,
                    "l2": 0.5,
                    "dt": 0.2,
                    "N": N,
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
    fig, ax = plt.subplots(2, 1, figsize=(8, 5), height_ratios=[1, 2], )

    # Count the number of points higher than 100 for each dataset size and algorithm
    count_data = df[df["Sum of Set-point Errors"] > 100].groupby(["Dataset size", "Algorithm"]).size().reset_index(name='Count')
    count_data.loc[len(count_data)] = [200, "Fragmented", 0]

    # Plot the bar chart
    sns.barplot(x="Dataset size", y="Count", hue="Algorithm", data=count_data, palette="viridis", ax=ax[0], hue_order=["Fragmented", "Segmented", "Original"], gap=0.1)
    ax[0].set_xlabel("Dataset size")
    ax[0].set_ylabel("Count of Errors > 100")
    ax[0].grid(True)
    ax[0].xaxis.set_label_position('top') 
    ax[0].xaxis.tick_top()
    #ax[0].set_title("Count of Errors Greater Than 100 by Dataset Size and Algorithm")

    #sns.violinplot(x="Damping factor", y="Value", hue="Algorithm", data=df, palette="Set2", inner="quart")
    sns.boxplot(x="Dataset size", y="Sum of Set-point Errors", hue="Algorithm", data=df, palette="viridis", gap=0.1, hue_order=["Fragmented", "Segmented", "Original"], ax=ax[1])
    viridis = plt.get_cmap()
    # Add text label for no data
    ax[1].text(
        0.12, 60,                   # Position (x, y) in axes coordinates
        "N/D",                # The text content
        fontsize=12,                  # Font size
        ha='right', va='bottom',         # Align the text
        alpha=0.9,
    ).set_bbox(dict(facecolor=cm.viridis(0.5), alpha=0.9, edgecolor='#2f2f2f', boxstyle='round'))
    ax[1].text(
        0.43, 60,                   # Position (x, y) in axes coordinates
        "N/D",                # The text content
        fontsize=12,                  # Font size
        ha='right', va='bottom',         # Align the text
        alpha=0.9,
        color='#2f2f2f'
    ).set_bbox(dict(facecolor=cm.viridis(0.7), alpha=0.9, edgecolor='#2f2f2f', boxstyle='round'))
    ax[1].text(
        1.43, 60,                   # Position (x, y) in axes coordinates
        "N/D",                # The text content
        fontsize=12,                  # Font size
        ha='right', va='bottom',         # Align the text
        alpha=0.9,
        color='#2f2f2f'
    ).set_bbox(dict(facecolor=cm.viridis(0.7), alpha=0.9, edgecolor='#2f2f2f', boxstyle='round'))


    ax[0].legend(loc='upper right')
    ax[1].legend(loc='upper right')

    # Add title and labels
    #plt.title("Violin Plot with Damping factor and Category")
    plt.xlabel("Dataset size")
    plt.ylabel("Sum of Set-point Errors")
    plt.ylim(0, 99)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.02)
    #plt.yscale('log')
    
    plt.grid(True)
    plt.savefig("img/dataset.pdf")

# Check if the script is being run as the main module
if __name__ == "__main__": 
    #varyingN()
    #hyperparameter_tuning()
    #nonlinear_track()
    #nonlinear_hyperparams()
    #pendulum_growing()
    #plot_matlabs("pendulum.mat", 0, 90, "Original")
    try_dataset()
    #try_violin()
    print("Hi, I am the main module.")
