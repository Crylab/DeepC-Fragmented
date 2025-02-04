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

def hyperparameter_tuning():
    n_seeds = 20
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))

    control_list = []
    plot_list = []
    shadow_list = []
    lambda_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    alg = "deepcgf"

    for control in control_list:
        error_list = []
        
        for seed in range(0, n_seeds):
            # Define the parameters for the experiment
            params = {
                "Q": [1.0],
                "R": [0.0],
                "lambda_g": 8,
                "N": 300,
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
    ax[0].set_xlabel("Control horizon, s")
    ax[0].set_ylabel("Displacement $x_2$, (m)")
    ax[0].set_title("Control horizon optimization")
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
                "N": 300,
                "lambda_y": [0.5],
                "algorithm": alg,
                "initial_horizon": 5,
                "control_horizon": 3,
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

    ax[1].plot(lambda_list, plot_list, label="Fragmented", color="blue")
    ax[1].fill_between(lambda_list, np.array(plot_list) - np.array(shadow_list), np.array(plot_list) + np.array(shadow_list), color="blue", alpha=0.1)
    ax[1].set_xlabel(r"Regularization weight $\lambda_g$")
    ax[1].set_ylabel("Displacement $x_2$, (m)")
    ax[1].set_title("Regularization optimization")
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

    for prediction_horizon in prediction_list:
        error_list = []
        result_list = []
        for seed in range(0, 100):
            # Define the parameters for the experiment
            params = {
                "Q": [1.0],
                "R": [0.0],
                "lambda_g": 7.5,
                "N": 300,
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
    data = scipy.io.loadmat("linear.mat")
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
    with open("matlab_outcomes.json", "r") as file:
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
    ax_bar.legend(handles=patch_list, loc=2)
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
    for seed, c in enumerate(np.arange(0.1, 1.0, 0.01)):
        parameters = {
            "k": 0.0,
            "c": c,
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
        ax[0].plot(time, error, color=cm.viridis(1.0-c), label="Fragmented" if seed == 0 else "")

    
    ax[0].plot(time, obj.trajectory.tolist()[5:105], label="Reference", linestyle="--", color="black")
    ax[0].set_xlabel("Time (s)")
    ax[0].set_ylabel("Angle $\theta$, (rad)")
    ax[0].set_title("Double Pendulum Tracking: Fragmented DeePC vs. Segmented DeePC")
    ax[0].legend()
    ax[0].set_ylim(-np.pi/2, np.pi/2)
    ax[0].set_xlim(0, 20)
    ax[0].grid(True)


    ### MATLAB PART
    lower_bound = 100
    upper_bound = 190
    # Load the .mat file
    data = scipy.io.loadmat("pendulum.mat")
    variable_name = "trajectories"    
    # Extract the trajectories variable
    if variable_name not in data:
        raise KeyError(f"Variable '{variable_name}' not found in the .mat file.")    
    trajectories = data[variable_name]    
    # Ensure trajectories is 2D
    if trajectories.ndim != 2:
        raise ValueError("Trajectories data must be a 2D array.")    
    # Plot each trajectory
    for i in range(lower_bound, upper_bound):  # Loop over columns
        ax[1].plot(time, trajectories[:, i], color=cm.viridis(1.0-((i-lower_bound)/100)), label="Segmented" if i == upper_bound-1 else "")
    
    ax[1].plot(time, obj.trajectory.tolist()[5:105], label="Reference", linestyle="--", color="black")
    
    # Customize the plot
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("Angle $\theta$, (rad)")
    ax[1].legend()
    ax[1].set_ylim(-np.pi/2, np.pi/2)
    ax[1].set_xlim(0, 20)
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

def plot_matlabs(mat_file_path, lower_bound, upper_bound, name = "Original"):
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

# Check if the script is being run as the main module
if __name__ == "__main__": 
    varyingN()
    #hyperparameter_tuning()
    #nonlinear_track()
    #nonlinear_hyperparams()
    #pendulum_growing()
    #plot_matlabs("pendulum.mat", 0, 90, "Original")
    print("Hi, I am the main module.")
