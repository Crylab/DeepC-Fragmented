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

def hyperparameter_tuning():
    n_seeds = 10
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))

    control_list = [1, 2, 3, 5, 10]
    plot_list = []
    shadow_list = []
    lambda_list = [3, 5, 7, 8, 9, 12, 15]
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
    # Plot the chart
    prediction_list = [10, 20, 40]
    error_dict = {}
    alg = "deepcgf"

    for prediction_horizon in prediction_list:
        error_list = []
        
        for seed in range(0, 10):
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
            error_list.append(float(obj.error))
            if prediction_horizon == 10:
                if seed == 0:
                    ax[0].plot(result, color='blue', alpha=0.5, linewidth=0.5, label="Fragmented")
                else:
                    ax[0].plot(result, color='blue', alpha=0.5, linewidth=0.5)

        error_dict[str(prediction_horizon)] = {
            "mean": np.mean(error_list),
            "std": np.std(error_list),
            "min": np.min(error_list),
            "max": np.max(error_list),
        }

    ax[0].plot(obj.trajectory.tolist()[params["initial_horizon"]:105], label="Reference", linestyle="--", color="black")
    ax[0].set_xlabel("Time (s)")
    ax[0].set_ylabel("Displacement $x_2$, (m)")
    ax[0].set_title("DeePC with linear function: Linear Tracking")
    ax[0].legend()
    ax[0].grid(True)

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

    # Set the x-axis labels
    x_len = len(prediction_list) * len(alg_list)
    ax[1].set_xticks(range(x_len))
    x_label = ["", "10", "", "", "20", "", "", "40", ""]
    print(x_label[0:x_len])
    ax[1].set_xticklabels(x_label[0:x_len])
    # Set the y-axis label
    ax[1].set_ylabel('Sum of set-point errors')
    # Set the x-axis label
    ax[1].set_xlabel('Prediction horizon')
    # Set the title
    ax[1].set_title('Tracking errors')
    patch_list = []
    for alg in alg_list:
        patch = mpatches.Patch(color=color_dict[alg], label=alg, alpha=0.3)
        patch_list.append(patch)
    ax[1].legend(handles=patch_list, loc=2)
    ax[1].grid(True)
    # Save the plot
    plt.savefig("img/linear.pdf")

def 

# Check if the script is being run as the main module
if __name__ == "__main__": 
    #varyingN()
    #hyperparameter_tuning()
    


    print("Hi, I am the main module.")
