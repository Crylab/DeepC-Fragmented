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


# Check if the script is being run as the main module
if __name__ == "__main__": 
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), width_ratios=[3, 1])
    # Plot the chart
    alg_list = ["deepcf", "deepce"]
    error_list = {}

    for alg in alg_list:
        error_list[alg] = []
        
        for seed in range(0, 2):
            # Define the parameters for the experiment
            params = {
                "Q": [1.0],
                "R": [0.0],
                "lambda_g": 0.5,
                "N": 300,
                "lambda_y": [0.5],
                "algorithm": alg,
                "initial_horizon": 5,
                "prediction_horizon": 10,
                "seed": seed,
                "noise": True,
            }
            obj = Linear_tracking(params)
            result = obj.trajectory_tracking()
            error_list[alg].append(float(obj.error))
            ax[0].plot(result, color='blue' if alg == "deepcf" else 'red')

    ax[0].plot(obj.trajectory.tolist()[params["initial_horizon"]:105], label="Reference", linestyle="--")
    ax[0].set_xlabel("Time (s)")
    ax[0].set_ylabel("Displacement $x_2$, (m)")
    ax[0].set_title("DeePC with linear function: Linear Tracking")
    ax[0].legend()
    ax[0].grid(True)

    # Read from the JSON file
    export_list = ["deepcs"]
    with open("segment.json", "r") as file:
        data = json.load(file)
        error_list["deepcs"] = data["deepcs"]
        alg_list.append("deepcs")


    # Create a candlestick chart in ax[1]
    # Create a list of means and standard deviations
    means = [np.mean(error_list[alg]) for alg in alg_list]
    stds  = [np.std(error_list[alg]) for alg in alg_list]
    mins  = [np.min(error_list[alg]) for alg in alg_list]
    maxs  = [np.max(error_list[alg]) for alg in alg_list]

    color_dict = {"deepcf": "blue", "deepce": "orange", "deepcs": "green"}

    # Create the candlestick plot
    for i, alg in enumerate(alg_list):
        ax[1].plot([i, i], [mins[i], maxs[i]], color='black')
        ax[1].plot([i - 0.1, i + 0.1], [mins[i], mins[i]], color='black')
        ax[1].plot([i - 0.1, i + 0.1], [maxs[i], maxs[i]], color='black')
        ax[1].plot([i - 0.2, i + 0.2], [means[i], means[i]], color='red')
        ax[1].add_patch(plt.Rectangle((i - 0.4, means[i] - stds[i]), 0.8, 2 * stds[i], 
                                    color=color_dict[alg], alpha=0.3))

    # Set the x-axis labels
    ax[1].set_xticks(range(len(alg_list)))
    ax[1].set_xticklabels(alg_list)

    # Set the y-axis label
    ax[1].set_ylabel('Sum of set-point errors')

    # Set the x-axis label
    ax[1].set_xlabel('Prediction horizon')

    # Set the title
    ax[1].set_title('DeePC: Fragmented vs. Segmented')

    patch_list = []
    label_dict = {"deepcf": "Fragmented", "deepce": "Original", "deepcs": "Segmented"}
    for alg in alg_list:
        patch = mpatches.Patch(color=color_dict[alg], label=label_dict[alg], alpha=0.3)
        patch_list.append(patch)
    ax[1].legend(handles=patch_list)

    ax[1].grid(True)


    # Save the plot
    plt.savefig("img/linear.pdf")