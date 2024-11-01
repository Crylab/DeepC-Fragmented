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
    
    # Plot the chart
    error_list = []
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), width_ratios=[3, 1])
    for seed in range(0, 10):
        # Define the parameters for the experiment
        params = {
            "Q": [1.0],
            "R": [0.0],
            "lambda_g": 0.5,
            "lambda_y": [0.5],
            "algorithm": "deepce",
            "prediction_horizon": 10,
            "seed": seed,
            "noise": True,
        }
        obj = Linear_tracking(params)
        result = obj.trajectory_tracking()
        error_list.append(float(obj.error))
        ax[0].plot(result)

    ax[0].plot(obj.trajectory.tolist()[5:105], label="Reference", linestyle="--")
    ax[0].set_xlabel("Time (s)")
    ax[0].set_ylabel("Displacement $x_2$, (m)")
    ax[0].set_title("DeePC with linear function: Linear Tracking")
    ax[0].legend()
    ax[0].grid(True)
    # Save the plot
    plt.savefig("img/linear.pdf")