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
from mpl_toolkits.mplot3d import Axes3D

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
    rss = obj.rss
    return rss
    
# Check if the script is being run as the main module
if __name__ == "__main__":  
    # Define the parameters for the experiment
    range_g = np.logspace(-2, 2, num=2)
    range_y = np.logspace(-2, 2, num=2)
    tasks = []
    for lambda_g in range_g:
        for lambda_y in range_y:
            parameters = {
                "algorithm": "deepcf",
                "prediction_horizon": 8,
                "Q": [1, 1, 1, 100],
                "R": [0.1, 0.1],
                "N": 50,
                "lambda_g": lambda_g,
                "lambda_y": [lambda_y] * 4,
                "Lissajous_circle_time": 3700,
                #"print_out": "Nothing",
            }
            tasks.append(parameters.copy())
            
    # Run the experiment in parallel
    pool = multiprocessing.Pool(processes=90)
    res = pool.map(run_experiment, tasks)
    pool.close()
    pool.join()
    print(res)
    
    # Create a meshgrid for x and y
    X, Y = np.meshgrid(range_g, range_y)

    # Reshape z to match the 10x10 grid
    Z = np.array(res).reshape(2, 2)

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    ax.plot_surface(X, Y, Z, cmap='viridis')

    # Add labels
    ax.set_xlabel('lambda_g')
    ax.set_ylabel('lambda_y')
    ax.set_zlabel('RSS')

    # Show the plot
    plt.show()
    
    
    
    
    