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
    parameters = {
        "algorithm": "deepcf",
        "prediction_horizon": 1,
    }
    run_experiment(parameters)
    
    