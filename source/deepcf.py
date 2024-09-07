import numpy as np
from scipy import sparse
import osqp
import DeePC.source.deepc as deepc

class DeepC_Fragment(deepc.DeepC):
    def __init__(self, params: dict):
        print("Hello from DeepC_Fragment")