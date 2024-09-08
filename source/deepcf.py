import sys
# Add the path to the sys.path
sys.path.append("DeePC/source")

import numpy as np
from scipy import sparse
import osqp
import DeePC.source.deepc as deepc
import DeePC.source.deepc_tracking as deepc_tracking

class DeepC_Fragment(deepc.DeepC):
    def dataset_reformulation(self, dataset_in):
        """
        Reformulate the dataset for optimization.
        
        Inputs:
        - dataset_in (np.ndarray): Input dataset to be reformulated.
        
        Outputs:
        - None
        
        Raises:
        - Exception: If dataset does not exist.
        """
        if not self.dataset_exists:
            raise Exception("Attempt to reformulate H without dataset")
        self.H = [[0.0] * ((self.init_length + 1) * self.channels * self.finish_length)] * self.N * self.finish_length
        for ite in range(0, self.finish_length):
            for i in range(0, self.N):
                chunk = np.zeros(((self.init_length + 1) * self.channels * ite))
                for j in range(0, self.channels):
                    chunk = np.hstack((chunk, dataset_in[i][j][ite : self.init_length+ite+1]))
                self.H[ite*self.N+i] = np.concatenate((chunk, np.zeros((self.init_length+1) * self.channels * (self.finish_length-ite-1))))
        self.H = np.array(self.H)
        self.dataset_formulated = True
    
    def solve_raw(self):
        """
        Solve the optimization problem and provide raw OSQP output.
        
        Inputs:
        - None
        
        Outputs:
        - results: The results from the OSQP solver.
        
        Raises:
        - Exception: If any required data or criteria is missing.
        """
        if not self.dataset_formulated:
            raise Exception("Attempt to solve the problem without formulated dataset")
        if not self.init_cond_exists:
            raise Exception("Attempt to solve the problem without initial conditions")
        if not self.reference_exists:
            raise Exception("Attempt to solve the problem without reference")
        if not self.criteria_exists:
            raise Exception("Attempt to solve the problem without criteria")

        #
        #
        #
        #C1 = zeros(1, finish_length*heg);
        #for o=0:finish_length-1
        #    chunk = [u_init, zeros(size(u_init))];
        #    chunk = circshift(chunk, -o);
        #    chunk = [chunk(1:init_length), 0];

        #    chunk2 = [y_init, zeros(size(y_init))];
        #    chunk2 = circshift(y_init, -o);
        #    chunk2 = [y_init(1:init_length), 0];

        #C1(o*2*(init_length+1)+1:(o+1)*2*(init_length+1)) = [chunk, chunk2];

        #end
        
        heg = self.channels * (self.init_length + 1)
        
        C1 = np.array([])
        
        for o in range(0, self.finish_length):
            for i in range(0, self.n_inputs):
                temp1 = np.concatenate((self.input_init[i], np.zeros(self.init_length)))
                temp2 = np.roll(temp1, -o)[0:self.init_length]
                C1 = np.concatenate((C1, temp2, np.array([0.0])))
            for i in range(0, self.n_outputs):
                temp1 = np.concatenate((self.output_init[i], np.zeros(self.init_length)))
                temp2 = np.roll(temp1, -o)[0:self.init_length]
                C1 = np.concatenate((C1, temp2, np.array([0.0])))
        
        print(C1)
        
        raise NotImplementedError("Method not implemented")
        
class local_deepc_tracking(deepc_tracking.DEEPC_Tracking):
    
    def hi(self):
        print("Hello from DeepC_tracking local")