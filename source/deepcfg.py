import sys

# Add the path to the sys.path
sys.path.append("DeePC/source")

import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.linalg import block_diag
import osqp
import DeePC.source.deepc as deepc
import DeePC.source.deepc_tracking as deepc_tracking
import source.deepcf as deepcf

class DeepCgFTracking(deepc_tracking.DEEPC_Tracking):
    def __init__(self, params):
        super().__init__(params)
        self.algorithm = "deepcgf"
        self.set_default_parameters(params, "control_horizon", 2)
        self.deepc = DeepCgF(self.parameters)
        self.deepc.set_opt_criteria(self.parameters.copy())

class DeepCgF(deepcf.DeepC_Fragment):
    def __init__(self, params):
        super().__init__(params)
        self.control_horizon = params["control_horizon"]

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
        self.H = (
            [
                [0.0]
                * ((self.init_length + self.control_horizon) * self.channels)
            ]
            * (self.N)
        )
        for i in range(0, self.N):
            chunk = np.array([])
            for j in range(0, self.channels):
                chunk = np.hstack(
                    (
                        chunk,
                        dataset_in[i][j][0 : self.init_length + self.control_horizon],
                    )
                )
            self.H[i] = chunk
        
        self.H = np.array(self.H)
        self.dataset_formulated = True

    def magic_matrix(self, offset: int) -> np.ndarray:
        """
        Generates a "magic matrix" based on the initial and final lengths and an offset.
        The function creates a negative identity matrix of size determined by the larger of
        `self.finish_length` and `self.init_length`. It then rolls this matrix along the
        specified offset and slices it to produce the final matrix.

        Args:
        offset (int): The offset by which to roll the matrix.

        Returns:
        np.ndarray: The resulting "magic matrix" after rolling and slicing.
        """

        if self.finish_length >= self.init_length:
            test = -np.eye(2 * self.finish_length)
        else:
            test = -np.eye(2 * self.init_length)
        test2 = np.roll(test, self.init_length - offset, axis=0)
        matrix = test2[0 : self.init_length + 1, 0 : self.init_length]
        return matrix

    def show_matrix(self, matrix: np.ndarray) -> None:
        """
        Displays a given matrix with scaled values and saves the visualization as a PDF.

        Args:
            matrix (np.ndarray): The input matrix to be displayed.
        Notes:
            - The function scales the values in the matrix to a range between 0 and 1, except for values that are -1, which remain unchanged.
            - The scaled matrix is displayed using a grayscale colormap.
            - The resulting image is saved as "img/matrix.pdf".
        """

        min_val = matrix.min()
        max_val = matrix.max()

        def scale_value(x):
            if x == -1:
                return x  # Leave it unchanged if it's already -1
            if x == 1:
                return -1
            if x == 0:
                return 1.0
            return (x - min_val) / (max_val - min_val)

        # Apply scaling function to each element in the matrix
        scaled_matrix = np.vectorize(scale_value)(matrix)

        plt.ion()
        fig, ax = plt.subplots(figsize=(7, 4))
        plt.imshow(scaled_matrix, cmap="gray", interpolation="none")
        plt.tight_layout(rect=[0, 0.0, 1, 1])
        plt.savefig("img/matrix.pdf")

    def __bilevel_optimization(self):

        # C1 Matrix

        # Upper and lower bound matrix
        max_input = [1] # Hardcoded SISO
        min_input = [-1] # Hardcoded SISO
        upper = np.array([])
        lower = np.array([])

        n_sub = int(self.finish_length / self.control_horizon)

        # Positive part
        for i in range(n_sub):
            term1 = np.concatenate((self.input_init[0], max_input[0] * np.ones(self.finish_length)))
            upper = np.concatenate((upper, term1[i:self.init_length + i + self.control_horizon]))
            term2 = np.concatenate((self.input_init[0], min_input[0] * np.ones(self.finish_length)))
            lower = np.concatenate((lower, term2[i:self.init_length + i + self.control_horizon]))

            term3 = np.concatenate((self.output_init[0], self.reference[0]))
            term4 = np.array([np.inf] * (self.init_length + self.control_horizon))
            upper = np.concatenate((upper, term4))
            lower = np.concatenate((lower, term3[i:self.init_length + i + self.control_horizon]))

        # Negative part
        for i in range(n_sub):
            term1 = np.concatenate((self.input_init[0], max_input[0] * np.ones(self.finish_length)))
            upper = np.concatenate((upper, term1[i:self.init_length + i + self.control_horizon]))
            term2 = np.concatenate((self.input_init[0], min_input[0] * np.ones(self.finish_length)))
            lower = np.concatenate((lower, term2[i:self.init_length + i + self.control_horizon]))

            term3 = np.concatenate((self.output_init[0], self.reference[0]))
            term4 = np.array([-np.inf] * (self.init_length + self.control_horizon))
            upper = np.concatenate((upper, term3[i:self.init_length + i + self.control_horizon]))
            lower = np.concatenate((lower, term4))

        # Epsilon must be positive
        # but less then
        J_star = 0.0
        upper = np.concatenate((upper, np.array([np.inf] * self.total_length)))
        lower = np.concatenate((lower, np.zeros(self.total_length)))
        upper = np.concatenate((upper, np.array([np.inf])))
        lower = np.concatenate((lower, np.array([0.0])))

        # D1 Matrix
        com = (self.N * self.finish_length) + self.total_length
        D0 = np.zeros((com, com))
        
        # E0 Matrix
        E0 =  np.concatenate((np.zeros(self.N * self.finish_length), np.ones(self.init_length), np.zeros(self.finish_length))) # Hardcoded SISO

        ### B1 Matrix
        B41 = np.array([])

        # Positive part
        for i in range(n_sub):
            term1 = np.zeros((self.init_length+self.control_horizon, self.init_length))
            term2 = np.eye(3*self.finish_length)
            term3 = np.roll(term2, -i*self.control_horizon, axis=0)
            term4 = term3[0:self.init_length+self.control_horizon, 0:self.init_length]
            if B41.size == 0:
                B41 = np.concatenate((term1, term4))
            else:
                B41 = np.concatenate((B41, term1, term4))

        # Negative part
        for i in range(n_sub):
            term1 = np.zeros((self.init_length+self.control_horizon, self.init_length))
            term2 = -np.eye(3*self.finish_length)
            term3 = np.roll(term2, -i*self.control_horizon, axis=0)
            term4 = term3[0:self.init_length+self.control_horizon, 0:self.init_length]
            B41 = np.concatenate((B41, term1, term4))

        B42 = np.array([])

        # Positive part
        for i in range(n_sub):
            term1 = np.zeros((self.init_length+self.control_horizon, self.finish_length))
            term2 = np.eye(3*self.finish_length)
            term3 = np.roll(term2, self.init_length-i*self.control_horizon, axis=0)
            term4 = term3[0:self.init_length+self.control_horizon, 0:self.finish_length]
            if B42.size == 0:
                B42 = np.concatenate((term1, term4))
            else:
                B42 = np.concatenate((B42, term1, term4))

        # Negative part
        for i in range(n_sub):
            term1 = np.zeros((self.init_length+self.control_horizon, self.finish_length))
            term2 = -np.eye(3*self.finish_length)
            term3 = np.roll(term2, self.init_length-i*self.control_horizon, axis=0)
            term4 = term3[0:self.init_length+self.control_horizon, 0:self.finish_length]
            B42 = np.concatenate((B42, term1, term4))

        data_matrix = block_diag(*([self.H.T] * n_sub))
        data_matrix = np.concatenate((data_matrix, data_matrix), axis=0)
        B11 = np.concatenate((data_matrix, B41, B42), axis=1)
        B12 = np.concatenate((np.zeros((self.total_length, self.N * n_sub)), np.eye(self.total_length)), axis=1)
        B13 = np.concatenate((np.zeros((1, self.N * n_sub)), np.ones((1, self.init_length)), np.zeros((1, self.finish_length))), axis=1)
        B1 = np.concatenate((B11, B12, B13), axis=0)

        self.solver = osqp.OSQP()
        D0_sparse = sparse.csc_matrix(D0)
        B1_sparse = sparse.csc_matrix(B1)
        self.solver.setup(D0_sparse, E0, B1_sparse, lower, upper, verbose=False)
        results = self.solver.solve()
        if results.info.status_val != 1:
            return 0.01
        J = results.info.obj_val
        return J if J > 0 else 0.0

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
            raise Exception(
                "Attempt to solve the problem without formulated dataset"
            )
        if not self.init_cond_exists:
            raise Exception(
                "Attempt to solve the problem without initial conditions"
            )
        if not self.reference_exists:
            raise Exception("Attempt to solve the problem without reference")
        if not self.criteria_exists:
            raise Exception("Attempt to solve the problem without criteria")

        # Compute linear bilevel optimization
        #J_star = self.__bilevel_optimization()
        #print(f"J_star: {J_star}")

        # Upper and lower bound matrix
        max_input = [1] # Hardcoded SISO
        min_input = [-1] # Hardcoded SISO
        upper = np.array([])
        lower = np.array([])

        n_sub = int(self.finish_length / self.control_horizon)

        # Positive part
        for i in range(n_sub):
            term1 = np.concatenate((self.input_init[0], max_input[0] * np.ones(self.finish_length)))
            upper = np.concatenate((upper, term1[i:self.init_length + i + self.control_horizon]))
            term2 = np.concatenate((self.input_init[0], min_input[0] * np.ones(self.finish_length)))
            lower = np.concatenate((lower, term2[i:self.init_length + i + self.control_horizon]))


            term3 = np.concatenate((self.output_init[0], self.reference[0]))
            term4 = np.array([np.inf] * (self.init_length + self.control_horizon))
            upper = np.concatenate((upper, term4))
            lower = np.concatenate((lower, term3[i:self.init_length + i + self.control_horizon]))

        # Negative part
        for i in range(n_sub):
            term1 = np.concatenate((self.input_init[0], max_input[0] * np.ones(self.finish_length)))
            upper = np.concatenate((upper, term1[i:self.init_length + i + self.control_horizon]))
            term2 = np.concatenate((self.input_init[0], min_input[0] * np.ones(self.finish_length)))
            lower = np.concatenate((lower, term2[i:self.init_length + i + self.control_horizon]))

            term3 = np.concatenate((self.output_init[0], self.reference[0]))
            term4 = np.array([-np.inf] * (self.init_length + self.control_horizon))
            upper = np.concatenate((upper, term3[i:self.init_length + i + self.control_horizon]))
            lower = np.concatenate((lower, term4))

        # Epsilon must be positive
        # but less then
        upper = np.concatenate((upper, np.array([np.inf] * self.total_length)))
        lower = np.concatenate((lower, np.zeros(self.total_length)))
        upper = np.concatenate((upper, np.array([0.0])))
        lower = np.concatenate((lower, np.array([0.0])))

        # D1 Matrix
        D0 = np.diag([self.criteria["lambda_g"]] * self.N * n_sub + np.zeros(self.total_length).tolist())
        #D0 = np.diag([self.criteria["lambda_g"]] * com)
        
        # E0 Matrix
        E0 =  np.concatenate((np.zeros(self.N * n_sub), np.ones(self.init_length), np.ones(self.finish_length))) # Hardcoded SISO

        ### B1 Matrix
        B41 = np.array([])

        # Positive part
        for i in range(n_sub):
            term1 = np.zeros((self.init_length+self.control_horizon, self.init_length))
            term2 = np.eye(3*self.finish_length)
            term3 = np.roll(term2, -i*self.control_horizon, axis=0)
            term4 = term3[0:self.init_length+self.control_horizon, 0:self.init_length]
            if B41.size == 0:
                B41 = np.concatenate((term1, term4))
            else:
                B41 = np.concatenate((B41, term1, term4))

        # Negative part
        for i in range(n_sub):
            term1 = np.zeros((self.init_length+self.control_horizon, self.init_length))
            term2 = -np.eye(3*self.finish_length)
            term3 = np.roll(term2, -i*self.control_horizon, axis=0)
            term4 = term3[0:self.init_length+self.control_horizon, 0:self.init_length]
            B41 = np.concatenate((B41, term1, term4))

        B42 = np.array([])

        # Positive part
        for i in range(n_sub):
            term1 = np.zeros((self.init_length+self.control_horizon, self.finish_length))
            term2 = np.eye(3*self.finish_length)
            term3 = np.roll(term2, self.init_length-i*self.control_horizon, axis=0)
            term4 = term3[0:self.init_length+self.control_horizon, 0:self.finish_length]
            if B42.size == 0:
                B42 = np.concatenate((term1, term4))
            else:
                B42 = np.concatenate((B42, term1, term4))

        # Negative part
        for i in range(n_sub):
            term1 = np.zeros((self.init_length+self.control_horizon, self.finish_length))
            term2 = -np.eye(3*self.finish_length)
            term3 = np.roll(term2, self.init_length-i*self.control_horizon, axis=0)
            term4 = term3[0:self.init_length+self.control_horizon, 0:self.finish_length]
            B42 = np.concatenate((B42, term1, term4))

        data_matrix = block_diag(*([self.H.T] * n_sub))
        data_matrix = np.concatenate((data_matrix, data_matrix), axis=0)

        B11 = np.concatenate((data_matrix, B41, B42), axis=1)
        B12 = np.concatenate((np.zeros((self.total_length, self.N * n_sub)), np.eye(self.total_length)), axis=1)
        B13 = np.concatenate((np.zeros((1, self.N * n_sub)), np.ones((1, self.init_length)), np.zeros((1, self.finish_length))), axis=1)
        B1 = np.concatenate((B11, B12, B13), axis=0)
        #self.show_matrix(B41)

        self.solver = osqp.OSQP()
        D0_sparse = sparse.csc_matrix(D0)
        B1_sparse = sparse.csc_matrix(B1)
        self.solver.setup(D0_sparse, E0, B1_sparse, lower, upper, verbose=False)#, eps_abs=1e-5)
        results = self.solver.solve()
        #print(f"Epsilon f: {results.x[com-self.total_length: com-self.finish_length]}")
        #print(f"Epsilon y: {results.x[com-self.finish_length: com]}")
        
        if results.info.status_val != 1:
            return None
        return results

    def solve(self):
        """
        Solve the optimization problem and extract the solution.

        Inputs:
        - None

        Outputs:
        - solution (np.ndarray): The optimized solution for the input channels.

        Raises:
        - None
        """
        results = self.solve_raw()
        if results is None:
            return None
        
        n_sub = int(self.finish_length / self.control_horizon)

        data_matrix = block_diag(*([self.H.T] * n_sub))
        raw_result = np.matmul(data_matrix, results.x[0 : self.N * n_sub])
        solution = []
        for i in range(n_sub):
            for j in range(self.control_horizon):
                solution.append(raw_result[(i) * 2 * (self.init_length + self.control_horizon) + self.init_length + j])
        return np.array([solution]) # Hardcoded SISO


