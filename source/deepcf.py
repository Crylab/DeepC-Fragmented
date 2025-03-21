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


class DEEPCF_Tracking(deepc_tracking.DEEPC_Tracking):
    def __init__(self, params):
        super().__init__(params)
        self.algorithm = "deepcf"
        self.deepc = DeepC_Fragment(self.parameters)
        self.deepc.set_opt_criteria(self.parameters.copy())
        
        
class DEEPCS_Tracking(deepc_tracking.DEEPC_Tracking):
    def __init__(self, params):
        super().__init__(params)
        self.algorithm = "deepcs"
        self.deepc = DeepC_Segment(self.parameters)
        self.deepc.set_opt_criteria(self.parameters.copy())


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
        self.H = (
            [
                [0.0]
                * ((self.init_length + 1) * self.channels * self.finish_length)
            ]
            * self.N
            * self.finish_length
        )
        for ite in range(0, self.finish_length):
            for i in range(0, self.N):
                chunk = np.zeros(((self.init_length + 1) * self.channels * ite))
                for j in range(0, self.channels):
                    chunk = np.hstack(
                        (
                            chunk,
                            dataset_in[i][j][ite : self.init_length + ite + 1],
                        )
                    )
                self.H[ite * self.N + i] = np.concatenate(
                    (
                        chunk,
                        np.zeros(
                            (self.init_length + 1)
                            * self.channels
                            * (self.finish_length - ite - 1)
                        ),
                    )
                )
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
        matrix = test2[0 : self.init_length + 1, 0 : self.finish_length]
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

        # C1 Matrix

        C1 = np.array([])

        for o in range(0, self.finish_length):
            for i in range(0, self.n_inputs):
                temp1 = np.concatenate(
                    (self.input_init[i], np.zeros(self.finish_length))
                )
                temp2 = np.roll(temp1, -o)[0 : self.init_length]
                C1 = np.concatenate((C1, temp2, np.array([0.0])))
            for i in range(0, self.n_outputs):
                temp1 = np.concatenate(
                    (self.output_init[i], np.zeros(self.finish_length))
                )
                temp2 = np.roll(temp1, -o)[0 : self.init_length]
                C1 = np.concatenate((C1, temp2, np.array([0.0])))

        # D1 Matrix

        com = (
            self.N * self.finish_length
            + (self.channels * self.finish_length)
            + (self.n_outputs * self.init_length)
        )
        D0 = self.criteria["lambda_g"] * np.identity(com)
        for input_iter in range(0, self.n_inputs):
            counter = -1
            for iterator in range(
                self.N * self.finish_length + (input_iter * self.finish_length),
                self.N * self.finish_length
                + ((input_iter + 1) * self.finish_length),
            ):
                counter += 1
                D0[iterator][iterator] = self.criteria["R"][input_iter] * (
                    self.criteria["beta_R"][input_iter] ** counter
                )
        for output_iter in range(0, self.n_outputs):
            counter = -1
            for iterator in range(
                self.N * self.finish_length
                + ((input_iter + output_iter + 1) * self.finish_length),
                self.N * self.finish_length
                + ((input_iter + output_iter + 2) * self.finish_length),
            ):
                counter += 1
                D0[iterator][iterator] = self.criteria["Q"][output_iter] * (
                    self.criteria["beta_Q"][output_iter] ** counter
                )
        for output_iter in range(0, self.n_outputs):
            counter = -1
            pre = self.N * self.finish_length + (
                self.channels * self.finish_length
            )
            for iterator in range(
                pre + (output_iter * self.init_length),
                pre + ((output_iter + 1) * self.init_length),
            ):
                counter += 1
                D0[iterator][iterator] = self.criteria["lambda_y"][
                    output_iter
                ] * (self.criteria["beta_lambda_y"][output_iter] ** counter)

        # E0 Matrix
        E1 = np.copy(self.reference)
        for i in range(0, self.n_outputs):
            E1[i] *= -self.criteria["Q"][i]
        E2 = np.concatenate(E1)
        F1 = np.zeros(
            self.N * self.finish_length + (self.n_inputs * self.finish_length)
        )
        F2 = np.zeros(self.n_outputs * self.init_length)
        E0 = np.concatenate((F1, E2, F2))

        # B1 Matrix
        B31 = np.array([])
        for j in range(0, self.finish_length):
            mat_in = block_diag(*([self.magic_matrix(j)] * self.n_inputs))
            mat_out = block_diag(*([self.magic_matrix(j)] * self.n_outputs))
            mat = block_diag(mat_in, mat_out)
            if B31.size == 0:
                B31 = mat
            else:
                B31 = np.concatenate((B31, mat))

        B32 = np.array([])
        for j in range(0, self.finish_length):
            mat_in = np.zeros(
                (
                    self.n_inputs * (self.init_length + 1),
                    (self.init_length) * self.n_outputs,
                )
            )
            if j < self.init_length:
                temp1 = np.eye(self.init_length)
                temp2 = np.zeros((1, self.init_length))
                temp3 = np.concatenate((temp1, temp2))
                mat_out = block_diag(*([temp3] * self.n_outputs))
            else:
                mat_out = np.zeros(
                    (
                        self.n_outputs * (self.init_length + 1),
                        (self.init_length) * self.n_outputs,
                    )
                )
            mat = np.concatenate((mat_in, mat_out))
            if B32.size == 0:
                B32 = mat
            else:
                B32 = np.concatenate((B32, mat))

        B1 = np.concatenate((self.H.T, B31, B32), axis=1)
        self.solver = osqp.OSQP()
        D0_sparse = sparse.csc_matrix(D0)
        B1_sparse = sparse.csc_matrix(B1)
        self.solver.setup(D0_sparse, E0, B1_sparse, C1, C1, verbose=False)
        results = self.solver.solve()
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
        solution = np.array([[0.0] * self.finish_length] * self.n_inputs)
        for input_iter in range(0, self.n_inputs):
            solution[input_iter] = results.x[
                self.N * self.finish_length
                + (input_iter * self.finish_length) : self.N
                * self.finish_length
                + ((input_iter + 1) * self.finish_length)
            ]
        return solution


class DeepC_Segment(DeepC_Fragment):
    
    def sigma_matrix(self, offset: int) -> np.ndarray:
        com = self.finish_length+self.init_length-1
        test = -np.eye(2*com)
        test2 = np.roll(test, -offset, axis=0)
        test3 = test2[0:self.init_length, 0:com]
        matrix = np.concatenate((test3, np.zeros((1, com))), axis=0)
        return matrix
    
    
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
        
        heg = self.channels * (self.init_length + 1)
        
        # C1 Matrix
        
        C1 = np.array([])
        
        for o in range(0, self.finish_length):
            for i in range(0, self.n_inputs):
                temp1 = np.concatenate((self.input_init[i], np.zeros(self.finish_length)))
                temp2 = np.roll(temp1, -o)[0:self.init_length]
                C1 = np.concatenate((C1, temp2, np.array([0.0])))
            for i in range(0, self.n_outputs):
                temp1 = np.concatenate((self.output_init[i], np.zeros(self.finish_length)))
                temp2 = np.roll(temp1, -o)[0:self.init_length]
                C1 = np.concatenate((C1, temp2, np.array([0.0])))
        
        
        # D1 Matrix
        
        com = self.N * self.finish_length + (self.channels * self.finish_length) + (self.n_outputs * (self.init_length + self.finish_length - 1))
        D0 = self.criteria["lambda_g"] * np.identity(com)
        for input_iter in range(0, self.n_inputs):
            counter = -1
            for iterator in range(
                self.N * self.finish_length + (input_iter * self.finish_length),
                self.N * self.finish_length + ((input_iter + 1) * self.finish_length),
            ):
                counter += 1
                D0[iterator][iterator] = self.criteria["R"][input_iter] * (
                    self.criteria["beta_R"][input_iter] ** counter
                )
        for output_iter in range(0, self.n_outputs):
            counter = -1
            for iterator in range(
                self.N * self.finish_length + ((input_iter + output_iter + 1) * self.finish_length),
                self.N * self.finish_length + ((input_iter + output_iter + 2) * self.finish_length),
            ):
                counter += 1
                D0[iterator][iterator] = self.criteria["Q"][output_iter] * (
                    self.criteria["beta_Q"][output_iter] ** counter
                )
        for output_iter in range(0, self.n_outputs):
            counter = -1
            pre = self.N * self.finish_length + (self.channels * self.finish_length)
            for iterator in range(
                pre + (output_iter * (self.init_length + self.finish_length - 1)),
                pre + ((output_iter + 1) * (self.init_length + self.finish_length - 1)),
            ):
                counter += 1
                D0[iterator][iterator] = self.criteria["lambda_y"][output_iter] * (
                    self.criteria["beta_lambda_y"][output_iter] ** counter
                )
        
        # E0 Matrix
        E1 = np.copy(self.reference)
        for i in range(0, self.n_outputs):
            E1[i] *= -self.criteria["Q"][i]
        E2 = np.concatenate(E1)
        F1 = np.zeros(self.N * self.finish_length + (self.n_inputs * self.finish_length))
        F2 = np.zeros(self.n_outputs * (self.init_length + self.finish_length - 1))
        E0 = np.concatenate((F1, E2, F2))
        
        # B1 Matrix
        B31 = np.array([])
        for j in range(0, self.finish_length):
            mat_in = block_diag(*([self.magic_matrix(j)] * self.n_inputs))
            mat_out = block_diag(*([self.magic_matrix(j)] * self.n_outputs))
            mat = block_diag(mat_in, mat_out)
            if B31.size == 0:
                B31 = mat
            else:
                B31 = np.concatenate((B31, mat))
                
        B32 = np.array([])
        for j in range(0, self.finish_length):
            mat_in = np.zeros((
                self.n_inputs*(self.init_length+1), 
                self.init_length*self.finish_length*self.n_outputs
            ))
            mat_out = block_diag(*([self.sigma_matrix(j)] * self.n_outputs))
            mat = np.concatenate((mat_in, mat_out))
            if B32.size == 0:
                B32 = mat
            else:
                B32 = np.concatenate((B32, mat))
        
        B1 = np.concatenate((self.H.T, B31, B32), axis=1)
        #self.show_matrix(B1)
        self.solver = osqp.OSQP()
        D0_sparse = sparse.csc_matrix(D0)
        B1_sparse = sparse.csc_matrix(B1)
        self.solver.setup(D0_sparse, E0, B1_sparse, C1, C1, verbose=False)
        results = self.solver.solve()
        if results.info.status_val != 1:
            return None
        return results
