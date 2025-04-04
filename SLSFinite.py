import numpy as np
import cvxpy as cp
from utils import *
from scipy.linalg import block_diag
import time

def low_block_tri_variable(n, m, Tp1):
    var = (n*Tp1)*[None]
    for t in range(Tp1):
        for i in range(n):
            add_var = cp.Variable((1, m*(t+1)))
            if t == 0 and i == 0:
                var[0] = cp.hstack([add_var, np.zeros((1, m*(Tp1-(t+1))))])
            elif t == Tp1-1:
                var[t*n+i] = add_var
            else:
                var[t*n+i] = cp.hstack([add_var, np.zeros((1, m*(Tp1-(t+1))))])
    var = cp.vstack(var)
    assert var.shape == (n*Tp1, m*Tp1)
    return var

def row_sparse_low_block_tri_variable(n, m, Tp1, rem_row):
    var = (n*Tp1)*[None]
    for t in range(Tp1):
        for i in range(n):
            if t*n + i in rem_row:
                add_var = np.zeros((1, m*(t+1)))
            else:
                add_var = cp.Variable((1, m*(t+1)))
            if t == 0 and i == 0:
                var[0] = cp.hstack([add_var, np.zeros((1, m*(Tp1-(t+1))))])
            elif t == Tp1-1:
                var[t*n+i] = add_var
            else:
                var[t*n+i] = cp.hstack([add_var, np.zeros((1, m*(Tp1-(t+1))))])
    var = cp.vstack(var)
    assert var.shape == (n*Tp1, m*Tp1)
    return var

def col_sparse_low_block_tri_variable(n, m, Tp1, rem_col):
    var = (m*Tp1)*[None]
    for t in range(Tp1):
        for i in range(m):
            if t*m + i in rem_col:
                add_var = np.zeros((1, n*(Tp1-t)))
            else:
                add_var = cp.Variable((1, n*(Tp1-t)))
            if t == 0 and i == 0:
                var[0] = add_var
            elif t == 0:
                var[t*m+i] = add_var
            else:
                var[t*m+i] = cp.hstack([np.zeros((1, n*t)), add_var])
    var = cp.vstack(var)
    assert var.T.shape == (n*Tp1, m*Tp1)
    return var.T

class SLSFinite():
    def __init__(self, A_list, B_list, C_list, delay, norm=None):
        """
        Store the variables used for convex optimization in finite time system level synthesis framework.
    
        Parameters
        ----------
        A_list: list of matrices [A_0, ...A_T]
        B_list: list of matrices [B_0, ...B_T]
        C_list: list of matrices [C_0, ...C_T]
            where A_t, B_t, C_t are the matrices in the dynamics of the system at time t.
        
        Attributes
        ----------
        Phi_xx: cvxpy.Variable, shape ((T+1)*nx, (T+1)*nx)
        Phi_xy: cvxpy.Variable, shape ((T+1)*nx, (T+1)*ny)
        Phi_ux: cvxpy.Variable, shape ((T+1)*nu, (T+1)*nx)
        Phi_uy: cvxpy.Variable, shape ((T+1)*nu, (T+1)*ny)
        """
    #### basic SLS objects ####################################################################
        assert len(A_list) == len(B_list) == len(C_list)
        # init basic contents
        self.A_list = A_list
        self.B_list = B_list
        self.C_list = C_list

        self.T = len(A_list) - 1
        self.delay = delay
        self.nx = A_list[0].shape[0]
        self.nu = B_list[0].shape[1]
        self.ny = C_list[0].shape[0]

        self.Phi_xx = low_block_tri_variable(self.nx, self.nx, self.T+1)
        if norm == None:
            self.Phi_uy = low_block_tri_variable(self.nu, self.ny, self.T+1)
            self.Phi_ux = low_block_tri_variable(self.nu, self.nx, self.T+1)
            self.Phi_xy = low_block_tri_variable(self.nx, self.ny, self.T+1)
        else:
            if norm[0] == 'Reweighted Actuator Norm':
                self.Phi_uy = row_sparse_low_block_tri_variable(self.nu, self.ny, self.T+1, norm[1])
                self.Phi_ux = row_sparse_low_block_tri_variable(self.nu, self.nx, self.T+1, norm[1])
                self.Phi_xy = low_block_tri_variable(self.nx, self.ny, self.T+1)
            elif norm[0] == 'Reweighted Sensor Norm':
                self.Phi_uy = col_sparse_low_block_tri_variable(self.nu, self.ny, self.T+1, norm[1])
                self.Phi_ux = low_block_tri_variable(self.nu, self.nx, self.T+1)
                self.Phi_xy = col_sparse_low_block_tri_variable(self.nx, self.ny, self.T+1, norm[1])
            else:
                raise Exception("Reweighted Actuator or Sensor.")
            
        self.Phi_matrix = cp.bmat( [[self.Phi_xx,   self.Phi_xy], 
                                    [self.Phi_ux,   self.Phi_uy]] )
        # define downshift operator
        self.Z = np.block([ [np.zeros([self.nx,self.T*self.nx]),    np.zeros([self.nx,self.nx])        ],
                            [np.eye(self.T*self.nx),                np.zeros([self.T*self.nx, self.nx])]
                            ])
        # define block-diagonal matrices
        self.cal_A = block_diag(*A_list)
        self.cal_B = block_diag(*B_list)
        self.cal_C = block_diag(*C_list)

        assert self.Z.shape == self.cal_A.shape
        assert self.Z.shape[0] == self.cal_B.shape[0]
        assert self.Z.shape[0] == self.cal_C.shape[1]

        # dependent variables
        self.F = None
        self.Phi_yx = None
        self.Phi_yy = None
        self.E = None
        self.D = None
        self.F_trunc = None
        self.F_causal_rows_basis = None
        self.Phi_trunc = None
        self.Phi_uy_trunc = None
        self.causal_time = None

    def SLP_constraints(self, constant_matrix=None):
        """
        Compute the system level parametrization constraints used in finite time system level synthesis.

        Return
        ------
        SLP: list of 6 cvxpy.Constraint objects
            These are constraints on the Phi variables consisting of
            1. 2 affine system level parametrization constraints and
            2. 4 lower block triangular constraints.
        """
        Tp1 = self.T + 1
        I = np.eye(Tp1*self.nx)
        SLP = [cp.bmat([[I - self.Z @ self.cal_A, -self.Z @ self.cal_B]]) @ self.Phi_matrix == cp.bmat([[I, np.zeros( (Tp1*self.nx, Tp1*self.ny) )]]),
                self.Phi_matrix @ cp.bmat([[I - self.Z @ self.cal_A], [-self.cal_C]]) == cp.bmat([[I], [np.zeros( (Tp1*self.nu, Tp1*self.nx) )]])]
        return SLP


    #### basic SLS operations ####################################################################

    def calculate_dependent_variables(self, key):
        """
        Compute the controller F
        """
        F_test = self.Phi_uy.value - self.Phi_ux.value @ np.linalg.inv((self.Phi_xx.value).astype('float64')) @ self.Phi_xy.value
        if key=="Reweighted Nuclear Norm" or key=="Reweighted Sensor Norm":
            self.F = np.linalg.inv( (np.eye(self.nu*(self.T+1)) + self.Phi_ux.value @ self.Z @ self.cal_B).astype('float64') ) @ self.Phi_uy.value
        elif key=="Reweighted Actuator Norm":
            self.F = self.Phi_uy.value @ np.linalg.inv( (np.eye(self.ny*(self.T+1)) + self.cal_C @ self.Phi_xy.value).astype('float64') )
        print(key + ':', np.max(np.abs(F_test - self.F)))
        # assert np.all(np.isclose( self.F.astype('float64'), F_test.astype('float64')) )
        assert np.all(np.isclose(self.F, F_test, rtol=1e-5, atol=1e-5))
        filter = np.kron( np.tril(np.ones([self.T+1,self.T+1])) , np.ones([self.nu, self.ny]) )
        self.F = filter*self.F
        return
    
    def F_trunc_to_Phi_trunc(self):
        Phi_xx = np.linalg.inv( (np.eye(self.nx*(self.T+1)) - self.Z @ self.cal_A - self.Z @ self.cal_B @ self.F_trunc @ self.cal_C).astype('float64') )
        Phi_xy = Phi_xx.dot(self.Z).dot(self.cal_B).dot(self.F_trunc)
        Phi_ux = self.F_trunc.dot(self.cal_C).dot(Phi_xx)
        Phi_uy = ( np.eye(self.nu*(self.T+1)) + Phi_ux.dot(self.Z).dot(self.cal_B)).dot(self.F_trunc)
        Phi_uy_sum = self.F_trunc + self.F_trunc.dot(self.cal_C).dot(Phi_xx).dot(self.Z).dot(self.cal_B).dot(self.F_trunc)
        assert np.all(np.isclose( Phi_uy.astype('float64'), Phi_uy_sum.astype('float64')) )
        self.Phi_trunc = np.bmat([[Phi_xx, Phi_xy], [Phi_ux, Phi_uy]])
        return
    
    def causal_factorization(self, rank_eps):
        start = time.time()
        low_btri = self.F
        assert low_btri.shape[0] == self.nu*(self.T+1)
        assert low_btri.shape[1] == self.ny*(self.T+1)
        E = np.array([]).reshape((0,self.F.shape[1]))
        D = np.array([]).reshape((0,0)) # trick
        rank_counter = 0
        rank_low_btri = np.linalg.matrix_rank(low_btri, tol=rank_eps)
        added_rows = rank_low_btri*[None]
        for t in range(self.T+1):
            for s in range(self.nu):
                row = t*self.nu + s
                submat_new = low_btri[0:row+1, :] # rows up to row (note: last step row = (T+1)*nu)
                rank_new = np.linalg.matrix_rank(submat_new, tol=rank_eps)
                if rank_new - rank_counter == 1:
                    added_rows[rank_counter] = row
                    rank_counter += 1
                    # add vector to E matrix
                    E = np.vstack([E, submat_new[row:row+1, :]])
                    # modify D matrix
                    unit = np.zeros([1, rank_counter]) 
                    unit[0, -1] = 1. # add a 1 at the last column of unit
                    D = np.hstack([D, np.zeros([row, 1])]) # D is (row, rank_counter)
                    D = np.vstack([D, unit]) # D is (row+1, rank_counter)
                    assert E.shape == (rank_counter, low_btri.shape[1])
                    assert D.shape == (row+1, rank_counter)

                elif rank_new == rank_counter:
                    # solve linear system
                    c = np.linalg.lstsq( E.T , low_btri[row, :])[0]
                    c = c.reshape(([1, rank_counter]))
                    D = np.vstack([D, c])
                    assert E.shape == (rank_counter, low_btri.shape[1])
                    assert D.shape == (row+1, rank_counter)
                else:
                    raise Exception('Rank increased more than 1.')
                
        assert E.shape == (rank_counter, low_btri.shape[1])
        assert D.shape == (low_btri.shape[0], rank_counter)
        assert rank_counter == rank_low_btri
        assert len(added_rows) == rank_low_btri
        # set attributees and compute truncated F
        self.E = E
        self.D = D
        self.F_trunc = D.dot(E)
        self.rank_F_trunc = rank_low_btri
        self.F_causal_row_basis = added_rows
        self.causal_time = time.time() - start
        return





    #### operations in Phi space ####################################################################

    def extract_offdiag_expr(self, direction='21'):
        Tplus1 = self.T + 1
        block_row = 4
        block_col = 4

        if self.Phi_uy.shape != (4*Tplus1, 4*Tplus1):
            raise ValueError(f"Phi_uy has shape {self.Phi_uy.shape}, but expected {(4*Tplus1, 4*Tplus1)}.")

        sub_blocks = []
        for t in range(Tplus1):
            row_list = []
            for tau in range(Tplus1):
                row_start = t * block_row
                col_start = tau * block_col
                sub_4x4 = self.Phi_uy[row_start : row_start+block_row,
                                    col_start : col_start+block_col]
                if direction == '21':
                    sub_2x2 = sub_4x4[0:2, 2:4]
                else:
                    sub_2x2 = sub_4x4[2:4, 0:2]

                row_list.append(sub_2x2)
            sub_blocks.append(row_list)
        big_expr = cp.bmat(sub_blocks)  
        return big_expr


    def extract_Phi_subcom_mat(self, direction='21'):
        Tplus1 = self.T + 1
        block_row = self.nu
        block_col = self.ny

        big_mat_phiuy = np.zeros((2*Tplus1, 2*Tplus1))
        for t in range(Tplus1):
            for tau in range(Tplus1):
                row_start = t * block_row
                col_start = tau * block_col
                sub_4x4 = self.Phi_uy.value[row_start : row_start+block_row,
                                col_start : col_start+block_col]
                
                if direction == '21':
                    sub_2x2 = sub_4x4[0:2, 2:4]
                else:
                    sub_2x2 = sub_4x4[2:4, 0:2]

                big_mat_phiuy[t*2 : (t+1)*2,  tau*2 : (tau+1)*2] = sub_2x2

        block_row = self.nu
        block_col = self.nx
        big_mat_phiux = np.zeros((2*Tplus1, 4*Tplus1))
        for t in range(Tplus1):
            for tau in range(Tplus1):
                row_start = t * block_row
                col_start = tau * block_col
                sub_4x8 = self.Phi_ux.value[row_start : row_start+block_row,
                                col_start : col_start+block_col]
                
                if direction == '21':
                    sub_2x4 = sub_4x8[0:2, 4:8]
                else:
                    sub_2x4 = sub_4x8[2:4, 0:4]

                big_mat_phiux[t*2 : (t+1)*2,  tau*4 : (tau+1)*4] = sub_2x4

        block_row = self.nx
        block_col = self.ny
        big_mat_phixy = np.zeros((4*Tplus1, 2*Tplus1))
        for t in range(Tplus1):
            for tau in range(Tplus1):
                row_start = t * block_row
                col_start = tau * block_col
                sub_8x4 = self.Phi_xy.value[row_start : row_start+block_row,
                                col_start : col_start+block_col]
                
                if direction == '21':
                    sub_4x2 = sub_8x4[0:4, 2:4]
                else:
                    sub_4x2 = sub_8x4[4:8, 0:2]

                big_mat_phixy[t*4 : (t+1)*4,  tau*2 : (tau+1)*2] = sub_4x2
        return big_mat_phiuy, big_mat_phiux, big_mat_phixy





    #### recording and print result ####################################################################

    def extract_sub_communication_matrix(self, direction='21'):
        Tplus1 = self.T + 1
        block_row = 4
        block_col = 4

        big_mat = np.zeros((2*Tplus1, 2*Tplus1))

        for t in range(Tplus1):
            for tau in range(Tplus1):
                row_start = t * block_row
                col_start = tau * block_col
                sub_4x4 = self.F[row_start : row_start+block_row,
                                col_start : col_start+block_col]
                
                if direction == '21':
                    sub_2x2 = sub_4x4[0:2, 2:4]
                else:
                    sub_2x2 = sub_4x4[2:4, 0:2]

                big_mat[t*2 : (t+1)*2,  tau*2 : (tau+1)*2] = sub_2x2
        return big_mat


    def compute_communication_messages(self, rank_eps=1e-7):
        L21 = self.extract_sub_communication_matrix(direction='21')
        L12 = self.extract_sub_communication_matrix(direction='12')

        rank_21 = np.linalg.matrix_rank(L21, tol=rank_eps)
        rank_12 = np.linalg.matrix_rank(L12, tol=rank_eps)

        return (rank_21, rank_12)




    def compute_offdiag_rank_of_Phi(self, rank_eps=1e-7):
        L1_mat = self.extract_offdiag_expr(direction='21')
        if hasattr(L1_mat, 'value'):
            L1_val = L1_mat.value
            if L1_val is None:
                raise ValueError("extract_offdiag_expr('21') => Expression has no .value? Check if solver ran.")
        else:
            L1_val = L1_mat
        rank1 = np.linalg.matrix_rank(L1_val, tol=rank_eps)

        L2_mat = self.extract_offdiag_expr(direction='12')
        if hasattr(L2_mat, 'value'):
            L2_val = L2_mat.value
            if L2_val is None:
                raise ValueError("extract_offdiag_expr('12') => Expression has no .value? Check if solver ran.")
        else:
            L2_val = L2_mat
        rank2 = np.linalg.matrix_rank(L2_val, tol=rank_eps)

        return rank1, rank2




    def display_message_row(self, rank_eps=1e-7, save_file="intermediate_results.npz"):
        L21 = self.extract_sub_communication_matrix(direction='21')
        L12 = self.extract_sub_communication_matrix(direction='12')

        print("Message time in L21:")
        L21_D, L21_E, L21_times = row_factorization_causal(L21, rank_eps)
        print(f"  Times (based on rank increase): {L21_times}\n")

        print("Message time in L12:")
        L12_D, L12_E, L12_times = row_factorization_causal(L12, rank_eps)
        print(f"  Times (based on rank increase): {L12_times}\n")

        L21_Phiuy, L21_Phiux, L21_Phixy = self.extract_Phi_subcom_mat(direction='21')
        L12_Phiuy, L12_Phiux, L12_Phixy = self.extract_Phi_subcom_mat(direction='12')

        print("Message time in L21_Phiuy:")
        L21_PhiuyD, L21_PhiuyE, L21_Phiuy_times = row_factorization_causal(L21_Phiuy, rank_eps)
        print(f"  Times (based on rank increase): {L21_Phiuy_times}\n")

        print("Message time in L21_Phiux:")
        L21_PhiuxD, L21_PhiuxE, L21_Phiux_times = row_factorization_causal(L21_Phiux, rank_eps)
        print(f"  Times (based on rank increase): {L21_Phiux_times}\n")

        print("Message time in L21_Phixy:")
        L21_PhixyD, L21_PhixyE, L21_Phixy_times = row_factorization_causal(L21_Phixy, rank_eps)
        print(f"  Times (based on rank increase): {L21_Phixy_times}\n")

        print("Message time in L12_Phiuy:")
        L12_PhiuyD, L12_PhiuyE, L12_Phiuy_times = row_factorization_causal(L12_Phiuy, rank_eps)
        print(f"  Times (based on rank increase): {L12_Phiuy_times}\n")

        print("Message time in L12_Phiux:")
        L12_PhiuxD, L12_PhiuxE, L12_Phiux_times = row_factorization_causal(L12_Phiux, rank_eps)
        print(f"  Times (based on rank increase): {L12_Phiux_times}\n")

        print("Message time in L12_Phixy:")
        L12_PhixyD, L12_PhixyE, L12_Phixy_times = row_factorization_causal(L12_Phixy, rank_eps)
        print(f"  Times (based on rank increase): {L12_Phixy_times}\n")

        save_dict = {
            "L21": L21, "L21_D": L21_D, "L21_E": L21_E, "L21_times": L21_times,
            "L12": L12, "L12_D": L12_D, "L12_E": L12_E, "L12_times": L12_times,

            "L21_Phiuy": L21_Phiuy, "L21_PhiuyD": L21_PhiuyD, "L21_PhiuyE": L21_PhiuyE, "L21_Phiuy_times": L21_Phiuy_times,
            "L21_Phiux": L21_Phiux, "L21_PhiuxD": L21_PhiuxD, "L21_PhiuxE": L21_PhiuxE, "L21_Phiux_times": L21_Phiux_times,
            "L21_Phixy": L21_Phixy, "L21_PhixyD": L21_PhixyD, "L21_PhixyE": L21_PhixyE, "L21_Phixy_times": L21_Phixy_times,

            "L12_Phiuy": L12_Phiuy, "L12_PhiuyD": L12_PhiuyD, "L12_PhiuyE": L12_PhiuyE, "L12_Phiuy_times": L12_Phiuy_times,
            "L12_Phiux": L12_Phiux, "L12_PhiuxD": L12_PhiuxD, "L12_PhiuxE": L12_PhiuxE, "L12_Phiux_times": L12_Phiux_times,
            "L12_Phixy": L12_Phixy, "L12_PhixyD": L12_PhixyD, "L12_PhixyE": L12_PhixyE, "L12_Phixy_times": L12_Phixy_times
        }

        np.savez(save_file, **save_dict)

        print(f"Saved to: {save_file}")
        return


    def display_message_time(self, rank_eps=1e-7, save_file="intermediate_results.npz"):
        def adjust_time(times, scale):
            return [t // scale for t in times]

        L21 = self.extract_sub_communication_matrix(direction='21')
        L12 = self.extract_sub_communication_matrix(direction='12')

        print("Message time in L21:")
        L21_D, L21_E, L21_times = row_factorization_causal(L21, rank_eps)
        L21_times = adjust_time(L21_times, 2) 
        print(f"  Adjusted Times: {L21_times}\n")

        print("Message time in L12:")
        L12_D, L12_E, L12_times = row_factorization_causal(L12, rank_eps)
        L12_times = adjust_time(L12_times, 2)
        print(f"  Adjusted Times: {L12_times}\n")

        L21_Phiuy, L21_Phiux, L21_Phixy = self.extract_Phi_subcom_mat(direction='21')
        L12_Phiuy, L12_Phiux, L12_Phixy = self.extract_Phi_subcom_mat(direction='12')

        print("Message time in L21_Phiuy:")
        L21_PhiuyD, L21_PhiuyE, L21_Phiuy_times = row_factorization_causal(L21_Phiuy, rank_eps)
        L21_Phiuy_times = adjust_time(L21_Phiuy_times, 2)
        print(f"  Adjusted Times: {L21_Phiuy_times}\n")

        print("Message time in L21_Phiux:")
        L21_PhiuxD, L21_PhiuxE, L21_Phiux_times = row_factorization_causal(L21_Phiux, rank_eps)
        L21_Phiux_times = adjust_time(L21_Phiux_times, 2)
        print(f"  Adjusted Times: {L21_Phiux_times}\n")

        print("Message time in L21_Phixy:")
        L21_PhixyD, L21_PhixyE, L21_Phixy_times = row_factorization_causal(L21_Phixy, rank_eps)
        L21_Phixy_times = adjust_time(L21_Phixy_times, 4)
        print(f"  Adjusted Times: {L21_Phixy_times}\n")

        print("Message time in L12_Phiuy:")
        L12_PhiuyD, L12_PhiuyE, L12_Phiuy_times = row_factorization_causal(L12_Phiuy, rank_eps)
        L12_Phiuy_times = adjust_time(L12_Phiuy_times, 2)
        print(f"  Adjusted Times: {L12_Phiuy_times}\n")

        print("Message time in L12_Phiux:")
        L12_PhiuxD, L12_PhiuxE, L12_Phiux_times = row_factorization_causal(L12_Phiux, rank_eps)
        L12_Phiux_times = adjust_time(L12_Phiux_times, 2)
        print(f"  Adjusted Times: {L12_Phiux_times}\n")

        print("Message time in L12_Phixy:")
        L12_PhixyD, L12_PhixyE, L12_Phixy_times = row_factorization_causal(L12_Phixy, rank_eps)
        L12_Phixy_times = adjust_time(L12_Phixy_times, 4)
        print(f"  Adjusted Times: {L12_Phixy_times}\n")

        save_dict = {
            "L21": L21, "L21_D": L21_D, "L21_E": L21_E, "L21_times": L21_times,
            "L12": L12, "L12_D": L12_D, "L12_E": L12_E, "L12_times": L12_times,

            "L21_Phiuy": L21_Phiuy, "L21_PhiuyD": L21_PhiuyD, "L21_PhiuyE": L21_PhiuyE, "L21_Phiuy_times": L21_Phiuy_times,
            "L21_Phiux": L21_Phiux, "L21_PhiuxD": L21_PhiuxD, "L21_PhiuxE": L21_PhiuxE, "L21_Phiux_times": L21_Phiux_times,
            "L21_Phixy": L21_Phixy, "L21_PhixyD": L21_PhixyD, "L21_PhixyE": L21_PhixyE, "L21_Phixy_times": L21_Phixy_times,

            "L12_Phiuy": L12_Phiuy, "L12_PhiuyD": L12_PhiuyD, "L12_PhiuyE": L12_PhiuyE, "L12_Phiuy_times": L12_Phiuy_times,
            "L12_Phiux": L12_Phiux, "L12_PhiuxD": L12_PhiuxD, "L12_PhiuxE": L12_PhiuxE, "L12_Phiux_times": L12_Phiux_times,
            "L12_Phixy": L12_Phixy, "L12_PhixyD": L12_PhixyD, "L12_PhixyE": L12_PhixyE, "L12_Phixy_times": L12_Phixy_times
        }

        np.savez(save_file, **save_dict)

        print(f"Saved to: {save_file}")
        return

        




