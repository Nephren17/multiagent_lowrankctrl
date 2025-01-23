import numpy as np
import cvxpy as cp
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
    def __init__(self, A_list, B_list, C_list, norm=None):
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
        # init variables
        assert len(A_list) == len(B_list) == len(C_list)
        # define dimanesions
        self.T = len(A_list) - 1
        self.nx = A_list[0].shape[0]
        self.nu = B_list[0].shape[1]
        self.ny = C_list[0].shape[0]
        # define optimization variables
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

    def extract_sub_communication_matrix(self, direction='21'):
        if self.F is None:
            raise ValueError("self.F is not computed yet. Please run calculate_dependent_variables() first.")

        block_size_out = 2
        block_size_in  = 2

        if direction == '21':
            row_slice = slice(0, 2)
            col_slice = slice(2, 4)
        elif direction == '12':
            row_slice = slice(2, 4)
            col_slice = slice(0, 2)
        else:
            raise ValueError("direction must be '21' or '12'")

        Tplus1 = self.T + 1
        big_mat = np.zeros((block_size_out * Tplus1, block_size_in * Tplus1))

        for t in range(Tplus1):
            for tau in range(t+1):
                F_block = self.F[t*self.nu : (t+1)*self.nu,
                                tau*self.ny: (tau+1)*self.ny]
                sub_2x2 = F_block[row_slice, col_slice]

                row_offset = t * block_size_out
                col_offset = tau * block_size_in
                big_mat[row_offset : row_offset+block_size_out,
                        col_offset : col_offset+block_size_in] = sub_2x2

        return big_mat


    def compute_communication_messages(self, rank_eps=1e-7):
        L21 = self.extract_sub_communication_matrix(direction='21')
        L12 = self.extract_sub_communication_matrix(direction='12')

        rank_21 = np.linalg.matrix_rank(L21, tol=rank_eps)
        rank_12 = np.linalg.matrix_rank(L12, tol=rank_eps)

        return (rank_21, rank_12)

    def extract_offdiag_expr(self, direction='21'):
        Tplus1 = self.T + 1
        if self.nu != 4 or self.ny != 4:
            raise ValueError("extract_offdiag_expr assumes nu=4, ny=4 for 2 UAV each 2*2. Adjust if needed.")

        if direction == '21':
            row_start, row_end = 0,   2*Tplus1
            col_start, col_end = 2*Tplus1, 4*Tplus1
        elif direction == '12':
            row_start, row_end = 2*Tplus1, 4*Tplus1
            col_start, col_end = 0,   2*Tplus1
        else:
            raise ValueError("direction must be '21' or '12'")

        return self.Phi_uy[row_start:row_end, col_start:col_end]


    def compute_offdiag_rank_of_Phi(self, rank_eps=1e-7):
        Phi_uy_val = self.Phi_uy.value
        Tplus1 = self.T+1
        rowU1_start, rowU1_end = 0,   2*Tplus1
        colU2_start, colU2_end = 2*Tplus1, 4*Tplus1
        L1 = Phi_uy_val[rowU1_start:rowU1_end, colU2_start:colU2_end]
        
        rowU2_start, rowU2_end = 2*Tplus1, 4*Tplus1
        colU1_start, colU1_end = 0,        2*Tplus1
        L2 = Phi_uy_val[rowU2_start:rowU2_end, colU1_start:colU1_end]
        
        rank1 = np.linalg.matrix_rank(L1, tol=rank_eps)
        rank2 = np.linalg.matrix_rank(L2, tol=rank_eps)
        return rank1, rank2




    def polytope_constraints_diagI(SLS_data, Poly_x, Poly_u, Poly_w):
        constraints = SLS_data.SLP_constraints()
        Poly_xu = Poly_x.cart(Poly_u)
        Lambda = cp.Variable((Poly_xu.H.shape[0], Poly_w.H.shape[0]), nonneg=True)
        constraints += [Lambda @ Poly_w.H == Poly_xu.H @ SLS_data.Phi_matrix,
                        Lambda @ Poly_w.h <= Poly_xu.h]

        Tplus1 = SLS_data.T + 1
        nx = SLS_data.nx
        constraints += diagonal_identity_block_constraints_xx(SLS_data.Phi_xx, nx, Tplus1)
        if nx == SLS_data.ny:
            constraints += diagonal_identity_block_constraints_xy(SLS_data.Phi_xy, nx, nx, Tplus1)

        return constraints, Lambda




