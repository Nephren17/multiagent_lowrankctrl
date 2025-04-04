import numpy as np
import cvxpy as cp
from SLSFinite import *
from Polytope import *
from utils import *
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import copy
import os



#### basic operations #######################################################################################

def polytope_constraints(SLS_data, Poly_x, Poly_u, Poly_w):
    constraints = SLS_data.SLP_constraints()
    # add polytope containment constraints
    Poly_xu = Poly_x.cart(Poly_u)
    Lambda = cp.Variable((np.shape(Poly_xu.H)[0], np.shape(Poly_w.H)[0]), nonneg=True)
    constraints += [Lambda @ Poly_w.H == Poly_xu.H @ SLS_data.Phi_matrix,
                    Lambda @ Poly_w.h <= Poly_xu.h]
    return constraints, Lambda

def poly_intersect(poly1, poly2):
    """
    Return the intersection of two polytopes in H-rep.
    """
    import numpy as np
    H_new = np.vstack((poly1.H, poly2.H))
    h_new = np.concatenate((poly1.h, poly2.h))
    return Polytope(H_new, h_new)



#### Constraint functions #######################################################################################

def zero_diag_blocks(phi, T, block_row, block_col):
    Tplus1 = T + 1
    expected_shape = (block_row*Tplus1, block_col*Tplus1)
    if phi.shape != expected_shape:
        raise ValueError(
            f"the shape of phi is {phi.shape}, with expectation {expected_shape}."
        )
    
    mask = np.ones((block_row, block_col))
    r_half = block_row // 2 
    c_half = block_col // 2 
    
    mask[:r_half, :c_half] = 0
    mask[r_half:, c_half:] = 0
        
    sub_blocks = []
    for t_idx in range(Tplus1):
        row_list = []
        for tau_idx in range(Tplus1):
            row_start = t_idx * block_row
            col_start = tau_idx * block_col
            sub_block = phi[row_start : row_start + block_row,
                            col_start : col_start + block_col]
            
            sub_block_zeroed = cp.multiply(mask, sub_block)
            row_list.append(sub_block_zeroed)
        sub_blocks.append(row_list)

    M_expr = cp.bmat(sub_blocks)
    return M_expr



def diagonal_identity_block_constraints_xx(Phi_xx, nx, Tplus1):
    constraints = []
    half_nx = nx // 2
    for t in range(Tplus1):
        for tau in range(Tplus1):
            row_block = t * nx
            col_block = tau * nx
            for i in range(0, half_nx):
                for j in range(half_nx, nx):
                    constraints.append(Phi_xx[row_block + i, col_block + j] == 0.0)
            for i in range(half_nx, nx):
                for j in range(0, half_nx):
                    constraints.append(Phi_xx[row_block + i, col_block + j] == 0.0)
    return constraints



def diagonal_identity_block_constraints_ux(Phi_ux, nx, nu, Tplus1):
    constraints = []
    half_nu = nu // 2
    half_nx = nx // 2

    for t in range(Tplus1):
        for tau in range(Tplus1):
            row_block = t * nu
            col_block = tau * nx
            for i in range(0, half_nu):
                for j in range(half_nx, nx):
                    constraints.append(Phi_ux[row_block + i, col_block + j] == 0.0)
            for i in range(half_nu, nu):
                for j in range(0, half_nx):
                    constraints.append(Phi_ux[row_block + i, col_block + j] == 0.0)
    return constraints


def diagonal_identity_block_constraints_xy(Phi_xy, nx, ny, Tplus1):
    constraints = []
    half_nx = nx // 2
    half_ny = ny // 2

    for t in range(Tplus1):
        for tau in range(Tplus1):
            row_block = t * nx
            col_block = tau * ny
            for i in range(0, half_nx):
                for j in range(half_ny, ny):
                    constraints.append(
                        Phi_xy[row_block + i, col_block + j] == 0.0
                    )
            for i in range(half_nx, nx):
                for j in range(0, half_ny):
                    constraints.append(
                        Phi_xy[row_block + i, col_block + j] == 0.0
                    )
    return constraints


def diagonal_identity_block_constraints_uy(Phi_uy, nu, ny, Tplus1):
    constraints = []
    half_nu = nu // 2
    half_ny = ny // 2

    for t in range(Tplus1):
        for tau in range(Tplus1):
            row_block = t * nu
            col_block = tau * ny
            for i in range(0, half_nu):
                for j in range(half_ny, ny):
                    constraints.append(Phi_uy[row_block + i, col_block + j] == 0.0)
            for i in range(half_nu, nu):
                for j in range(0, half_nu):
                    constraints.append(Phi_uy[row_block + i, col_block + j] == 0.0)
    return constraints



def quad_lower_triangular_block_constraints_xx(Phi_xx, nx, Tplus1):
    constraints = []
    half_nx = nx // 2
    
    for t in range(Tplus1):
        for tau in range(Tplus1):
            if t < tau:
                continue
                
            row_start = t * nx
            col_start = tau * nx
            
            for i_sub in range(half_nx):
                for j_sub in range(half_nx):
                    row = row_start + i_sub
                    col = col_start + half_nx + j_sub
                    constraints.append(Phi_xx[row, col] == 0.0)
    return constraints

def time_delay_constraints(Phi_uy, Phi_ux, Phi_xx, Phi_xy,nu, ny, nx, Tplus1, delay):
    constraints = []
    half_nx = nx // 2
    half_nu = nu // 2
    half_ny = ny // 2

    # =========== 1) Phi_xx ===========
    for t in range(Tplus1):
        for tau in range(Tplus1):
            if (t - tau) < delay:
                row_block = t * nx
                col_block = tau * nx
                for i in range(0, half_nx):
                    for j in range(half_nx, nx):
                        constraints.append(Phi_xx[row_block + i, col_block + j] == 0)
                for i in range(half_nx, nx):
                    for j in range(0, half_nx):
                        constraints.append(Phi_xx[row_block + i, col_block + j] == 0)

    # =========== 2) Phi_ux ===========
    for t in range(Tplus1):
        for tau in range(Tplus1):
            if (t - tau) < delay:
                row_block = t * nu
                col_block = tau * nx
                for i in range(0, half_nu):
                    for j in range(half_nx, nx):
                        constraints.append(Phi_ux[row_block + i, col_block + j] == 0)
                for i in range(half_nu, nu):
                    for j in range(0, half_nx):
                        constraints.append(Phi_ux[row_block + i, col_block + j] == 0)

    # =========== 3) Phi_xy ===========
    for t in range(Tplus1):
        for tau in range(Tplus1):
            if (t - tau) < delay:
                row_block = t * nx
                col_block = tau * ny
                for i in range(0, half_nx):
                    for j in range(half_ny, ny):
                        constraints.append(Phi_xy[row_block + i, col_block + j] == 0)
                for i in range(half_nx, nx):
                    for j in range(0, half_ny):
                        constraints.append(Phi_xy[row_block + i, col_block + j] == 0)

    # =========== 4) Phi_uy ===========
    for t in range(Tplus1):
        for tau in range(Tplus1):
            if (t - tau) < delay:
                row_block = t * nu
                col_block = tau * ny
                for i in range(0, half_nu):
                    for j in range(half_ny, ny):
                        constraints.append(Phi_uy[row_block + i, col_block + j] == 0)
                for i in range(half_nu, nu):
                    for j in range(0, half_ny):
                        constraints.append(Phi_uy[row_block + i, col_block + j] == 0)

    return constraints



def polytope_constraints_no_comm_both_ways(SLS_data, Poly_x, Poly_u, Poly_w):
    """
    Add polyconstraint: no comm between two drones
    """
    constraints = SLS_data.SLP_constraints()

    Poly_xu = Poly_x.cart(Poly_u)
    Lambda = cp.Variable((Poly_xu.H.shape[0], Poly_w.H.shape[0]), nonneg=True)
    constraints += [
        Lambda @ Poly_w.H == Poly_xu.H @ SLS_data.Phi_matrix,
        Lambda @ Poly_w.h <= Poly_xu.h
    ]

    Tplus1 = SLS_data.T + 1
    nx = SLS_data.nx
    nu = SLS_data.nu
    ny = SLS_data.ny
    constraints += diagonal_identity_block_constraints_xx(SLS_data.Phi_xx, nx, Tplus1)
    constraints += diagonal_identity_block_constraints_xy(SLS_data.Phi_xy, nx, ny, Tplus1)
    constraints += diagonal_identity_block_constraints_ux(SLS_data.Phi_ux, nx, nu, Tplus1)
    constraints += diagonal_identity_block_constraints_uy(SLS_data.Phi_uy, nu, ny, Tplus1)

    return constraints, Lambda



def partial_time_dist_poly(times_list, T, dist=5):
    '''
    This function is used when communication distance constraint is desired at specific time points
    '''
    H_comm_1time = np.array([
        [ 1,  1,  0,  0, -1, -1,  0,  0],
        [ 1, -1,  0,  0, -1,  1,  0,  0],
        [-1,  1,  0,  0,  1, -1,  0,  0],
        [-1, -1,  0,  0,  1,  1,  0,  0],
    ])
    h_comm_1time = dist * np.ones(4)


    n_constraints = 4 * len(times_list)
    n_dim = (T+1)*8

    H_time = np.zeros( (n_constraints, n_dim) )
    h_time = np.zeros( (n_constraints,) )

    for i, t in enumerate(times_list):
        row_start = i * 4
        col_start = t * 8

        H_time[row_start : row_start+4, col_start : col_start+8] = H_comm_1time
        h_time[row_start : row_start+4] = h_comm_1time

    return Polytope(H_time, h_time)




#### optimization functions #######################################################################################

def optimize(A_list, B_list, C_list, delay, Poly_x, Poly_u, Poly_w, opt_eps, norm=None):

    # constraints
    SLS_data = SLSFinite(A_list, B_list, C_list, delay, norm)
    [constraints, Lambda] = polytope_constraints(SLS_data, Poly_x, Poly_u, Poly_w)
    constraints += time_delay_constraints(SLS_data.Phi_uy, SLS_data.Phi_ux, SLS_data.Phi_xx, SLS_data.Phi_xy, SLS_data.nu, SLS_data.ny, SLS_data.nx, SLS_data.T+1, SLS_data.delay)
    constraints += SLS_data.SLP_constraints()

    # objective function
    objective = cp.Minimize(0)
    problem = cp.Problem(objective, constraints)
    result = problem.solve( solver=cp.MOSEK,
                            mosek_params = {'MSK_DPAR_INTPNT_CO_TOL_DFEAS':  opt_eps,
                                            },
                            verbose=True)

    if problem.status != cp.OPTIMAL:
        raise Exception("Solver did not converge!")



def optimize_decentral_feasibility(A_list, B_list, C_list, delay, Poly_x, Poly_u, Poly_w, opt_eps, norm=None):

    # constraints
    SLS_data = SLSFinite(A_list, B_list, C_list, delay, norm)
    [constraints, Lambda] = polytope_constraints(SLS_data, Poly_x, Poly_u, Poly_w)
    constraints += time_delay_constraints(SLS_data.Phi_uy, SLS_data.Phi_ux, SLS_data.Phi_xx, SLS_data.Phi_xy, SLS_data.nu, SLS_data.ny, SLS_data.nx, SLS_data.T+1, SLS_data.delay)
    constraints += SLS_data.SLP_constraints()

    # This is the constraints that don't allow inter-agent communication, corresponding to the decentral case
    constraints += diagonal_identity_block_constraints_xx(SLS_data.Phi_xx, SLS_data.nx, SLS_data.T + 1)
    constraints += diagonal_identity_block_constraints_xy(SLS_data.Phi_xx, SLS_data.nx, SLS_data.ny, SLS_data.T + 1)
    constraints += diagonal_identity_block_constraints_ux(SLS_data.Phi_xx, SLS_data.nx, SLS_data.nu, SLS_data.T + 1)
    constraints += diagonal_identity_block_constraints_uy(SLS_data.Phi_xx, SLS_data.nu, SLS_data.ny, SLS_data.T + 1)

    # objective function
    objective = cp.Minimize(0)
    problem = cp.Problem(objective, constraints)
    result = problem.solve( solver=cp.MOSEK,
                            mosek_params = {'MSK_DPAR_INTPNT_CO_TOL_DFEAS':  opt_eps,
                                            },
                            verbose=True)

    if problem.status != cp.OPTIMAL:
        raise Exception("Solver did not converge! Not feasiblt for no communication!")


def optimize_RTH(A_list, B_list, C_list, delay, Poly_x, Poly_u, Poly_w, N, delta, rank_eps, opt_eps):
    """
    Baseline controller
    """

    SLS_data = SLSFinite(A_list, B_list, C_list, delay)
    [constraints, Lambda] = polytope_constraints(SLS_data, Poly_x, Poly_u, Poly_w)
    constraints += time_delay_constraints(SLS_data.Phi_uy, SLS_data.Phi_ux, SLS_data.Phi_xx, SLS_data.Phi_xy, SLS_data.nu, SLS_data.ny, SLS_data.nx, SLS_data.T+1, SLS_data.delay)

    # Initialize Paramters
    W_1 = cp.Parameter(2*[SLS_data.nu*(SLS_data.T+1)])
    W_2 = cp.Parameter(2*[SLS_data.ny*(SLS_data.T+1)])
    W_1.value = delta**(-1/2)*np.eye(SLS_data.nu*(SLS_data.T+1))
    W_2.value = delta**(-1/2)*np.eye(SLS_data.ny*(SLS_data.T+1))
    result_list = N*[None]
    SLS_data_list = N*[None]
    objective = cp.Minimize(cp.norm(W_1 @ SLS_data.Phi_uy @ W_2, 'nuc'))
    #define problem
    problem = cp.Problem(objective, constraints)
    for k in range(N):
        result = problem.solve(solver=cp.MOSEK,
                                mosek_params={
                                                "MSK_DPAR_INTPNT_TOL_PFEAS": 1e-5,
                                                "MSK_DPAR_INTPNT_TOL_DFEAS": 1e-5,
                                                "MSK_DPAR_INTPNT_TOL_REL_GAP": 1e-5,
                                                "MSK_IPAR_NUM_THREADS": 4,
                                                "MSK_DPAR_OPTIMIZER_MAX_TIME": 60000,
                                            },
                               verbose=True)
        if problem.status != cp.OPTIMAL:
            raise Exception("Solver did not converge!")
        result_list[k] = result
        SLS_data_list[k] = copy.deepcopy(SLS_data)
        #update params
        [U, S, Vh] = np.linalg.svd((W_1 @ SLS_data.Phi_uy @ W_2).value , full_matrices=False)
        Y = np.linalg.inv(W_1.value).dot((W_1 @ SLS_data.Phi_uy @ W_2).value).dot(Vh.T).dot(U.T).dot(np.linalg.inv(W_1.value))
        Z = np.linalg.inv(W_2.value).dot(Vh.T).dot(U.T).dot((W_1 @ SLS_data.Phi_uy @ W_2).value).dot(np.linalg.inv(W_2.value))
        # help function
        def update_W(Q, dim, delta):
            W = (Q + delta*np.eye(dim))
            [eig, eigv] = np.linalg.eigh(W)
            assert np.all(eig > 0)
            W = eigv.dot(np.diag(eig**(-1/2))).dot(np.linalg.inv(eigv))
            return W
        W_1.value = update_W(Y, SLS_data.nu*(SLS_data.T+1), delta)
        W_2.value = update_W(Z, SLS_data.ny*(SLS_data.T+1), delta)
        
    SLS_data_list[-1].calculate_dependent_variables("Reweighted Nuclear Norm")
    SLS_data_list[-1].causal_factorization(rank_eps)
    SLS_data_list[-1].F_trunc_to_Phi_trunc()

    Poly_xu = Poly_x.cart(Poly_u)
    print("rank F:", SLS_data_list[-1].rank_F_trunc)
    print("band (D,E) = messages:", SLS_data_list[-1].E.shape[0])
    print("Error true F and truncated F:", np.max( np.abs(SLS_data_list[-1].F - SLS_data_list[-1].F_trunc) ) )
    print("Error true Phi and truncated Phi:", np.max( np.abs(SLS_data_list[-1].Phi_matrix.value - SLS_data_list[-1].Phi_trunc) ) )
    print("Error truncated polytope constraint:", np.max( np.abs(Lambda.value.dot(Poly_w.H) - Poly_xu.H.dot(SLS_data_list[-1].Phi_trunc)) ) )
    assert np.all( Lambda.value.dot(Poly_w.h) <= Poly_xu.h + 1e-6 )

    return [result_list, SLS_data_list, Lambda]#, reopt_Lambda]




def optimize_RTH_proposed(A_list, B_list, C_list, delay, Poly_x, Poly_u, Poly_w, N=10, delta=0.01, rank_eps=1e-7, opt_eps=1e-11):
    """
    Of(Phi) = Off-Diag Parts of Phi
    minimize rank(Of(Phi_uy)) + rank(Of(Phi_ux)) + rank(Of(Phi_xy)),
    s.t. Of(Phi_xx) = 0
    """

    SLS_data = SLSFinite(A_list, B_list, C_list, delay)

    [constraints, Lambda] = polytope_constraints(SLS_data, Poly_x, Poly_u, Poly_w)
    constraints += diagonal_identity_block_constraints_xx(SLS_data.Phi_xx, SLS_data.nx, SLS_data.T + 1)
    constraints += time_delay_constraints(SLS_data.Phi_uy, SLS_data.Phi_ux, SLS_data.Phi_xx, SLS_data.Phi_xy, SLS_data.nu, SLS_data.ny, SLS_data.nx, SLS_data.T+1, SLS_data.delay)

    L_phi_uy = zero_diag_blocks(SLS_data.Phi_uy, SLS_data.T, SLS_data.nu, SLS_data.ny)
    L_phi_ux = zero_diag_blocks(SLS_data.Phi_ux, SLS_data.T, SLS_data.nu, SLS_data.nx)
    L_phi_xy = zero_diag_blocks(SLS_data.Phi_xy, SLS_data.T, SLS_data.nx, SLS_data.ny)


    m_uy, n_uy = L_phi_uy.shape
    m_ux, n_ux = L_phi_ux.shape
    m_xy, n_xy = L_phi_xy.shape

    W1_left = cp.Parameter((m_uy, m_uy), PSD=True)
    W1_right= cp.Parameter((n_uy, n_uy), PSD=True)
    W2_left = cp.Parameter((m_ux, m_ux), PSD=True)
    W2_right= cp.Parameter((n_ux, n_ux), PSD=True)
    W3_left = cp.Parameter((m_xy, m_xy), PSD=True)
    W3_right= cp.Parameter((n_xy, n_xy), PSD=True)

    W1_left.value  = delta**(-0.5) * np.eye(m_uy)
    W1_right.value = delta**(-0.5) * np.eye(n_uy)
    W2_left.value  = delta**(-0.5) * np.eye(m_ux)
    W2_right.value = delta**(-0.5) * np.eye(n_ux)
    W3_left.value  = delta**(-0.5) * np.eye(m_xy)
    W3_right.value = delta**(-0.5) * np.eye(n_xy)


    # objective = cp.Minimize(cp.norm( W1_left@L1_expr@W1_right, 'nuc') + cp.norm( W2_left@L2_expr@W2_right, 'nuc'))
    objective = cp.Minimize(1.0 * cp.norm( W1_left@L_phi_uy@W1_right, 'nuc') + 1.0 * cp.norm( W2_left@L_phi_ux@W2_right, 'nuc')
    + 1.0 * cp.norm( W3_left@L_phi_xy@W3_right, 'nuc'))

    problem = cp.Problem(objective, constraints)

    result_list = []
    SLS_data_list = []

    def update_reweight_stable_rect(L_val, Wleft_val, Wright_val, delta=1e-2, epsilon=1e-5):
        U, s, Vt = np.linalg.svd(L_val, full_matrices=False)
        s = np.maximum(s, epsilon)
        inv_sqrt_s = 1.0 / np.sqrt(s)

        Lambda_U = np.diag(inv_sqrt_s)
        Lambda_V = np.diag(inv_sqrt_s)

        Wleft_new = U @ Lambda_U @ U.T
        Wright_new= Vt.T @ Lambda_V @ Vt
        return Wleft_new, Wright_new


    for k in range(N):
        result = problem.solve(solver=cp.MOSEK, 
                                mosek_params={
                                        "MSK_DPAR_INTPNT_TOL_PFEAS": 1e-5,
                                        "MSK_DPAR_INTPNT_TOL_DFEAS": 1e-5,
                                        "MSK_DPAR_INTPNT_TOL_REL_GAP": 1e-5,
                                        "MSK_IPAR_NUM_THREADS": 4,
                                        "MSK_DPAR_OPTIMIZER_MAX_TIME": 60000,
                                    },
                                verbose=True
                            )
        if problem.status != cp.OPTIMAL:
            raise Exception("Solver did not converge!")
        
        result_list.append(result)
        SLS_data_list.append(copy.deepcopy(SLS_data))

        L_phi_uy_val = L_phi_uy.value
        L_phi_ux_val = L_phi_ux.value
        L_phi_xy_val = L_phi_xy.value

        W1_left.value, W1_right.value = update_reweight_stable_rect(L_phi_uy_val, W1_left.value, W1_right.value)
        W2_left.value, W2_right.value = update_reweight_stable_rect(L_phi_ux_val, W2_left.value, W2_right.value)
        W3_left.value, W3_right.value = update_reweight_stable_rect(L_phi_xy_val, W3_left.value, W3_right.value)

    SLS_data.calculate_dependent_variables("Reweighted Nuclear Norm")
    SLS_data.causal_factorization(rank_eps)
    SLS_data.F_trunc_to_Phi_trunc()

    Poly_xu = Poly_x.cart(Poly_u)
    diff = Lambda.value@Poly_w.H - Poly_xu.H@SLS_data.Phi_trunc
    max_err = np.max( np.abs(diff) )
    print("Error truncated polytope constraint:", max_err)

    return [result_list, SLS_data, Lambda]






def optimize_RTH_proposed_lower_triangular(A_list, B_list, C_list, delay, Poly_x, Poly_u, Poly_w, N=10, delta=0.01, rank_eps=1e-7, opt_eps=1e-11):
    """
    Of(Phi) = Off-Diag Parts of Phi
    minimize rank(Of(Phi_uy)) + rank(Of(Phi_ux)) + rank(Of(Phi_xy)),
    s.t. Of(Phi_xx) = 0
    """


    SLS_data = SLSFinite(A_list, B_list, C_list, delay)

    [constraints, Lambda] = polytope_constraints(SLS_data, Poly_x, Poly_u, Poly_w)
    constraints += quad_lower_triangular_block_constraints_xx(SLS_data.Phi_xx, SLS_data.nx, SLS_data.T + 1)
    constraints += time_delay_constraints(SLS_data.Phi_uy, SLS_data.Phi_ux, SLS_data.Phi_xx, SLS_data.Phi_xy, SLS_data.nu, SLS_data.ny, SLS_data.nx, SLS_data.T+1, SLS_data.delay)

    L_phi_uy = zero_diag_blocks(SLS_data.Phi_uy, SLS_data.T, SLS_data.nu, SLS_data.ny)
    L_phi_ux = zero_diag_blocks(SLS_data.Phi_ux, SLS_data.T, SLS_data.nu, SLS_data.nx)
    L_phi_xy = zero_diag_blocks(SLS_data.Phi_xy, SLS_data.T, SLS_data.nx, SLS_data.ny)


    m_uy, n_uy = L_phi_uy.shape
    m_ux, n_ux = L_phi_ux.shape
    m_xy, n_xy = L_phi_xy.shape

    W1_left = cp.Parameter((m_uy, m_uy), PSD=True)
    W1_right= cp.Parameter((n_uy, n_uy), PSD=True)
    W2_left = cp.Parameter((m_ux, m_ux), PSD=True)
    W2_right= cp.Parameter((n_ux, n_ux), PSD=True)
    W3_left = cp.Parameter((m_xy, m_xy), PSD=True)
    W3_right= cp.Parameter((n_xy, n_xy), PSD=True)

    W1_left.value  = delta**(-0.5) * np.eye(m_uy)
    W1_right.value = delta**(-0.5) * np.eye(n_uy)
    W2_left.value  = delta**(-0.5) * np.eye(m_ux)
    W2_right.value = delta**(-0.5) * np.eye(n_ux)
    W3_left.value  = delta**(-0.5) * np.eye(m_xy)
    W3_right.value = delta**(-0.5) * np.eye(n_xy)


    # objective = cp.Minimize(cp.norm( W1_left@L1_expr@W1_right, 'nuc') + cp.norm( W2_left@L2_expr@W2_right, 'nuc'))
    objective = cp.Minimize(1.0 * cp.norm( W1_left@L_phi_uy@W1_right, 'nuc') + 1.0 * cp.norm( W2_left@L_phi_ux@W2_right, 'nuc')
    + 1.0 * cp.norm( W3_left@L_phi_xy@W3_right, 'nuc'))

    problem = cp.Problem(objective, constraints)

    result_list = []
    SLS_data_list = []

    def update_reweight_stable_rect(L_val, Wleft_val, Wright_val, delta=1e-2, epsilon=1e-5):
        U, s, Vt = np.linalg.svd(L_val, full_matrices=False)
        s = np.maximum(s, epsilon)
        inv_sqrt_s = 1.0 / np.sqrt(s)

        Lambda_U = np.diag(inv_sqrt_s)
        Lambda_V = np.diag(inv_sqrt_s)

        Wleft_new = U @ Lambda_U @ U.T
        Wright_new= Vt.T @ Lambda_V @ Vt
        return Wleft_new, Wright_new


    for k in range(N):
        result = problem.solve(solver=cp.MOSEK, 
                                mosek_params={
                                        "MSK_DPAR_INTPNT_TOL_PFEAS": 1e-5,
                                        "MSK_DPAR_INTPNT_TOL_DFEAS": 1e-5,
                                        "MSK_DPAR_INTPNT_TOL_REL_GAP": 1e-5,
                                        "MSK_IPAR_NUM_THREADS": 4,
                                        "MSK_DPAR_OPTIMIZER_MAX_TIME": 60000,
                                    },
                                verbose=True
                            )
        if problem.status != cp.OPTIMAL:
            raise Exception("Solver did not converge!")
        
        result_list.append(result)
        SLS_data_list.append(copy.deepcopy(SLS_data))

        L_phi_uy_val = L_phi_uy.value
        L_phi_ux_val = L_phi_ux.value
        L_phi_xy_val = L_phi_xy.value

        W1_left.value, W1_right.value = update_reweight_stable_rect(L_phi_uy_val, W1_left.value, W1_right.value)
        W2_left.value, W2_right.value = update_reweight_stable_rect(L_phi_ux_val, W2_left.value, W2_right.value)
        W3_left.value, W3_right.value = update_reweight_stable_rect(L_phi_xy_val, W3_left.value, W3_right.value)

    SLS_data.calculate_dependent_variables("Reweighted Nuclear Norm")
    SLS_data.causal_factorization(rank_eps)
    SLS_data.F_trunc_to_Phi_trunc()

    Poly_xu = Poly_x.cart(Poly_u)
    diff = Lambda.value@Poly_w.H - Poly_xu.H@SLS_data.Phi_trunc
    max_err = np.max( np.abs(diff) )
    print("Error truncated polytope constraint:", max_err)

    return [result_list, SLS_data, Lambda]




def optimize_decentral(A_list, B_list, C_list, delay, Poly_x, Poly_u, Poly_w, opt_eps):
    """
    No cross-communication between UAVs,
    objective = 0 for demonstration.
    """
    SLS_data = SLSFinite(A_list, B_list, C_list, delay)

    [constraints, Lambda] = polytope_constraints_no_comm_both_ways(
        SLS_data, Poly_x, Poly_u, Poly_w
    )

    objective = cp.Minimize(0)
    problem = cp.Problem(objective, constraints)

    result = problem.solve(
        solver=cp.MOSEK,
        mosek_params={
                "MSK_DPAR_INTPNT_TOL_PFEAS": 1e-5,
                "MSK_DPAR_INTPNT_TOL_DFEAS": 1e-5,
                "MSK_DPAR_INTPNT_TOL_REL_GAP": 1e-5,
                "MSK_IPAR_NUM_THREADS": 4,
                "MSK_DPAR_OPTIMIZER_MAX_TIME": 60000,
            },
        verbose=True
    )
    if problem.status != cp.OPTIMAL:
        raise Exception("Solver did not converge or feasible solution not found.")

    return [result, SLS_data, Lambda]





def optimize_RTH_offdiag_constrain_phixx_low_triangle(A_list, B_list, C_list, delay, Poly_x, Poly_u, Poly_w, N=10, delta=0.01, rank_eps=1e-7, opt_eps=1e-11):
    '''
    In this case, not all communication in Phi_xx is forbiddened. We only allow the communication from one specific agent to the other one.
    '''
    
    SLS_data = SLSFinite(A_list, B_list, C_list, delay)
    [constraints, Lambda] = polytope_constraints(SLS_data, Poly_x, Poly_u, Poly_w)

    # Constrain is switched from making Phi_xx from block diagnal to block lower triangular
    constraints += quad_lower_triangular_block_constraints_xx(SLS_data.Phi_xx, SLS_data.nx,  SLS_data.T + 1)

    constraints += time_delay_constraints(SLS_data.Phi_uy, SLS_data.Phi_ux, SLS_data.Phi_xx, SLS_data.Phi_xy, SLS_data.nu, SLS_data.ny, SLS_data.nx,SLS_data.T+1, SLS_data.delay)

    L1_expr = SLS_data.extract_offdiag_expr(direction='21')
    L2_expr = SLS_data.extract_offdiag_expr(direction='12')
    dim = 2*(SLS_data.T+1)
    W1_left = cp.Parameter((dim, dim), PSD=True)
    W1_right= cp.Parameter((dim, dim), PSD=True)
    W2_left = cp.Parameter((dim, dim), PSD=True)
    W2_right= cp.Parameter((dim, dim), PSD=True)

    W1_left.value  = delta**(-0.5) * np.eye(dim)
    W1_right.value = delta**(-0.5) * np.eye(dim)
    W2_left.value  = delta**(-0.5) * np.eye(dim)
    W2_right.value = delta**(-0.5) * np.eye(dim)

    objective = cp.Minimize(cp.norm( W1_left@L1_expr@W1_right, 'nuc') + cp.norm( W2_left@L2_expr@W2_right, 'nuc'))
    problem = cp.Problem(objective, constraints)

    result_list = []
    SLS_data_list = []

    def update_reweight_stable(L_val, Wleft_val, Wright_val, delta=1e-2, epsilon=1e-5):
        left_inv  = np.linalg.inv(Wleft_val + epsilon * np.eye(Wleft_val.shape[0]))
        right_inv = np.linalg.inv(Wright_val + epsilon * np.eye(Wright_val.shape[0]))
        Y = left_inv @ L_val @ right_inv
        Y = (Y + Y.T) / 2
        eigvals, eigvecs = np.linalg.eigh(Y)
        print(f"Eigenvalues min: {np.min(eigvals)}, max: {np.max(eigvals)}")

        min_eig = np.min(eigvals)
        if min_eig < -epsilon:
            shift = abs(min_eig) + delta
            print(f"Negative eigenvalue found ({min_eig:.6f}), shifting by {shift:.6f}")
            Y = Y + shift * np.eye(Y.shape[0])
            eigvals, eigvecs = np.linalg.eigh(Y)

        eigvals = np.maximum(eigvals, epsilon)
        W_new = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
        return W_new, W_new

    for k in range(N):
        result = problem.solve(solver=cp.MOSEK, 
                                mosek_params={
                                        "MSK_DPAR_INTPNT_TOL_PFEAS": 1e-5,
                                        "MSK_DPAR_INTPNT_TOL_DFEAS": 1e-5,
                                        "MSK_DPAR_INTPNT_TOL_REL_GAP": 1e-5,
                                        "MSK_IPAR_NUM_THREADS": 4,
                                        "MSK_DPAR_OPTIMIZER_MAX_TIME": 60000,
                                    },
                                verbose=True
                            )
        if problem.status != cp.OPTIMAL:
            raise Exception("Solver did not converge!")
        
        result_list.append(result)
        SLS_data_list.append(copy.deepcopy(SLS_data))

        L1_val = L1_expr.value
        L2_val = L2_expr.value
        W1_left.value, W1_right.value = update_reweight_stable(L1_val, W1_left.value, W1_right.value)
        W2_left.value, W2_right.value = update_reweight_stable(L2_val, W2_left.value, W2_right.value)


    SLS_data.calculate_dependent_variables("Reweighted Nuclear Norm")
    # causal_factorization
    SLS_data.causal_factorization(rank_eps)
    SLS_data.F_trunc_to_Phi_trunc()

    Poly_xu = Poly_x.cart(Poly_u)
    diff = Lambda.value@Poly_w.H - Poly_xu.H@SLS_data.Phi_trunc
    max_err = np.max( np.abs(diff) )
    print("Error truncated polytope constraint:", max_err)
    return [result_list, SLS_data, Lambda]



#### plotting functions #######################################################################################


def _draw_block_lines(ax, matrix_shape, row_block_size, col_block_size, color='lightcoral', linestyle='--', linewidth=0.8):

    nrows, ncols = matrix_shape

    for r in range(row_block_size, nrows, row_block_size):
        ax.axhline(r - 0.5, color=color, linestyle=linestyle, linewidth=linewidth)

    for c in range(col_block_size, ncols, col_block_size):
        ax.axvline(c - 0.5, color=color, linestyle=linestyle, linewidth=linewidth)


def plot_matrices_sparcity(SLS_data, save_path=None):
    phi_xx_val = SLS_data.Phi_xx.value
    phi_xy_val = SLS_data.Phi_xy.value
    phi_ux_val = SLS_data.Phi_ux.value
    phi_uy_val = SLS_data.Phi_uy.value
    F_val      = SLS_data.F 

    if phi_xx_val is None or phi_xy_val is None or phi_ux_val is None or phi_uy_val is None:
        raise ValueError("Some of Phi_xx, Phi_xy, Phi_ux, Phi_uy is None. The solver might not have run or not be optimal.")
    if F_val is None:
        raise ValueError("SLS_data.F is None. You must run calculate_dependent_variables(...) first.")

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    prec = 1e-7
    mk_size = 1.0

    ax_xx = axes[0, 0]
    ax_xy = axes[0, 1]
    ax_ux = axes[0, 2]

    ax_uy = axes[1, 0]
    ax_F  = axes[1, 1]

    ax_xx.spy(phi_xx_val, precision=prec, markersize=mk_size, aspect='auto')
    ax_xx.set_title("Phi_xx")
    _draw_block_lines(ax_xx, phi_xx_val.shape, row_block_size=8, col_block_size=8)

    ax_xy.spy(phi_xy_val, precision=prec, markersize=mk_size, aspect='auto')
    ax_xy.set_title("Phi_xy")
    _draw_block_lines(ax_xy, phi_xy_val.shape, row_block_size=8, col_block_size=4)

    ax_ux.spy(phi_ux_val, precision=prec, markersize=mk_size, aspect='auto')
    ax_ux.set_title("Phi_ux")
    _draw_block_lines(ax_ux, phi_ux_val.shape, row_block_size=4, col_block_size=8)

    ax_uy.spy(phi_uy_val, precision=prec, markersize=mk_size, aspect='auto')
    ax_uy.set_title("Phi_uy")
    _draw_block_lines(ax_uy, phi_uy_val.shape, row_block_size=4, col_block_size=4)

    ax_F.spy(F_val, precision=prec, markersize=mk_size, aspect='auto')
    ax_F.set_title("F (closed-loop)")
    _draw_block_lines(ax_F, F_val.shape, row_block_size=4, col_block_size=4)

    axes[1, 2].axis("off")
    axes[1, 2].set_title("")

    fig.suptitle("Phi and F Matrices Visualization", fontsize=14)
    fig.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()





def plot_two_trajectory(SLS_data, Poly_x, Poly_w, center_times, radius_times, T, save_path,
                        process_noise=0.05, meas_noise=0.05, dt=1.0, seed=23):
    np.random.seed(seed)
    
    num_traj = 4
    nx = SLS_data.nx
    nu = SLS_data.nu
    ny = SLS_data.ny

    colors_traj = ['#FF0000', '#5375E4']
    box_color = ['#E07197','#F1BE00','#989898']
    marker_size = 6

    all_trajectories = []
    for sample in range(num_traj):
        x0 = np.zeros(nx)
        for i in range(2):
            x_center = center_times[0][4*i]
            x_radius = radius_times[0][4*i]
            x0[4*i] = np.random.uniform(x_center - x_radius, x_center + x_radius)
            
            y_center = center_times[0][4*i+1]
            y_radius = radius_times[0][4*i+1]
            x0[4*i+1] = np.random.uniform(y_center - y_radius, y_center + y_radius)

        actual_traj = np.zeros((T+1, nx))
        y_meas = np.zeros((T+1, ny))
        
        actual_traj[0] = x0.copy()
        y_meas[0] = SLS_data.C_list[0] @ x0 + np.random.normal(0, meas_noise, ny)

        for t in range(T):
            F_block = SLS_data.F[t*nu:(t+1)*nu, :(t+1)*ny]
            u = F_block @ y_meas[:t+1].flatten()
            w = np.random.normal(0, process_noise, nx)
            x_next = SLS_data.A_list[t] @ actual_traj[t] + SLS_data.B_list[t] @ u + w
            actual_traj[t+1] = x_next
            y_meas[t+1] = SLS_data.C_list[t+1] @ x_next + np.random.normal(0, meas_noise, ny)
        
        all_trajectories.append(actual_traj)

    fig, ax = plt.subplots(figsize=(10, 8))

    for spine in ax.spines.values():
        spine.set_linewidth(2.0)

    for sample_idx, actual_traj in enumerate(all_trajectories):
        for i in range(2):
            x = actual_traj[:, 4*i]
            y = actual_traj[:, 4*i+1]

            ax.plot(x, y, color=colors_traj[i], linewidth=2.5, alpha=0.9)

            for t_marker in [0, 5, 10]:
                if t_marker <= T:
                    ax.plot(x[t_marker], y[t_marker], marker='o', markersize=12,
                            color=colors_traj[i], markeredgecolor='white', markeredgewidth=2.3, alpha=0.9)



    box_time_points = [0, 5, 10]
    for t in box_time_points:
        if t >= len(center_times):
            continue
        for i in range(2):
            center = center_times[t][4*i:4*i+2]
            radius = radius_times[t][4*i:4*i+2]
            if t == 0:
                linestyle = '--'
            elif t == 5:
                linestyle = '-.'
            else:
                linestyle = '-'
            alpha_val = 0.9

            rect = patches.Rectangle(
                (center[0]-radius[0], center[1]-radius[1]),
                2*radius[0], 2*radius[1],
                edgecolor=box_color[int(t/5)], facecolor='none',
                linestyle='-', linewidth=3.3, alpha=alpha_val
            )
            ax.add_patch(rect)

    ax.set_xlabel('X Position', fontsize=22)
    ax.set_ylabel('Y Position', fontsize=22)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2, linewidth=2.0)

    legend_elements = [
        Line2D([0], [0], color=box_color[0], linestyle='-', lw=2.5, label='Start Box'),
        Line2D([0], [0], color=box_color[1], linestyle='-', lw=2.5, label='Mid Box'),
        Line2D([0], [0], color=box_color[2], linestyle='-', lw=2.5, label='End Box'),
        Line2D([0], [0], color=colors_traj[0], lw=3, label='Vehicle 1'),
        Line2D([0], [0], color=colors_traj[1], lw=3, label='Vehicle 2'),
    ]


    legend = fig.legend(
        handles=legend_elements,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.05),
        ncol=2,
        frameon=True,
        edgecolor='black',
        fancybox=False,
        title_fontsize=18,
        fontsize=16
    )
    legend.get_frame().set_linewidth(2.0)
    legend.get_frame().set_edgecolor('black')

    plt.subplots_adjust(bottom=0.15)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

    print("Finished plotting trajectories")




def plot_two_noisy(SLS_data, Poly_x, Poly_w, center_times, radius_times, T, save_path,
                    process_noise_scales=[0.25, 0.60, 0.05, 0.05, 0.05, 0.10, 0.05, 0.05],
                    meas_noise_scales=[0.05, 0.05, 0.05, 0.05], dt=1.0, seed=27):


    np.random.seed(seed)

    nx = SLS_data.nx
    nu = SLS_data.nu
    ny = SLS_data.ny

    num_traj = 2
    colors_traj = ['#FF0000', '#5375E4']
    box_color = ['#E07197', '#989898']
    marker_size = 6

    all_trajectories = []

    for sample in range(num_traj):
        x0 = np.zeros(nx)
        for i in [0, 1]:
            for j in range(4):
                center = center_times[0][4 * i + j]
                radius = radius_times[0][4 * i + j]
                x0[4 * i + j] = np.random.uniform(center - radius, center + radius)

        actual_traj = np.zeros((T+1, nx))
        y_meas = np.zeros((T+1, ny))
        actual_traj[0] = x0.copy()

        y_meas[0] = SLS_data.C_list[0] @ x0 + [
            np.random.uniform(-meas_noise_scales[0], meas_noise_scales[0]),
            np.random.uniform(- meas_noise_scales[1], meas_noise_scales[1]),
            np.random.uniform(-meas_noise_scales[2], meas_noise_scales[2]),
            np.random.uniform(-meas_noise_scales[3], meas_noise_scales[3])
        ]

        for t in range(T):
            F_block = SLS_data.F[t*nu:(t+1)*nu, :(t+1)*ny]
            u = F_block @ y_meas[:t+1].flatten()

            w = np.array([
               np.random.uniform(-process_noise_scales[i], process_noise_scales[i])
                for i in range(nx)
            ])

            x_next = SLS_data.A_list[t] @ actual_traj[t] + SLS_data.B_list[t] @ u + w
            actual_traj[t+1] = x_next

            y_meas[t+1] = SLS_data.C_list[t+1] @ x_next + [
                np.random.uniform(-meas_noise_scales[i], meas_noise_scales[i]) for i in range(ny)
            ]

        all_trajectories.append(actual_traj)

    fig, ax = plt.subplots(figsize=(10, 8))

    for spine in ax.spines.values():
        spine.set_linewidth(2.0)

    for sample_idx, actual_traj in enumerate(all_trajectories):
        for uav_id in [0, 1]:
            offset = 4 * uav_id
            x = actual_traj[:, offset]
            y = actual_traj[:, offset+1]

            ax.plot(x, y, color=colors_traj[uav_id], linewidth=2.5, alpha=0.8)

            for t_marker in [0, 3, 7, 10]:
                if t_marker <= T:
                    ax.plot(x[t_marker], y[t_marker], marker='o', markersize=12,
                            color=colors_traj[uav_id], markeredgecolor='white',
                            markeredgewidth=2.3, alpha=0.9)
            for t_marker in [3, 7]:
                if t_marker <= T:
                    x1 = actual_traj[t_marker, 0]
                    y1 = actual_traj[t_marker, 1]
                    x2 = actual_traj[t_marker, 4]
                    y2 = actual_traj[t_marker, 5]

                    ax.plot([x1, x2], [y1, y2],
                            linestyle='--', linewidth=3.3,
                            color='#F1BE00',
                            alpha=0.9, zorder=10)

    box_time_points = [0, 10]
    for t in box_time_points:
        if t >= len(center_times):
            continue
        for i in [0,1]:
            center = center_times[t][4*i:4*i+2]
            radius = radius_times[t][4*i:4*i+2]
            linestyle = '-' if t == 0 else ('-' if t == 10 else '-')
            alpha_val = 0.9

            rect = patches.Rectangle(
                (center[0]-radius[0], center[1]-radius[1]),
                2*radius[0], 2*radius[1],
                edgecolor=box_color[int(t/10)], facecolor='none',
                linestyle=linestyle, linewidth=3.3, alpha=alpha_val
            )
            ax.add_patch(rect)

    ax.set_xlabel('X Position', fontsize=22)
    ax.set_ylabel('Y Position', fontsize=22)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2, linewidth=2.0)

    legend_elements = [
            Line2D([0], [0], color=box_color[0], linestyle='-', lw=2.5, label='Start Box'),
            # Line2D([0], [0], color=box_color, linestyle='-', lw=2.5, label='Mid Box'),
            Line2D([0], [0], color=box_color[1], linestyle='-', lw=2.5, label='End Box'),
            Line2D([0], [0], color='#F1BE00', linestyle='--', lw=2.5, label='Distance'),
            Line2D([0], [0], color=colors_traj[0], lw=3, label='Vehicle 1'),
            Line2D([0], [0], color=colors_traj[1], lw=3, label='Vehicle 2')
        ]

    legend = fig.legend(
        handles=legend_elements,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.05),
        ncol=2,
        frameon=True,
        edgecolor='black',
        fancybox=False,
        title_fontsize=18,
        fontsize=16
    )
    legend.get_frame().set_linewidth(2.0)
    legend.get_frame().set_edgecolor('black')

    plt.subplots_adjust(bottom=0.15)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

    print("Finished plotting noisy trajectories")



