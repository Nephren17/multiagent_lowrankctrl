import numpy as np
import cvxpy as cp
from SLSFinite import *
from Polytope import *
from utils import *
from functions import *
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.linalg import block_diag
from matplotlib.lines import Line2D
import copy
import os



### Constraints and Extract ###########################################################################

def swarm_time_delay_constraints(Phi_uy, Phi_ux, Phi_xx, Phi_xy, n_uwv, Tplus1, delay):
    '''
    Delay constraints for # uwv > 2
    '''
    constraints = []

    state_dim_per_uwv = 4
    input_dim_per_uwv = 2
    output_dim_per_uwv = 2
    
    total_state_dim = n_uwv * state_dim_per_uwv
    total_input_dim = n_uwv * input_dim_per_uwv
    total_output_dim = n_uwv * output_dim_per_uwv

    def add_block_constraints(Phi, single_row_dim, single_col_dim, desc):
        block_row_dim = single_row_dim * n_uwv
        block_col_dim = single_col_dim * n_uwv

        for t in range(Tplus1):
            for tau in range(Tplus1):
                if (t - tau) < delay:
                    t_start = t * block_row_dim
                    t_end = (t+1) * block_row_dim
                    tau_start = tau * block_col_dim
                    tau_end = (tau+1) * block_col_dim
                    
                    time_block = Phi[t_start:t_end, tau_start:tau_end]
                    
                    for i in range(n_uwv):
                        for j in range(n_uwv):
                            if i != j:
                                row_start = i * single_row_dim
                                row_end = (i+1) * single_row_dim
                                col_start = j * single_col_dim
                                col_end = (j+1) * single_col_dim
                                
                                constraints.append(
                                    time_block[row_start:row_end, col_start:col_end] == 0
                                )
        return constraints

    constraints += add_block_constraints(Phi_xx, state_dim_per_uwv, state_dim_per_uwv, "Phi_xx")
    constraints += add_block_constraints(Phi_ux, input_dim_per_uwv, state_dim_per_uwv, "Phi_ux")
    constraints += add_block_constraints(Phi_xy, state_dim_per_uwv, output_dim_per_uwv, "Phi_xy")
    constraints += add_block_constraints(Phi_uy, input_dim_per_uwv, output_dim_per_uwv, "Phi_uy")
    
    return constraints


def swarm_phixx_nocomm_constraint(Phi_xx, n_uav, Tplus1):
    """
    Constrain Phi_xx to no communication
    """
    constraints = []
    state_dim_per_uav=4
    total_state_dim = n_uav * state_dim_per_uav
    
    block_indices = [i*state_dim_per_uav for i in range(n_uav+1)]
    
    for t in range(Tplus1):
        for tau in range(Tplus1):
            t_start = t * total_state_dim
            t_end = (t+1) * total_state_dim
            tau_start = tau * total_state_dim
            tau_end = (tau+1) * total_state_dim
            
            time_block = Phi_xx[t_start:t_end, tau_start:tau_end]
            
            for i in range(n_uav):
                for j in range(n_uav):
                    if i != j:
                        row_start = block_indices[i]
                        row_end = block_indices[i+1]
                        
                        col_start = block_indices[j]
                        col_end = block_indices[j+1]
                        
                        constraints.append(
                            time_block[row_start:row_end, col_start:col_end] == 0
                        )
    return constraints


def swarm_zero_diag_blocks(phi, T, n_uwv, state_row, state_col):
    """
    extract communication part in phis
    """
    if isinstance(phi, cp.expressions.expression.Expression):
        m, n = phi.shape
    else:
        m, n = phi.shape
    
    time_block_size_row = n_uwv * state_row
    time_block_size_col = n_uwv * state_col
    
    Tplus1 = T + 1
    expected_rows = time_block_size_row * Tplus1
    expected_cols = time_block_size_col * Tplus1
    
    if m != expected_rows or n != expected_cols:
        raise ValueError(f"dimention error\n")
    
    mask = np.ones((m, n))
    
    for t in range(Tplus1):
        for tau in range(Tplus1):
            t_row_start = t * time_block_size_row
            t_col_start = tau * time_block_size_col
            
            for u in range(n_uwv):
                u_row_start = t_row_start + u * state_row
                u_col_start = t_col_start + u * state_col
                
                mask[
                    u_row_start : u_row_start+state_row,
                    u_col_start : u_col_start+state_col
                ] = 0
    
    if isinstance(phi, cp.Variable):
        return cp.multiply(phi, mask)
    else:
        return cp.multiply(phi, mask)


# def swarm_comm_blockdiag(Phi, T, n_uwv, state_row, state_col):

#     Tplus1 = T + 1
#     expected_rows = n_uwv * Tplus1 * state_row
#     expected_cols = n_uwv * Tplus1 * state_col

#     if isinstance(Phi, cp.Expression):
#         m, n = Phi.shape
#     else:
#         m, n = Phi.shape
#     if m != expected_rows or n != expected_cols:
#         raise ValueError(f"Size error: expected ({expected_rows}, {expected_cols}),but now({m}, {n})")

#     comm_pairs = [(i, j) for i in range(n_uwv) for j in range(n_uwv) if i != j]
    
#     block_rows = state_row * Tplus1
#     block_cols = state_col * Tplus1

#     comm_blocks = []
#     for (i, j) in comm_pairs:
#         time_blocks = []
#         for t in range(Tplus1):
#             row_blocks = []
#             for tau in range(Tplus1):
#                 row_start = t * n_uwv * state_row + i * state_row
#                 col_start = tau * n_uwv * state_col + j * state_col
#                 sub_block = Phi[row_start:row_start+state_row, 
#                               col_start:col_start+state_col]
#                 row_blocks.append(sub_block)
#             time_blocks.append(cp.hstack(row_blocks) if isinstance(Phi, cp.Expression) else np.hstack(row_blocks))
#         block_matrix = cp.vstack(time_blocks) if isinstance(Phi, cp.Expression) else np.vstack(time_blocks)
#         comm_blocks.append(block_matrix)

#     if isinstance(Phi, cp.Expression):
#         zero_block = cp.Constant(0) * cp.Variable((block_rows, block_cols))
#         diag_blocks = []
#         for i, blk in enumerate(comm_blocks):
#             row = [zero_block] * len(comm_blocks)
#             row[i] = blk
#             diag_blocks.append(row)
#         block_diag_matrix = cp.bmat(diag_blocks)
#     else:
#         block_diag_matrix = block_diag(*comm_blocks)

#         # return block_diag_matrix
#         return comm_blocks


def swarm_comm_blockdiag_distributed(Phi, T, n_uwv, state_row, state_col):
    Tplus1 = T + 1
    expected_rows = n_uwv * Tplus1 * state_row
    expected_cols = n_uwv * Tplus1 * state_col

    if not isinstance(Phi, (np.ndarray, cp.Expression)):
        raise TypeError("Phi must be numpy or cp expression")

    if isinstance(Phi, cp.Expression):
        m, n = Phi.shape
    else:
        m, n = Phi.shape

    if m != expected_rows or n != expected_cols:
        raise ValueError(f"size is expected ({expected_rows}, {expected_cols}), but now ({m}, {n})")

    comm_pairs = [(i, j) for i in range(n_uwv) for j in range(n_uwv) if i != j]
    comm_blocks = []

    for (i, j) in comm_pairs:
        time_blocks = []
        for t in range(Tplus1):
            row_blocks = []
            for tau in range(Tplus1):
                row_start = t * n_uwv * state_row + i * state_row
                col_start = tau * n_uwv * state_col + j * state_col
                sub_block = Phi[row_start:row_start+state_row, 
                              col_start:col_start+state_col]
                row_blocks.append(sub_block)
            time_blocks.append(cp.hstack(row_blocks) if isinstance(Phi, cp.Expression) else np.hstack(row_blocks))
        block_matrix = cp.vstack(time_blocks) if isinstance(Phi, cp.Expression) else np.vstack(time_blocks)
        comm_blocks.append(block_matrix)

    return comm_blocks





### Optimization Functions ################################################################

def swarm_optimize(A_list, B_list, C_list, delay, n_uwv, Poly_x, Poly_u, Poly_w, opt_eps, norm=None):
    """
    Optimization feasiblity test for swarm
    """
    SLS_data = SLSFinite(A_list, B_list, C_list, delay, norm)
    [constraints, Lambda] = polytope_constraints(SLS_data, Poly_x, Poly_u, Poly_w)
    constraints += swarm_phixx_nocomm_constraint(SLS_data.Phi_xx, n_uwv, SLS_data.T+1)
    # constraints += swarm_time_delay_constraints(SLS_data.Phi_uy, SLS_data.Phi_ux, SLS_data.Phi_xx, SLS_data.Phi_xy, n_uwv, SLS_data.T+1, SLS_data.delay)

    objective = cp.Minimize(0)
    problem = cp.Problem(objective, constraints)
    result = problem.solve( solver=cp.MOSEK,
                            mosek_params = {'MSK_DPAR_INTPNT_CO_TOL_DFEAS':  opt_eps,
                                            },
                            verbose=True)

    if problem.status != cp.OPTIMAL:
        raise Exception("Solver did not converge!")




def swarm_optimize_decentral_feasibility(A_list, B_list, C_list, delay, n_uwv, Poly_x, Poly_u, Poly_w, opt_eps, norm=None):
    """
    Optimization feasiblity test for swarm
    """
    SLS_data = SLSFinite(A_list, B_list, C_list, delay, norm)
    [constraints, Lambda] = polytope_constraints(SLS_data, Poly_x, Poly_u, Poly_w)
    constraints += swarm_phixx_nocomm_constraint(SLS_data.Phi_xx, n_uwv, SLS_data.T+1)

    # decentral controller is equivalent to a controller with a controller with delay > T
    constraints += swarm_time_delay_constraints(SLS_data.Phi_uy, SLS_data.Phi_ux, SLS_data.Phi_xx, SLS_data.Phi_xy, n_uwv, SLS_data.T+1, SLS_data.T+1)

    objective = cp.Minimize(0)
    problem = cp.Problem(objective, constraints)
    result = problem.solve( solver=cp.MOSEK,
                            mosek_params = {'MSK_DPAR_INTPNT_CO_TOL_DFEAS':  opt_eps,
                                            },
                            verbose=True)

    if problem.status != cp.OPTIMAL:
        raise Exception("Solver did not converge! Decentral contorller is not feasible for this task.")


def swarm_optimize_RTH(A_list, B_list, C_list, delay, n_uwv, Poly_x, Poly_u, Poly_w, N, delta, rank_eps, opt_eps):
    """
    Baseline controller
    """

    SLS_data = SLSFinite(A_list, B_list, C_list, delay)
    [constraints, Lambda] = polytope_constraints(SLS_data, Poly_x, Poly_u, Poly_w)
    # constraints += swarm_time_delay_constraints(SLS_data.Phi_uy, SLS_data.Phi_ux, SLS_data.Phi_xx, SLS_data.Phi_xy, n_uwv, SLS_data.T+1, SLS_data.delay)

    # Initialize Paramters
    W_1 = cp.Parameter(2*[SLS_data.nu*(SLS_data.T+1)])
    W_2 = cp.Parameter(2*[SLS_data.ny*(SLS_data.T+1)])
    W_1.value = delta**(-1/2)*np.eye(SLS_data.nu*(SLS_data.T+1))
    W_2.value = delta**(-1/2)*np.eye(SLS_data.ny*(SLS_data.T+1))
    result_list = N*[None]
    SLS_data_list = N*[None]
    objective = cp.Minimize(cp.norm(W_1 @ SLS_data.Phi_uy @ W_2, 'nuc'))
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
    assert np.all( Lambda.value.dot(Poly_w.h) <= Poly_xu.h + 1e-6 )

    return [result_list, SLS_data_list, Lambda]





# def swarm_optimize_RTH_offdiag_three_phis_constrain_phixx(A_list, B_list, C_list, delay, n_uwv, Poly_x, Poly_u, Poly_w, N=10, delta=0.01, rank_eps=1e-7, opt_eps=1e-11):
#     """
#     Of(Phi) = Off-Diag Parts of Phi
#     minimize rank(Of(Phi_uy)) + rank(Of(Phi_ux)) + rank(Of(Phi_xy)),
#     s.t. Of(Phi_xx) = 0
#     """
#     with open("check.txt", "a") as f:
#         f.write(f"Begin optimize_RTH_offdiag_three_phis_constrain_phixx\n")

#     # poly constraints
#     SLS_data = SLSFinite(A_list, B_list, C_list, delay)

#     [constraints, Lambda] = polytope_constraints(SLS_data, Poly_x, Poly_u, Poly_w)
#     constraints += diagonal_identity_block_constraints_xx(SLS_data.Phi_xx, SLS_data.nx, SLS_data.T + 1)
#     # constraints += time_delay_constraints(SLS_data.Phi_uy, SLS_data.Phi_ux, SLS_data.Phi_xx, SLS_data.Phi_xy, SLS_data.nu, SLS_data.ny, SLS_data.nx, SLS_data.T+1, SLS_data.delay)

#     L_phi_uy = swarm_comm_blockdiag(SLS_data.Phi_uy, SLS_data.T, n_uwv, 2, 2)
#     L_phi_ux = swarm_comm_blockdiag(SLS_data.Phi_ux, SLS_data.T, n_uwv, 2, 4)
#     L_phi_xy = swarm_comm_blockdiag(SLS_data.Phi_xy, SLS_data.T, n_uwv, 4, 2)

#     m_uy, n_uy = L_phi_uy.shape
#     m_ux, n_ux = L_phi_ux.shape
#     m_xy, n_xy = L_phi_xy.shape

#     W1_left = cp.Parameter((m_uy, m_uy), PSD=True)
#     W1_right= cp.Parameter((n_uy, n_uy), PSD=True)
#     W2_left = cp.Parameter((m_ux, m_ux), PSD=True)
#     W2_right= cp.Parameter((n_ux, n_ux), PSD=True)
#     W3_left = cp.Parameter((m_xy, m_xy), PSD=True)
#     W3_right= cp.Parameter((n_xy, n_xy), PSD=True)

#     W1_left.value  = delta**(-0.5) * np.eye(m_uy)
#     W1_right.value = delta**(-0.5) * np.eye(n_uy)
#     W2_left.value  = delta**(-0.5) * np.eye(m_ux)
#     W2_right.value = delta**(-0.5) * np.eye(n_ux)
#     W3_left.value  = delta**(-0.5) * np.eye(m_xy)
#     W3_right.value = delta**(-0.5) * np.eye(n_xy)


#     objective = cp.Minimize(1.0 * cp.norm( W1_left@L_phi_uy@W1_right, 'nuc') + 1.0 * cp.norm( W2_left@L_phi_ux@W2_right, 'nuc')
#     + 1.0 * cp.norm( W3_left@L_phi_xy@W3_right, 'nuc'))

#     problem = cp.Problem(objective, constraints)

#     result_list = N*[None]
#     SLS_data_list = N*[None]

#     def update_reweight_stable_rect(L_val, Wleft_val, Wright_val, delta=1e-2, epsilon=1e-5):
#         U, s, Vt = np.linalg.svd(L_val, full_matrices=False)
#         s = np.maximum(s, epsilon)
#         inv_sqrt_s = 1.0 / np.sqrt(s)

#         Lambda_U = np.diag(inv_sqrt_s)
#         Lambda_V = np.diag(inv_sqrt_s)

#         Wleft_new = U @ Lambda_U @ U.T
#         Wright_new= Vt.T @ Lambda_V @ Vt
#         return Wleft_new, Wright_new


#     for k in range(N):
#         result = problem.solve(solver=cp.MOSEK, 
#                                 mosek_params={
#                                         "MSK_DPAR_INTPNT_TOL_PFEAS": 1e-5,
#                                         "MSK_DPAR_INTPNT_TOL_DFEAS": 1e-5,
#                                         "MSK_DPAR_INTPNT_TOL_REL_GAP": 1e-5,
#                                         "MSK_IPAR_NUM_THREADS": 4,
#                                         "MSK_DPAR_OPTIMIZER_MAX_TIME": 60000,
#                                     },
#                                 verbose=True
#                             )
#         if problem.status != cp.OPTIMAL:
#             raise Exception("Solver did not converge!")
        
#         result_list[k] = result
#         SLS_data_list[k] = copy.deepcopy(SLS_data)

#         L_phi_uy_val = L_phi_uy.value
#         L_phi_ux_val = L_phi_ux.value
#         L_phi_xy_val = L_phi_xy.value

#         W1_left.value, W1_right.value = update_reweight_stable_rect(L_phi_uy_val, W1_left.value, W1_right.value)
#         W2_left.value, W2_right.value = update_reweight_stable_rect(L_phi_ux_val, W2_left.value, W2_right.value)
#         W3_left.value, W3_right.value = update_reweight_stable_rect(L_phi_xy_val, W3_left.value, W3_right.value)

#     SLS_data.calculate_dependent_variables("Reweighted Nuclear Norm")
#     # causal_factorization
#     SLS_data.causal_factorization(rank_eps)
#     SLS_data.F_trunc_to_Phi_trunc()

#     Poly_xu = Poly_x.cart(Poly_u)
#     # example check => truncated constraint
#     diff = Lambda.value@Poly_w.H - Poly_xu.H@SLS_data.Phi_trunc
#     max_err = np.max( np.abs(diff) )
#     print("Error truncated polytope constraint:", max_err)

#     return [result_list, SLS_data, Lambda]



# def swarm_optimize_RTH_offdiag_three_phis_constrain_phixx_optimized_distributed(A_list, B_list, C_list, delay, n_uwv, Poly_x, Poly_u, Poly_w, N=10, delta=0.01, rank_eps=1e-7, opt_eps=1e-11):
#     with open("check.txt", "a") as f:
#         f.write(f"Begin optimize_RTH_offdiag_three_phis_constrain_phixx (Distributed)\n")

#     SLS_data = SLSFinite(A_list, B_list, C_list, delay)
#     [constraints, Lambda] = polytope_constraints(SLS_data, Poly_x, Poly_u, Poly_w)
#     constraints += diagonal_identity_block_constraints_xx(SLS_data.Phi_xx, SLS_data.nx, SLS_data.T + 1)

#     blocks_uy = swarm_comm_blockdiag_distributed(SLS_data.Phi_uy, SLS_data.T, n_uwv, 2, 2)
#     blocks_ux = swarm_comm_blockdiag_distributed(SLS_data.Phi_ux, SLS_data.T, n_uwv, 2, 4)
#     blocks_xy = swarm_comm_blockdiag_distributed(SLS_data.Phi_xy, SLS_data.T, n_uwv, 4, 2)

#     if blocks_uy == None:
#         print("uy is none\n")
#     if blocks_ux == None:
#         print("ux is none\n")
#     if blocks_xy == None:
#         print("xy is none\n")

#     W_params = []
#     for blk in blocks_uy + blocks_ux + blocks_xy:
#         m, n = blk.shape
#         W_left = cp.Parameter((m, m), PSD=True)
#         W_right = cp.Parameter((n, n), PSD=True)
#         W_left.value = delta**(-0.5) * np.eye(m)
#         W_right.value = delta**(-0.5) * np.eye(n)
#         W_params.append( (W_left, W_right) )

#     objective_terms = []
#     for idx, blk in enumerate(blocks_uy):
#         W_left, W_right = W_params[idx]
#         objective_terms.append(1.0 * cp.norm(W_left @ blk @ W_right, 'nuc'))
    
#     offset = len(blocks_uy)
#     for idx, blk in enumerate(blocks_ux):
#         W_left, W_right = W_params[offset + idx]
#         objective_terms.append(1.0 * cp.norm(W_left @ blk @ W_right, 'nuc'))
    
#     offset += len(blocks_ux)
#     for idx, blk in enumerate(blocks_xy):
#         W_left, W_right = W_params[offset + idx]
#         objective_terms.append(1.0 * cp.norm(W_left @ blk @ W_right, 'nuc'))

#     problem = cp.Problem(cp.Minimize(sum(objective_terms)), constraints)

#     result_list = N*[None]
#     SLS_data_list = N*[None]

#     def update_reweight_stable_rect(L_val, Wleft_val, Wright_val, delta=1e-2, epsilon=1e-5):
#         U, s, Vt = np.linalg.svd(L_val, full_matrices=False)
#         s = np.maximum(s, epsilon)
#         inv_sqrt_s = 1.0 / np.sqrt(s)

#         Lambda_U = np.diag(inv_sqrt_s)
#         Lambda_V = np.diag(inv_sqrt_s)

#         Wleft_new = U @ Lambda_U @ U.T
#         Wright_new= Vt.T @ Lambda_V @ Vt
#         return Wleft_new, Wright_new


#     for k in range(N):
#         result = problem.solve(solver=cp.MOSEK, 
#                                 mosek_params={
#                                         "MSK_DPAR_INTPNT_TOL_PFEAS": 1e-5,
#                                         "MSK_DPAR_INTPNT_TOL_DFEAS": 1e-5,
#                                         "MSK_DPAR_INTPNT_TOL_REL_GAP": 1e-5,
#                                         "MSK_IPAR_NUM_THREADS": 4,
#                                         "MSK_DPAR_OPTIMIZER_MAX_TIME": 60000,
#                                     },
#                                 verbose=True
#                             )
#         if problem.status != cp.OPTIMAL:
#             raise Exception("Solver did not converge!")
        
#         result_list[k] = result
#         SLS_data_list[k] = copy.deepcopy(SLS_data)

#         try:
#             for idx, (blk, (W_left, W_right)) in enumerate(zip(blocks_uy + blocks_ux + blocks_xy, W_params)):
#                 if isinstance(blk, cp.Expression):
#                     current_blk = blk.value
#                 else:
#                     current_blk = blk
                
#                 if np.allclose(current_blk, 0):
#                     continue
                
#                 W_left_new, W_right_new = update_reweight_stable_rect(
#                     blk=blk,
#                     W_left=W_left,
#                     W_right=W_right
#                 )
                
#                 W_left.value = W_left_new
#                 W_right.value = W_right_new
#         except Exception as e:
#             print(f"update fail: {str(e)}")
#             break

#     SLS_data.calculate_dependent_variables("Reweighted Nuclear Norm")
#     # causal_factorization
#     SLS_data.causal_factorization(rank_eps)
#     SLS_data.F_trunc_to_Phi_trunc()

#     Poly_xu = Poly_x.cart(Poly_u)
#     # example check => truncated constraint
#     diff = Lambda.value@Poly_w.H - Poly_xu.H@SLS_data.Phi_trunc
#     max_err = np.max( np.abs(diff) )
#     print("Error truncated polytope constraint:", max_err)

#     return [result_list, SLS_data, Lambda]


# def swarm_optimize_RTH_offdiag_Phiuy_only(A_list, B_list, C_list, delay, n_uwv, Poly_x, Poly_u, Poly_w, N=10, delta=0.01, rank_eps=1e-7, opt_eps=1e-11):

#     SLS_data = SLSFinite(A_list, B_list, C_list, delay)
#     [constraints, Lambda] = polytope_constraints(SLS_data, Poly_x, Poly_u, Poly_w)

#     blocks_uy = swarm_comm_blockdiag_distributed(SLS_data.Phi_uy, SLS_data.T, n_uwv, 2, 2)
    
#     if blocks_uy is None:
#         print("uy is none\n")

#     W_params = []
#     for blk in blocks_uy:
#         m, n = blk.shape
#         W_left = cp.Parameter((m, m), PSD=True)
#         W_right = cp.Parameter((n, n), PSD=True)
#         W_left.value = delta**(-0.5) * np.eye(m)
#         W_right.value = delta**(-0.5) * np.eye(n)
#         W_params.append( (W_left, W_right) )

#     objective_terms = []
#     for idx, blk in enumerate(blocks_uy):
#         W_left, W_right = W_params[idx]
#         objective_terms.append(1.0 * cp.norm(W_left @ blk @ W_right, 'nuc'))

#     problem = cp.Problem(cp.Minimize(sum(objective_terms)), constraints)

#     result_list = N*[None]
#     SLS_data_list = N*[None]

#     def update_reweight_stable_rect(L_val, Wleft_val, Wright_val, delta=1e-2, epsilon=1e-5):
#         U, s, Vt = np.linalg.svd(L_val, full_matrices=False)
#         s = np.maximum(s, epsilon)
#         inv_sqrt_s = 1.0 / np.sqrt(s)

#         Lambda_U = np.diag(inv_sqrt_s)
#         Lambda_V = np.diag(inv_sqrt_s)

#         Wleft_new = U @ Lambda_U @ U.T
#         Wright_new= Vt.T @ Lambda_V @ Vt
#         return Wleft_new, Wright_new

#     for k in range(N):
#         result = problem.solve(solver=cp.MOSEK, 
#                                 mosek_params={
#                                         "MSK_DPAR_INTPNT_TOL_PFEAS": 1e-5,
#                                         "MSK_DPAR_INTPNT_TOL_DFEAS": 1e-5,
#                                         "MSK_DPAR_INTPNT_TOL_REL_GAP": 1e-5,
#                                         "MSK_IPAR_NUM_THREADS": 4,
#                                         "MSK_DPAR_OPTIMIZER_MAX_TIME": 60000,
#                                     },
#                                 verbose=True
#                             )
#         if problem.status != cp.OPTIMAL:
#             raise Exception("Solver did not converge!")
        
#         result_list[k] = result
#         SLS_data_list[k] = copy.deepcopy(SLS_data)

#         try:
#             for idx, (blk, (W_left, W_right)) in enumerate(zip(blocks_uy, W_params)):
#                 if isinstance(blk, cp.Expression):
#                     current_blk = blk.value
#                 else:
#                     current_blk = blk
                
#                 if np.allclose(current_blk, 0):
#                     continue
                
#                 W_left_new, W_right_new = update_reweight_stable_rect(
#                     L_val=current_blk,
#                     Wleft_val=W_left.value,
#                     Wright_val=W_right.value
#                 )
                
#                 W_left.value = W_left_new
#                 W_right.value = W_right_new
#         except Exception as e:
#             print(f"update fail: {str(e)}")
#             break

#     SLS_data.calculate_dependent_variables("Reweighted Nuclear Norm")
#     SLS_data.causal_factorization(rank_eps)
#     SLS_data.F_trunc_to_Phi_trunc()

#     Poly_xu = Poly_x.cart(Poly_u)
#     diff = Lambda.value@Poly_w.H - Poly_xu.H@SLS_data.Phi_trunc
#     max_err = np.max( np.abs(diff) )
#     print("Error truncated polytope constraint:", max_err)

#     return [result_list, SLS_data, Lambda]




def swarm_optimize_proposed(A_list, B_list, C_list, delay, n_uwv, Poly_x, Poly_u, Poly_w, N=10, delta=0.01, rank_eps=1e-7, opt_eps=1e-11):

    SLS_data = SLSFinite(A_list, B_list, C_list, delay)
    [constraints, Lambda] = polytope_constraints(SLS_data, Poly_x, Poly_u, Poly_w)
    constraints += diagonal_identity_block_constraints_xx(SLS_data.Phi_xx, SLS_data.nx, SLS_data.T + 1)

    blocks_uy = swarm_comm_blockdiag_distributed(SLS_data.Phi_uy, SLS_data.T, n_uwv, 2, 2)
    
    if blocks_uy is None:
        print("uy is none\n")

    W_params = []
    for blk in blocks_uy:
        m, n = blk.shape
        W_left = cp.Parameter((m, m), PSD=True)
        W_right = cp.Parameter((n, n), PSD=True)
        W_left.value = delta**(-0.5) * np.eye(m)
        W_right.value = delta**(-0.5) * np.eye(n)
        W_params.append( (W_left, W_right) )

    objective_terms = []
    for idx, blk in enumerate(blocks_uy):
        W_left, W_right = W_params[idx]
        objective_terms.append(1.0 * cp.norm(W_left @ blk @ W_right, 'nuc'))

    problem = cp.Problem(cp.Minimize(sum(objective_terms)), constraints)

    result_list = N*[None]
    SLS_data_list = N*[None]

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
        
        result_list[k] = result
        SLS_data_list[k] = copy.deepcopy(SLS_data)

        try:
            for idx, (blk, (W_left, W_right)) in enumerate(zip(blocks_uy, W_params)):
                if isinstance(blk, cp.Expression):
                    current_blk = blk.value
                else:
                    current_blk = blk
                
                if np.allclose(current_blk, 0):
                    continue
                
                W_left_new, W_right_new = update_reweight_stable_rect(
                    L_val=current_blk,
                    Wleft_val=W_left.value,
                    Wright_val=W_right.value
                )
                
                W_left.value = W_left_new
                W_right.value = W_right_new
        except Exception as e:
            print(f"update fail: {str(e)}")
            break

    SLS_data.calculate_dependent_variables("Reweighted Nuclear Norm")
    SLS_data.causal_factorization(rank_eps)
    SLS_data.F_trunc_to_Phi_trunc()

    Poly_xu = Poly_x.cart(Poly_u)
    diff = Lambda.value@Poly_w.H - Poly_xu.H@SLS_data.Phi_trunc
    max_err = np.max( np.abs(diff) )
    print("Error truncated polytope constraint:", max_err)

    return [result_list, SLS_data, Lambda]



### Plot and Record ################################################################

def save_sparsity_pattern(matrix, name, folder="simulation_results"):
    plt.figure(figsize=(10,10))
    plt.spy(matrix, markersize=1, precision=1e-5)
    plt.title(f"Sparsity Pattern of {name}")
    os.makedirs(folder, exist_ok=True)
    filename = f"{folder}/{name}.pdf"
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"Saved {filename}")



def plot_swarm_trajectory(SLS_data, Poly_x, Poly_w, center_times, radius_times, T, n_uwv, save_path,
                          process_noise=0.1, meas_noise=0.05, dt=1.0, seed=23):
    if seed is not None:
        np.random.seed(seed)
    
    nx = SLS_data.nx
    nu = SLS_data.nu
    ny = SLS_data.ny

    x0 = np.zeros(nx)
    for i in range(n_uwv):
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
        
        x_next = (
            SLS_data.A_list[t] @ actual_traj[t] 
            + SLS_data.B_list[t] @ u 
            + w
        )
        
        actual_traj[t+1] = x_next
        y_meas[t+1] = SLS_data.C_list[t+1] @ x_next + np.random.normal(0, meas_noise, ny)

    plt.figure(figsize=(10, 8))
    colors = ['#E07197', '#F1BE00', '#FF0000', '#5375E4']
    
    for i in range(n_uwv):
        x = actual_traj[:, 4*i]
        y = actual_traj[:, 4*i+1]
        
        plt.plot(x, y, color=colors[i], 
                 label=f'UWV {i+1}', marker='o',
                 markersize=5, linewidth=2, alpha=0.8)
        
    
        for t in range(T+1):
            center = center_times[t][4*i:4*i+2]
            radius = radius_times[t][4*i:4*i+2]
            
            if t == 0:
                linestyle = '--'
                label = f'Start Box {i+1}'
            elif t == T:
                linestyle = '-'
                label = f'End Box {i+1}'
            else:
                continue
            
            rect = patches.Rectangle(
                (center[0]-radius[0], center[1]-radius[1]),
                2*radius[0], 2*radius[1],
                edgecolor=colors[i], facecolor='none',
                linestyle=linestyle, linewidth=2, alpha=0.3,
                label=label
            )
            plt.gca().add_patch(rect)


    plt.title(f'UWV Trajectories')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join('simulation_results',save_path), 
                   bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()
    print("finished plotting trajectories")



def plot_swarm_trajectory_4sub(SLS_data, center_times, radius_times, T, n_uwv, save_path=None,
                              process_noise=0.05, meas_noise=0.05, seeds=[23, 42, 56, 89]):

    plt.rcParams.update({'font.size': 14})
    colors = ['#E07197', '#F1BE00', '#FF0000', '#5375E4']

    marker_size = 6

    fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    axs = axs.ravel()

    for exp_idx, seed in enumerate(seeds):
        ax = axs[exp_idx]
        np.random.seed(seed)
        
        for spine in ax.spines.values():
            spine.set_linewidth(2.0)

        nx = SLS_data.nx
        x0 = np.zeros(nx)
        for i in range(n_uwv):
            for dim in [0, 1]:
                idx = 4*i + dim
                center = center_times[0][idx]
                radius = radius_times[0][idx]
                x0[idx] = np.random.uniform(center - radius, center + radius)

        actual_traj = np.zeros((T+1, nx))
        y_meas = np.zeros((T+1, SLS_data.ny))
        actual_traj[0] = x0
        y_meas[0] = SLS_data.C_list[0] @ x0 + np.random.normal(0, meas_noise, SLS_data.ny)
        
        for t in range(T):
            F_block = SLS_data.F[t*SLS_data.nu:(t+1)*SLS_data.nu, :(t+1)*SLS_data.ny]
            u = F_block @ y_meas[:t+1].flatten()
            w = np.random.normal(0, process_noise, nx)
            actual_traj[t+1] = SLS_data.A_list[t] @ actual_traj[t] + SLS_data.B_list[t] @ u + w
            y_meas[t+1] = SLS_data.C_list[t+1] @ actual_traj[t+1] + np.random.normal(0, meas_noise, SLS_data.ny)

        for i in range(n_uwv):
            x = actual_traj[:, 4*i]
            y = actual_traj[:, 4*i+1]

            ax.plot(x, y, color=colors[i], linewidth=2.5, alpha=0.9)

            for t_marker in [0, 5, 10]:
                if t_marker <= T:
                    ax.plot(x[t_marker], y[t_marker], marker='o', markersize=12,
                            color=colors[i], markeredgecolor='white', markeredgewidth=2.3, alpha=0.9)


            for t in [0, T]:
                center = center_times[t][4*i:4*i+2]
                radius = radius_times[t][4*i:4*i+2]
                linestyle = '--' if t == 0 else '-'
                
                rect = patches.Rectangle(
                    (center[0]-radius[0], center[1]-radius[1]),
                    2*radius[0], 2*radius[1],
                    edgecolor=colors[i], facecolor='none',
                    linestyle=linestyle, linewidth=3.3, alpha=0.9
                )
                ax.add_patch(rect)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2, linewidth=2.0)

    fig.text(0.5, 0.06, 'X Position', ha='center', fontsize=22)
    fig.text(0.06, 0.5, 'Y Position', va='center', rotation='vertical', fontsize=22)

    legend_elements = [
        Line2D([0], [0], color='k', linestyle='--', lw=2.5, label='Start Box'),
        Line2D([0], [0], color='k', linestyle='-', lw=2.5, label='End Box'),
        *[Line2D([0], [0], color=colors[i], lw=3, label=f'Vehicle {i+1}') for i in range(n_uwv)]
    ]

    legend = fig.legend(
        handles=legend_elements,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.02),
        ncol=3,
        frameon=True,
        edgecolor='black',
        fancybox=False,
        title_fontsize=18,
        fontsize=16
    )

    legend.get_frame().set_linewidth(2.0)
    legend.get_frame().set_edgecolor('black')

    fig.legend(
        handles=legend_elements,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.02),
        ncol=3,
        frameon=True,
        edgecolor='black',
        title_fontsize=18,
        fontsize=16
    )

    plt.subplots_adjust(wspace=0.05, hspace=0.15, bottom=0.12)

    if save_path:
        # os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join('simulation_results', save_path), 
                   bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()
    print("finished plotting trajectories")






def swarm_calculate_communication(F, T, n_uwv, tol=1e-7):
    """
    calculate communication message number for final control matrix F
    """
    state_dim = 2
    Tplus1 = T + 1
    expected_size = n_uwv * Tplus1 * state_dim
    
    if F.shape != (expected_size, expected_size):
        raise ValueError(f"Expected matrix size ({expected_size}, {expected_size}), but now {F.shape}")
    
    mask = np.ones_like(F)
    block_size = n_uwv * state_dim 
    
    for t in range(Tplus1):
        for tau in range(Tplus1):
            row_start = t * block_size
            col_start = tau * block_size
            
            for robot in range(n_uwv):
                r_start = row_start + robot * state_dim
                c_start = col_start + robot * state_dim
                
                mask[r_start : r_start+state_dim, 
                     c_start : c_start+state_dim] = 0.0
    
    F_comm = F * mask
    tot_messag_number = np.linalg.matrix_rank(F_comm, tol=tol)
    print("Cross UWV total message number:", tot_messag_number)


def swarm_communication_message_num(F, T, n_uwv, tol=1e-7):
    Tplus1 = T + 1
    state_dim=2
    expected_size = n_uwv * Tplus1 * state_dim
    
    if F.shape != (expected_size, expected_size):
        raise ValueError(f"size should be({expected_size}, {expected_size}), but is{F.shape}")

    comm_pairs = [(i, j) for i in range(n_uwv) for j in range(n_uwv) if i != j]
    
    comm_blocks = []
    block_size = state_dim * Tplus1
    message_num = 0
    
    for (i, j) in comm_pairs:
        block_matrix = np.zeros((block_size, block_size))
        
        for t in range(Tplus1):
            for tau in range(Tplus1):
                row_start = t * n_uwv * state_dim + i * state_dim
                col_start = tau * n_uwv * state_dim + j * state_dim
                
                sub_block = F[row_start:row_start+state_dim, 
                                col_start:col_start+state_dim]
                
                block_matrix[t*state_dim:(t+1)*state_dim, 
                                tau*state_dim:(tau+1)*state_dim] = sub_block
        
        message_num += np.linalg.matrix_rank(block_matrix, tol=tol)
        comm_blocks.append(block_matrix)
    
    # block_diag_matrix = block_diag(*comm_blocks)
    # message_num = np.linalg.matrix_rank(block_diag_matrix, tol=tol)
    print("Cross UWV total msg num:", message_num)
