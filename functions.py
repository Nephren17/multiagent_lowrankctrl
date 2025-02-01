import numpy as np
import cvxpy as cp
from SLSFinite import *
from Polytope import *
import matplotlib
import matplotlib.pyplot as plt
import copy

def polytope_constraints(SLS_data, Poly_x, Poly_u, Poly_w):
    # load SLS constraints
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

def optimize(A_list, B_list, C_list, Poly_x, Poly_u, Poly_w, opt_eps, norm=None):
    """
    Parameters
    ----------
    A_list: list of matrices [A_0, ...A_T]
    B_list: list of tensors [B_0, ...B_T]
    C_list: list of tensors [C_0, ...C_T]
        where A_t, B_t, C_t are the matrices in the dynamics of the system at time t.
    Poly_x, Poly_u, Poly_w: Polytope
        Polytopes cartesian products of polytopes
        for x, u and w over all times t = 0,...,T.

    Returns
    -------
    result: float
        Optimal cost (np.inf if problem is not feasible).
    SLS_data: SLSFinite object
        Instance of the class SLSFinite containing the variables corresponding
        to the optimal cost.
    Lambda: cvxpy.Variable, shape (H_x[0], H_w[0])
        Polytope containment variable corresponding to the optimal cost.
    """
    # constraints
    SLS_data = SLSFinite(A_list, B_list, C_list, norm)
    [constraints, Lambda] = polytope_constraints(SLS_data, Poly_x, Poly_u, Poly_w)
    # objective function
    objective = cp.Minimize(0)
    problem = cp.Problem(objective, constraints)
    result = problem.solve( solver=cp.MOSEK,
                            mosek_params = {'MSK_DPAR_INTPNT_CO_TOL_DFEAS':  opt_eps,
                                            },
                            verbose=True)
    if problem.status != cp.OPTIMAL:
        raise Exception("Solver did not converge!")
    return result, SLS_data, Lambda

def optimize_RTH(A_list, B_list, C_list, Poly_x, Poly_u, Poly_w, N, delta, rank_eps, opt_eps):
    """
    Parameters
    ----------
    A_list: list of matrices [A_0, ...A_T]
    B_list: list of tensors [B_0, ...B_T]
    C_list: list of tensors [C_0, ...C_T]
        where A_t, B_t, C_t are the matrices in the dynamics of the system at time t.
    Poly_x, Poly_u, Poly_w: Polytope
        Polytopes cartesian products of polytopes
        for x, u and w over all times t = 0,...,T.
    N: int
        Number of iterations of the reweighted nuclear norm iteration.
    key: str
        Set the cost function.

    Returns
    -------
    result_list: float
        Optimal cost (np.inf if problem is not feasible).
    SLS_data_list: SLSFinite object
        List of Instance of the class SLSFinite containing the variables corresponding
        to the optimal cost.
    Lambda: cvxpy.Variable, shape (H_x[0] + H_u[0], H_w[0])
        Polytope containment variable corresponding to the optimal cost.
    """
    SLS_data = SLSFinite(A_list, B_list, C_list)
    [constraints, Lambda] = polytope_constraints(SLS_data, Poly_x, Poly_u, Poly_w)
    # constraints += diagonal_identity_block_constraints_xx(SLS_data.Phi_xx, SLS_data.nx, SLS_data.T + 1)
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
                               mosek_params = {'MSK_DPAR_INTPNT_CO_TOL_DFEAS':  opt_eps,
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
        
    # calculate F
    SLS_data_list[-1].calculate_dependent_variables("Reweighted Nuclear Norm")
    # compute causal factorization of F and truncate using rank_eps
    SLS_data_list[-1].causal_factorization(rank_eps)
    # compute truncated Phi matrix for checking feasibility
    SLS_data_list[-1].F_trunc_to_Phi_trunc()

    # check feasibility up to 1e-6
    Poly_xu = Poly_x.cart(Poly_u)
    print("rank F:", SLS_data_list[-1].rank_F_trunc)
    print("band (D,E) = messages:", SLS_data_list[-1].E.shape[0])
    print("Error true F and truncated F:", np.max( np.abs(SLS_data_list[-1].F - SLS_data_list[-1].F_trunc) ) )
    print("Error true Phi and truncated Phi:", np.max( np.abs(SLS_data_list[-1].Phi_matrix.value - SLS_data_list[-1].Phi_trunc) ) )
    print("Error truncated polytope constraint:", np.max( np.abs(Lambda.value.dot(Poly_w.H) - Poly_xu.H.dot(SLS_data_list[-1].Phi_trunc)) ) )
    assert np.all( Lambda.value.dot(Poly_w.h) <= Poly_xu.h + 1e-6 )
    assert np.all( np.isclose( Lambda.value.dot(Poly_w.H), (Poly_xu.H.dot(SLS_data_list[-1].Phi_trunc)).astype('float') , atol = 1e-5) )

    # check feasibility by reoptimizing over Lambda
    # Poly_xu = Poly_x.cart(Poly_u)
    # reopt_Lambda = cp.Variable((np.shape(Poly_xu.H)[0], np.shape(Poly_w.H)[0]), nonneg=True)
    # reopt_constraints = [reopt_Lambda @ Poly_w.H == Poly_xu.H @ SLS_data_list[-1].Phi_trunc,
    #                     reopt_Lambda @ Poly_w.h <= Poly_xu.h]
    # reopt_objective = cp.Minimize(0)
    # reopt_problem = cp.Problem(reopt_objective, reopt_constraints)
    # _ = reopt_problem.solve( solver=cp.MOSEK,
    #                         mosek_params = {'MSK_DPAR_INTPNT_CO_TOL_DFEAS':  opt_eps,
    #                                         },
    #                         verbose=True)
    # if reopt_problem.status != cp.OPTIMAL:
    #     raise Exception("Solver did not converge!")

    return [result_list, SLS_data_list, Lambda]#, reopt_Lambda]

def optimize_reweighted_atomic(A_list, B_list, C_list, Poly_x, Poly_u, Poly_w, N, key, delta, opt_eps):
    SLS_data = SLSFinite(A_list, B_list, C_list)
    [constraints, Lambda] = polytope_constraints(SLS_data, Poly_x, Poly_u, Poly_w)
    # Initialize Paramters
    if key == 'Reweighted Sensor Norm':
        W = cp.Parameter(SLS_data.ny*(SLS_data.T+1))
        objective = cp.Minimize(cp.sum(cp.norm(SLS_data.Phi_uy @ cp.diag(W), 2, 0)))
        W.value = delta**-1 * np.ones(SLS_data.ny*(SLS_data.T+1))
    elif key == 'Reweighted Actuator Norm':
        W = cp.Parameter(SLS_data.nu*(SLS_data.T+1))
        objective = cp.Minimize(cp.sum(cp.norm(cp.diag(W) @ SLS_data.Phi_uy, 2, 1)))
        W.value = delta**-1 * np.ones(SLS_data.nu*(SLS_data.T+1))
    
    result_list = N*[None]
    SLS_list = N*[None]
    norm_list = N*[None]
    #define problem
    problem = cp.Problem(objective, constraints)
    for k in range(N):
        result = problem.solve(solver=cp.MOSEK,
                               mosek_params = {'MSK_DPAR_INTPNT_CO_TOL_DFEAS':  opt_eps,
                                            },
                               verbose=True)
        if problem.status != cp.OPTIMAL:
            raise Exception("Solver did not converge!")
        result_list[k] = result
        SLS_list[k] = copy.deepcopy(SLS_data)
        
        #update params
        if key == 'Reweighted Sensor Norm':
            norm_list[k] = np.linalg.norm(SLS_data.Phi_uy.value, 2, 0)
            W.value = (np.linalg.norm(SLS_data.Phi_uy.value, 2, 0) + delta)**-1
        if key == 'Reweighted Actuator Norm':
            norm_list[k] = np.linalg.norm(SLS_data.Phi_uy.value, 2, 1)
            W.value = (np.linalg.norm(SLS_data.Phi_uy.value, 2, 1) + delta)**-1
    return result_list, SLS_list, Lambda, norm_list


def optimize_sparsity(A_list, B_list, C_list, Poly_x, Poly_u, Poly_w, key, N, delta, rank_eps, opt_eps):
    """
    Parameters
    ----------
    A_list: list of matrices [A_0, ...A_T]
    B_list: list of tensors [B_0, ...B_T]
    C_list: list of tensors [C_0, ...C_T]
        where A_t, B_t, C_t are the matrices in the dynamics of the system at time t.
    Poly_x, Poly_u, Poly_w: Polytope
        Polytopes cartesian products of polytopes
        for x, u and w over all times t = 0,...,T.
    N: int
        Number of iterations of the reweighted nuclear norm iteration.
    key: str
        Set the cost function.

    Returns
    -------
    result_list: float
        Optimal cost (np.inf if problem is not feasible).
    SLS_data_list: SLSFinite object
        List of Instance of the class SLSFinite containing the variables corresponding
        to the optimal cost.
    Lambda: cvxpy.Variable, shape (H_x[0] + H_u[0], H_w[0])
        Polytope containment variable corresponding to the optimal cost.
    """
    result_list, SLS_list, Lambda, norm_list  = optimize_reweighted_atomic(A_list, B_list, C_list, Poly_x, Poly_u, Poly_w, N, key, delta, opt_eps)
    # reoptimize over last iteration by removing columns or rows of the lowest 2-norms
    argmin = SLS_list[-1].Phi_uy.value
    if key == 'Reweighted Sensor Norm':
        # 2-norms of column vectors = sensor norm
        norm_argmin = np.linalg.norm(argmin, 2, 0)
    elif key == 'Reweighted Actuator Norm':
        # 2-norms of row vectors = actuator norm
        norm_argmin = np.linalg.norm(argmin, 2, 1)
    else:
        raise Exception('Choose either the reweighted sensor or actuator norm to minimize!')
    ind = np.where(norm_argmin<=rank_eps)[0]
    reopt_result, reopt_SLS, reopt_Lambda = optimize(A_list, B_list, C_list, Poly_x, Poly_u, Poly_w, opt_eps, norm=[key, ind])
    reopt_kept_indices = [i for i in np.arange(len(norm_argmin)) if i not in ind] # columns/rows kept
    reopt_SLS.calculate_dependent_variables(key) # only get F, no other truncation needed.
    reopt_SLS.F_trunc = reopt_SLS.F
    reopt_SLS.F_trunc_to_Phi_trunc()

    # check feasibility after truncating lower triangularity
    Poly_xu = Poly_x.cart(Poly_u)
    print("Error true F and truncated F:", np.max( np.abs(reopt_SLS.F - reopt_SLS.F_trunc) ) )
    print("Error true Phi and truncated Phi:", np.max( np.abs(reopt_SLS.Phi_matrix.value - reopt_SLS.Phi_trunc) ) )
    print("Error truncated polytope constraint:", np.max( np.abs(reopt_Lambda.value.dot(Poly_w.H) - Poly_xu.H.dot(reopt_SLS.Phi_trunc)) ) )
    assert np.all( reopt_Lambda.value.dot(Poly_w.h) <= Poly_xu.h + 1e-6 )
    assert np.all( np.isclose( reopt_Lambda.value.dot(Poly_w.H), (Poly_xu.H.dot(reopt_SLS.Phi_trunc)).astype('float') , atol = 1e-6) )

    return [reopt_result, reopt_SLS, reopt_Lambda, norm_list, reopt_kept_indices, SLS_list]




def optimize_RTH_offdiag(A_list, B_list, C_list, Poly_x, Poly_u, Poly_w, N=10, delta=0.01, rank_eps=1e-7, opt_eps=1e-11):
    """
    minimize rank(L_1) + rank(L_2),
    """
    # poly constraints
    SLS_data = SLSFinite(A_list, B_list, C_list)

    [constraints, Lambda] = polytope_constraints(SLS_data, Poly_x, Poly_u, Poly_w)
    # constraints += diagonal_identity_block_constraints_xx(SLS_data.Phi_xx, SLS_data.nx, SLS_data.T + 1)

    L1_expr = SLS_data.extract_offdiag_expr(direction='21')
    L2_expr = SLS_data.extract_offdiag_expr(direction='12')
    # F1_expr = SLS_data.extract_diag_expr(direction='1')
    # F2_expr = SLS_data.extract_diag_expr(direction='2')


    dim = 2*(SLS_data.T+1)
    W1_left = cp.Parameter((dim, dim), PSD=True)
    W1_right= cp.Parameter((dim, dim), PSD=True)
    W2_left = cp.Parameter((dim, dim), PSD=True)
    W2_right= cp.Parameter((dim, dim), PSD=True)
    # W3_left = cp.Parameter((dim, dim), PSD=True)
    # W3_right= cp.Parameter((dim, dim), PSD=True)
    # W4_left = cp.Parameter((dim, dim), PSD=True)
    # W4_right= cp.Parameter((dim, dim), PSD=True)

    W1_left.value  = delta**(-0.5) * np.eye(dim)
    W1_right.value = delta**(-0.5) * np.eye(dim)
    W2_left.value  = delta**(-0.5) * np.eye(dim)
    W2_right.value = delta**(-0.5) * np.eye(dim)
    # W3_left.value  = delta**(-0.5) * np.eye(dim)
    # W3_right.value = delta**(-0.5) * np.eye(dim)
    # W4_left.value  = delta**(-0.5) * np.eye(dim)
    # W4_right.value = delta**(-0.5) * np.eye(dim)

    objective = cp.Minimize(cp.norm( W1_left@L1_expr@W1_right, 'nuc') + cp.norm( W2_left@L2_expr@W2_right, 'nuc'))
    # objective = cp.Minimize(1.0 * cp.norm( W1_left@L1_expr@W1_right, 'nuc') + 1.0 * cp.norm( W2_left@L2_expr@W2_right, 'nuc')
    # + 1.0 * cp.norm( W3_left@F1_expr@W3_right, 'nuc') + 1.0 * cp.norm( W4_left@F2_expr@W4_right, 'nuc'))
    problem = cp.Problem(objective, constraints)

    result_list = []
    SLS_data_list = []

    def update_reweight(L_val, Wleft_val, Wright_val):
        left_inv  = np.linalg.inv(Wleft_val)
        right_inv = np.linalg.inv(Wright_val)
        Y = left_inv @ L_val @ right_inv
        Y_reg = Y + delta*np.eye(Y.shape[0])
        eigvals, eigvecs = np.linalg.eigh(Y_reg)
        assert np.all(eigvals>0), "reweighting: negative eigenvalue found!"
        W_new = eigvecs @ np.diag(eigvals**(-0.5)) @ eigvecs.T
        return W_new, W_new

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
            print(f"⚠️ WARNING: Negative eigenvalue found ({min_eig:.6f}), shifting by {shift:.6f}")
            Y = Y + shift * np.eye(Y.shape[0])
            eigvals, eigvecs = np.linalg.eigh(Y)

        eigvals = np.maximum(eigvals, epsilon)

        W_new = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
        return W_new, W_new



    for k in range(N):
        result = problem.solve(solver=cp.MOSEK, mosek_params={'MSK_DPAR_INTPNT_CO_TOL_DFEAS':opt_eps}, verbose=True)
        if problem.status != cp.OPTIMAL:
            raise Exception("Solver did not converge!")
        
        result_list.append(result)
        SLS_data_list.append(copy.deepcopy(SLS_data))

        L1_val = L1_expr.value
        L2_val = L2_expr.value
        # F1_val = F1_expr.value
        # F2_val = F2_expr.value

        W1_left.value, W1_right.value = update_reweight(L1_val, W1_left.value, W1_right.value)
        W2_left.value, W2_right.value = update_reweight(L2_val, W2_left.value, W2_right.value)
        # W3_left.value, W3_right.value = update_reweight_stable(F1_val, W3_left.value, W3_right.value)
        # W4_left.value, W4_right.value = update_reweight_stable(F2_val, W4_left.value, W4_right.value)


    SLS_data.calculate_dependent_variables("Reweighted Nuclear Norm")
    # causal_factorization
    SLS_data.causal_factorization(rank_eps)
    SLS_data.F_trunc_to_Phi_trunc()

    Poly_xu = Poly_x.cart(Poly_u)
    # example check => truncated constraint
    diff = Lambda.value@Poly_w.H - Poly_xu.H@SLS_data.Phi_trunc
    max_err = np.max( np.abs(diff) )
    print("Error truncated polytope constraint:", max_err)

    return [result_list, SLS_data, Lambda]



def polytope_constraints_diagI(SLS_data, Poly_x, Poly_u, Poly_w):
    constraints = SLS_data.SLP_constraints()
    Poly_xu = Poly_x.cart(Poly_u)
    Lambda = cp.Variable((Poly_xu.H.shape[0], Poly_w.H.shape[0]), nonneg=True)
    constraints += [Lambda @ Poly_w.H == Poly_xu.H @ SLS_data.Phi_matrix,
                    Lambda @ Poly_w.h <= Poly_xu.h]

    Tplus1 = SLS_data.T + 1
    nx = SLS_data.nx
    nu = SLS_data.nu
    ny = SLS_data.ny
    constraints += diagonal_identity_block_constraints_xx(SLS_data.Phi_xx, nx, Tplus1)
    constraints += diagonal_identity_block_constraints_xy(SLS_data.Phi_xy, nx, ny, Tplus1)
    constraints += diagonal_identity_block_constraints_ux(SLS_data.Phi_ux, nx, nu, Tplus1)

    return constraints, Lambda



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



def optimize_no_comm_both_ways(A_list, B_list, C_list, Poly_x, Poly_u, Poly_w, opt_eps):
    """
    No cross-communication between UAVs,
    objective = 0 for demonstration.
    """
    SLS_data = SLSFinite(A_list, B_list, C_list)

    [constraints, Lambda] = polytope_constraints_no_comm_both_ways(
        SLS_data, Poly_x, Poly_u, Poly_w
    )

    objective = cp.Minimize(0)
    problem = cp.Problem(objective, constraints)

    result = problem.solve(
        solver=cp.MOSEK,
        mosek_params={'MSK_DPAR_INTPNT_CO_TOL_DFEAS': opt_eps},
        verbose=True
    )
    if problem.status != cp.OPTIMAL:
        raise Exception("Solver did not converge or feasible solution not found.")

    return [result, SLS_data, Lambda]




# Optimize without constraint on phi_ux
'''
The constraint on phi_ux is removed. Optimize Off-diag of phi_uy - phi_ux
'''
def optimize_RTH_offdiag_diagI(A_list, B_list, C_list, Poly_x, Poly_u, Poly_w, N=10, delta=0.01, rank_eps=1e-7, opt_eps=1e-11):
    SLS_data = SLSFinite(A_list, B_list, C_list)
    [constraints, Lambda] = polytope_constraints_diagI(SLS_data, Poly_x, Poly_u, Poly_w)

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

    # revised object function
    objective = cp.Minimize(cp.norm( W1_left@L1_expr@W1_right, 'nuc') + cp.norm( W2_left@L2_expr@W2_right, 'nuc'))
    # objective = cp.Minimize(cp.norm( W1_left@L1_expr@W1_right, 'nuc'))
    # objective = cp.Minimize( cp.norm(L1_expr, 2) + cp.norm(L2_expr, 2) )
    # objective = cp.Minimize( cp.norm(L1_expr, 2) )

    problem = cp.Problem(objective, constraints)

    result_list = []
    SLS_data_list = []

    def update_reweight(L_val, Wleft_val, Wright_val):
        left_inv  = np.linalg.inv(Wleft_val)
        right_inv = np.linalg.inv(Wright_val)
        Y = left_inv @ L_val @ right_inv
        Y_reg = Y + delta*np.eye(Y.shape[0])
        eigvals, eigvecs = np.linalg.eigh(Y_reg)
        assert np.all(eigvals>0), "reweighting: negative eigenvalue found!"
        W_new = eigvecs @ np.diag(eigvals**(-0.5)) @ eigvecs.T
        return W_new, W_new

    for k in range(N):
        result = problem.solve(solver=cp.MOSEK, mosek_params={'MSK_DPAR_INTPNT_CO_TOL_DFEAS':opt_eps}, verbose=True)
        if problem.status != cp.OPTIMAL:
            raise Exception("Solver did not converge!")
        
        result_list.append(result)
        SLS_data_list.append(copy.deepcopy(SLS_data))

        L1_val = L1_expr.value
        L2_val = L2_expr.value
        W1_left.value, W1_right.value = update_reweight(L1_val, W1_left.value, W1_right.value)
        W2_left.value, W2_right.value = update_reweight(L2_val, W2_left.value, W2_right.value)


    SLS_data.calculate_dependent_variables("Reweighted Nuclear Norm")
    # causal_factorization
    SLS_data.causal_factorization(rank_eps)
    SLS_data.F_trunc_to_Phi_trunc()

    Poly_xu = Poly_x.cart(Poly_u)
    # example check => truncated constraint
    diff = Lambda.value@Poly_w.H - Poly_xu.H@SLS_data.Phi_trunc
    max_err = np.max( np.abs(diff) )
    print("Error truncated polytope constraint:", max_err)
    return [result_list, SLS_data, Lambda]



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


    fig, axes = plt.subplots(2, 3, figsize=(14,8))

    prec = 1e-7
    mk_size = 1.0

    ax_xx = axes[0,0]
    ax_xy = axes[0,1]
    ax_ux = axes[0,2]

    ax_uy = axes[1,0]
    ax_F  = axes[1,1]

    # Phi_xx
    ax_xx.spy(phi_xx_val, precision=prec, markersize=mk_size, aspect='auto')
    ax_xx.set_title("Phi_xx")

    # Phi_xy
    ax_xy.spy(phi_xy_val, precision=prec, markersize=mk_size, aspect='auto')
    ax_xy.set_title("Phi_xy")

    # Phi_ux
    ax_ux.spy(phi_ux_val, precision=prec, markersize=mk_size, aspect='auto')
    ax_ux.set_title("Phi_ux")

    # Phi_uy
    ax_uy.spy(phi_uy_val, precision=prec, markersize=mk_size, aspect='auto')
    ax_uy.set_title("Phi_uy")

    # F
    ax_F.spy(F_val, precision=prec, markersize=mk_size, aspect='auto')
    ax_F.set_title("F (closed-loop)")

    axes[1,2].axis("off")
    axes[1,2].set_title("")

    fig.suptitle("Phi and F Matrices Visualization", fontsize=14)
    fig.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()



def partial_time_dist_poly(times_list, T, dist=5):
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
