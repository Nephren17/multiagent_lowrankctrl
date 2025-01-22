import numpy as np
import cvxpy as cp
from SLSFinite import *
from Polytope import *
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
    assert np.all( np.isclose( Lambda.value.dot(Poly_w.H), (Poly_xu.H.dot(SLS_data_list[-1].Phi_trunc)).astype('float') , atol = 1e-6) )

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


def poly_intersect(poly1, poly2):
    """
    Return the intersection of two polytopes in H-rep.
    """
    import numpy as np
    H_new = np.vstack((poly1.H, poly2.H))
    h_new = np.concatenate((poly1.h, poly2.h))
    return Polytope(H_new, h_new)