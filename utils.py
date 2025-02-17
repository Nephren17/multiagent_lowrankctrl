import numpy as np
import cvxpy as cp
import matplotlib
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
import time
import math

def row_factorization(M, rank_eps=1e-7):
    m, n = M.shape
    rank_M = np.linalg.matrix_rank(M, tol=rank_eps)

    E = np.zeros((0, n))
    D = np.zeros((0, 0))

    rank_counter = 0
    rank_increase_rows = [None]*rank_M

    for row in range(m):
        submat_new = M[:row+1, :]
        rank_new = np.linalg.matrix_rank(submat_new, tol=rank_eps)

        if rank_new - rank_counter == 1:
            rank_increase_rows[rank_counter] = row
            rank_counter += 1

            E = np.vstack([E, submat_new[row:row+1, :]]) 
            if D.shape[1] < rank_counter:
                D = np.hstack([D, np.zeros((D.shape[0], 1))])

            unit = np.zeros((1, rank_counter))
            unit[0, -1] = 1.0
            D = np.vstack([D, unit])

        elif rank_new == rank_counter:
            c, _, _, _ = np.linalg.lstsq(E.T, M[row, :], rcond=None)
            c = c.reshape(1, rank_counter)  # (1, rank_counter)

            D = np.vstack([D, c])

        else:
            raise ValueError("Rank increased by more than 1 at row {}.".format(row))

        # assert E.shape == (rank_counter, n), "E shape mismatch"
        # assert D.shape == (row+1, rank_counter), "D shape mismatch"

    assert E.shape == (rank_counter, n)
    assert D.shape == (m, rank_counter)
    assert rank_counter == rank_M

    return D, E, rank_increase_rows



def compute_message_time_from_E(E, eps=1e-7):
    r, n = E.shape
    times = []

    for i in range(r):
        row_i = E[i, :]
        nonzero_indices = np.where(np.abs(row_i) > eps)[0]

        if len(nonzero_indices) == 0:
            time_i = -1
        else:
            c_i = nonzero_indices[-1]
            time_i = math.ceil((c_i+1)/2) - 1

        times.append(time_i)
    return times
