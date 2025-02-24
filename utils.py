import numpy as np
import cvxpy as cp
import matplotlib
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
import time
import math

def row_factorization_causal(L, rank_eps=1e-7):
    m, n = L.shape
    rank_L = np.linalg.matrix_rank(L, tol=rank_eps)

    E = np.zeros((0, n))
    D = np.zeros((0, 0))

    rank_counter = 0
    rank_increase_times = [None]*rank_L

    for time in range(m):
        submat_new = L[:time+1, :]
        rank_new = np.linalg.matrix_rank(submat_new, tol=rank_eps)

        if rank_new - rank_counter == 1:
            rank_increase_times[rank_counter] = time
            rank_counter += 1

            E = np.vstack([E, L[time:time+1, :]])
            if D.shape[1] < rank_counter:
                D = np.hstack([D, np.zeros((D.shape[0], 1))])

            unit = np.zeros((1, rank_counter))
            unit[0, -1] = 1.0
            D = np.vstack([D, unit])

        elif rank_new == rank_counter:
            c, _, _, _ = np.linalg.lstsq(E.T, L[time, :], rcond=None)
            c = c.reshape(1, rank_counter)
            D = np.vstack([D, c])

        else:
            raise ValueError(f"Rank increased by more than 1 at time={time}.")

    assert E.shape[0] == rank_counter
    assert D.shape[0] == m
    assert D.shape[1] == rank_counter
    assert rank_counter == rank_L, f"Got rank_counter={rank_counter}, but rank_L={rank_L}."

    return D, E, rank_increase_times



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
