import pickle
import numpy as np
import scipy as sp
from SLSFinite import *
from Polytope import *
from functions import *
from utils import *
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib
import math
import time
import sys


### PARAMETER SELECTION #############################################################################################################
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
np.random.seed(1)

T = 10
dt = 1
delay = 0

A_0 = np.block([[np.zeros([2, 2]), np.eye(2)], [np.zeros([2, 2]), np.zeros([2, 2])]])
B_0 = np.block([[np.zeros([2, 2])], [np.eye(2)]])
A = sp.linalg.expm(A_0 * dt)
B = np.sum([np.linalg.matrix_power(A_0 * dt, i) / math.factorial(i + 1) for i in np.arange(100)], axis=0).dot(B_0)
C = np.block([[np.eye(2), np.zeros([2, 2])]])


A_joint = sp.linalg.block_diag(A, A)
B_joint = sp.linalg.block_diag(B, B)
C_joint = sp.linalg.block_diag(C, C)

A_list = (T + 1) * [A_joint]
B_list = (T + 1) * [B_joint]
C_list = (T + 1) * [C_joint]

max_v1 = 2
max_v2 = 4
max_x0 = 1
max_v0 = 0
box_x = 11

comm_dist = 12

center_times_uav1 = (T + 1) * [[0, 0, 0, 0]]
center_times_uav2 = (T + 1) * [[0, 0, 0, 0]]
radius_times_uav1 = (T + 1) * [[box_x, box_x, max_v1, max_v1]]
radius_times_uav2 = (T + 1) * [[box_x, box_x, max_v2, max_v2]]

box_check = [0,10]

center_times_uav1[box_check[0]] = [-5, 4, 0, 0]
radius_times_uav1[box_check[0]] = [max_x0, max_x0, max_v0, max_v0]
center_times_uav1[box_check[1]] = [5, 4, 0, 0]
radius_times_uav1[box_check[1]] = [4, 4.5, 1, 1]

center_times_uav2[box_check[0]] = [-5, -5, 0, 0]
radius_times_uav2[box_check[0]] = [max_x0, max_x0, max_v0, max_v0]
center_times_uav2[box_check[1]] = [5, -4, 0, 0]
radius_times_uav2[box_check[1]] = [1.5, 1.5, 1, 1]

center_times = [center_times_uav1[i] + center_times_uav2[i] for i in range(T + 1)]
radius_times = [radius_times_uav1[i] + radius_times_uav2[i] for i in range(T + 1)]

Poly_x = cart_H_cube(center_times, radius_times)


uav1_x_max = 1.2
uav1_y_max = 0.2
uav2_x_max = 3.5
uav2_y_max = 4.0
Poly_u = H_cube([0, 0, 0, 0],[uav1_x_max, uav1_y_max, uav2_x_max, uav2_y_max]).cartpower(T + 1)

wx_1_scale = 0.25
wy_1_scale = 0.60
wx_2_scale = 0.05
wy_2_scale = 0.05
wxdot_1_scale = 0.05
wydot_1_scale = 0.10
wxdot_2_scale = 0.05
wydot_2_scale = 0.05

v_scale = 0.05
delta = 0.01
RTH_opt_eps = 1e-11
rank_eps = 1e-7
N = 8

Poly_w = H_cube(center_times[0], radius_times[0]).cart(
    H_cube([0, 0, 0, 0, 0, 0, 0, 0], [wx_1_scale,wy_1_scale,wx_2_scale,wy_2_scale,wxdot_1_scale,wydot_1_scale,wxdot_2_scale,wydot_2_scale]).cartpower(T)
).cart(H_cube([0, 0, 0, 0], [v_scale] * 4).cartpower(T + 1))


Poly_comm = partial_time_dist_poly([3,7], T, dist=5)
Poly_x = poly_intersect(Poly_x, Poly_comm)



test_feas = True
test_decentral_feas = False
file = "simulation_results/simulationT10_joint_noisy_agent.pickle"
save = True


if test_feas:
    _ = optimize(A_list, B_list, C_list, delay, Poly_x, Poly_u, Poly_w, RTH_opt_eps)

# Test the feasibility of decentral controllers
if test_decentral_feas:
    _ = optimize_decentral_feasibility(A_list, B_list, C_list, delay, Poly_x, Poly_u, Poly_w, RTH_opt_eps)




### SIMULATION #########################################################################################################


key = 'Reweighted Nuclear Norm'
start = time.time()
optimize_RTH_output = optimize_RTH(A_list, B_list, C_list, delay, Poly_x, Poly_u, Poly_w, N, delta, rank_eps, RTH_opt_eps)
t1 = time.time() - start
optimize_RTH_output.append(t1)


print("begin offdiag three phi optimization")
key = 'Proposed Offdiag Result'
start = time.time()
optimize_offdiag_three_output = optimize_RTH_proposed_lower_triangular(A_list, B_list, C_list, delay, Poly_x, Poly_u, Poly_w, N=8, delta=0.01, rank_eps=1e-7, opt_eps=1e-8)
t2 = time.time() - start
optimize_offdiag_three_output.append(t2)


data = {
'Reweighted Nuclear Norm': optimize_RTH_output,
'Proposed Offdiag Result': optimize_offdiag_three_output,
}


with open(file,"wb") as f:
    pickle.dump(data, f)
    
simulation_data = pickle.load(open(file, "rb"))
optimize_RTH_data = simulation_data['Reweighted Nuclear Norm']
proposed_data = simulation_data['Proposed Offdiag Result']









### Print Results ############################################################################################

SLS_nuc_list= optimize_RTH_data[1]
SLS_nuc = SLS_nuc_list[-1]
Lambda_nuc = optimize_RTH_data[2]
Poly_xu = Poly_x.cart(Poly_u)
print()
print("--- Baseline Communication Messages ----------------------------------------------------")
print("max |K - K_trunc|:", np.max( np.abs(SLS_nuc.F - SLS_nuc.F_trunc) ) )
print("max |Phi - Phi_trunc|:", np.max( np.abs(SLS_nuc.Phi_matrix.value - SLS_nuc.Phi_trunc) ) )
print("Error true polytope constraint:", np.max(np.abs( Lambda_nuc.value.dot(Poly_w.H) - Poly_xu.H.dot(SLS_nuc.Phi_matrix.value))))
print("Error truncated polytope constraint:", np.max( np.abs(Lambda_nuc.value.dot(Poly_w.H) - Poly_xu.H.dot(SLS_nuc.Phi_trunc)) ) )
SLS_nuc.calculate_dependent_variables(key="Reweighted Nuclear Norm")
msg_21, msg_12 = SLS_nuc.compute_communication_messages(rank_eps=1e-7)
print("UAV2->UAV1 messages:", msg_21)
print("UAV1->UAV2 messages:", msg_12)
SLS_nuc.display_message_time(save_file="baseline_results.npz")




SLS_proposed = proposed_data[1]
print()
print("--- Proposed Controller Communication Messages ------------------------------------")
SLS_proposed.calculate_dependent_variables("Reweighted Nuclear Norm")
SLS_proposed.causal_factorization(rank_eps=1e-7)
msg_21_proposed, msg_12_proposed = SLS_proposed.compute_communication_messages(rank_eps=1e-7)
print("UAV2->UAV1 messages:", msg_21_proposed)
print("UAV1->UAV2 messages:", msg_12_proposed)
plot_two_noisy(SLS_proposed, Poly_x, Poly_w, center_times, radius_times, T, "simulation_results/noisy_proposed_trajectories.pdf")
SLS_proposed.display_message_time(save_file="SLS_proposed_results.npz")




def run():
    t = ""
    with open('simulation.py') as f:
        t = f.read()
    return t

