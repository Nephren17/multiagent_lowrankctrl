import pickle
import numpy as np
import scipy as sp
from SLSFinite import *
from Polytope import *
from functions import *
from functions_swarm import *
from utils import *
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib
import math
import time
import sys
import os
from datetime import datetime

### PARAMETER SELECTION #############################################################################
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
np.random.seed(1)

T = 5
dt = 1
delay = 0

A_0 = np.block([[np.zeros([2, 2]), np.eye(2)], [np.zeros([2, 2]), np.zeros([2, 2])]])
B_0 = np.block([[np.zeros([2, 2])], [np.eye(2)]])
A = sp.linalg.expm(A_0 * dt)
B = np.sum([np.linalg.matrix_power(A_0 * dt, i) / math.factorial(i + 1) for i in np.arange(100)], axis=0).dot(B_0)

n_uwv = 4
A_joint = sp.linalg.block_diag(*(A for _ in range(n_uwv)))
B_joint = sp.linalg.block_diag(*(B for _ in range(n_uwv)))

# if every UWV knows its own position
C_joint = np.zeros((8, 16)) 
for i in range(n_uwv):
    C_joint[2*i:2*i+2, 4*i:4*i+2] = np.eye(2)


# Special information case: UWV1 knows x1, x2, UWV2 knows y2, y3, UWV3 knows x3, x4, UWV4 knows y4, y1
C_test = np.zeros((8, 16))

C_test[0, 0] = 1.0
C_test[1, 4] = 1.0 

C_test[2, 5] = 1.0
C_test[3, 9] = 1.0

C_test[4, 8]  = 1.0
C_test[5, 12] = 1.0

C_test[6, 13] = 1.0
C_test[7, 1]  = 1.0

C_joint = C_test


A_list = (T+1)*[A_joint]
B_list = (T+1)*[B_joint]
C_list = (T+1)*[C_joint]


max_v = 2.0
max_x0 = 1.0 
max_xe = 1.5
max_v0 = 0.0
max_ve = 1.0
box_x = 6.0   
box_check = [0, 5]

center_uav1 = (T+1)*[[0,0,0,0]]
radius_uav1 = (T+1)*[[box_x,box_x,max_v,max_v]]
center_uav1[box_check[0]] = [-3, -3, 0, 0]
radius_uav1[box_check[0]] = [max_x0, max_x0, max_v0, max_v0]
center_uav1[box_check[1]] = [3, -3, 0, 0]
radius_uav1[box_check[1]] = [max_xe, max_xe, max_ve, max_ve]

center_uav2 = (T+1)*[[0,0,0,0]]
radius_uav2 = (T+1)*[[box_x,box_x,max_v,max_v]]
center_uav2[box_check[0]] = [-3, 3, 0, 0]
radius_uav2[box_check[0]] = [max_x0, max_x0, max_v0, max_v0]
center_uav2[box_check[1]] = [-3, -3, 0, 0] 
radius_uav2[box_check[1]] = [max_xe, max_xe, max_ve, max_ve]

center_uav3 = (T+1)*[[0,0,0,0]]
radius_uav3 = (T+1)*[[box_x,box_x,max_v,max_v]]
center_uav3[box_check[0]] = [3, 3, 0, 0]
radius_uav3[box_check[0]] = [max_x0, max_x0, max_v0, max_v0]
center_uav3[box_check[1]] = [-3, 3, 0, 0]
radius_uav3[box_check[1]] = [max_xe, max_xe, max_ve, max_ve]

center_uav4 = (T+1)*[[0,0,0,0]]
radius_uav4 = (T+1)*[[box_x,box_x,max_v,max_v]]
center_uav4[box_check[0]] = [3, -3, 0, 0]
radius_uav4[box_check[0]] = [max_x0, max_x0, max_v0, max_v0]
center_uav4[box_check[1]] = [3, 3, 0, 0]
radius_uav4[box_check[1]] = [max_xe, max_xe, max_ve, max_ve]

center_times = []
radius_times = []
for t in range(T+1):
    merged_center = (
        center_uav1[t] + 
        center_uav2[t] + 
        center_uav3[t] + 
        center_uav4[t]
    )
    merged_radius = (
        radius_uav1[t] + 
        radius_uav2[t] + 
        radius_uav3[t] + 
        radius_uav4[t]
    )
    center_times.append(merged_center)
    radius_times.append(merged_radius)





Poly_x = cart_H_cube(center_times, radius_times)





u_max = 2.0
Poly_u = H_cube([0]*8, [u_max]*8).cartpower(T+1)

pos_noise = 0.00
vel_noise = 0.05
ctrl_noise = 0.05
RTH_opt_eps = 1e-11
delta = 0.01
sparse_opt_eps = 1e-10
rank_eps = 1e-7
N = 8

w_scale = [pos_noise, pos_noise, vel_noise, vel_noise]*n_uwv
v_scale = [ctrl_noise]*8

Poly_w = H_cube(center_times[0], radius_times[0]).cart(
    H_cube([0]*(16), w_scale).cartpower(T)
).cart(H_cube([0]*8, v_scale).cartpower(T+1))



# commmunication distrance contstaint
comm_dist = 10.0
H_comm_12 = np.array([
    [1, 1, 0, 0, -1, -1, 0, 0,  0,0,0,0, 0,0,0,0],   #  |x1-x2| + |y1-y2| <= 10
    [1,-1, 0, 0, -1, 1, 0, 0,   0,0,0,0, 0,0,0,0],
    [-1,1, 0, 0, 1,-1, 0, 0,    0,0,0,0, 0,0,0,0],
    [-1,-1,0, 0, 1,1, 0,0,      0,0,0,0, 0,0,0,0],
])
h_comm_12 = comm_dist * np.ones(4)
Poly_comm_12 = Polytope(H_comm_12, h_comm_12)

# 2-3
H_comm_23 = np.zeros((4,16))
H_comm_23[:,4]  = [ 1,  1, -1, -1]  # x2
H_comm_23[:,5]  = [ 1, -1,  1, -1]  # y2
H_comm_23[:,8]  = [-1, -1,  1,  1]  # x3
H_comm_23[:,9]  = [-1,  1, -1,  1]  # y3
h_comm_23 = comm_dist * np.ones(4)
Poly_comm_23 = Polytope(H_comm_23, h_comm_23)

# 3-4
H_comm_34 = np.zeros((4,16))
H_comm_34[:,8]  = [ 1,  1, -1, -1]  
H_comm_34[:,9]  = [ 1, -1,  1, -1]  
H_comm_34[:,12] = [-1, -1,  1,  1]  
H_comm_34[:,13] = [-1,  1, -1,  1]
h_comm_34 = comm_dist * np.ones(4)
Poly_comm_34 = Polytope(H_comm_34, h_comm_34)

# 4-1
H_comm_41 = np.zeros((4,16))
H_comm_41[:,0]  = [-1, -1,  1,  1]  # x1
H_comm_41[:,1]  = [-1,  1, -1,  1]  # y1
H_comm_41[:,12] = [ 1,  1, -1, -1]  # x4
H_comm_41[:,13] = [ 1, -1,  1, -1]  # y4
h_comm_41 = comm_dist * np.ones(4)
Poly_comm_41 = Polytope(H_comm_41, h_comm_41)

Poly_comm_12_T = Poly_comm_12.cartpower(T+1)
Poly_comm_23_T = Poly_comm_23.cartpower(T+1)
Poly_comm_34_T = Poly_comm_34.cartpower(T+1)
Poly_comm_41_T = Poly_comm_41.cartpower(T+1)

Poly_x = poly_intersect(Poly_x, Poly_comm_12_T)
Poly_x = poly_intersect(Poly_x, Poly_comm_23_T)
Poly_x = poly_intersect(Poly_x, Poly_comm_34_T)
Poly_x = poly_intersect(Poly_x, Poly_comm_41_T)



test_feas = True
file = "simulation_results/simulationT10_4drones.pickle"
save = True

if test_feas:
    [result, SLS_test, Lambda] = swarm_optimize(A_list, B_list, C_list, delay, n_uwv, Poly_x, Poly_u, Poly_w, RTH_opt_eps)
    plot_swarm_trajectory(SLS_test, Poly_x, Poly_w, center_times, radius_times, T, n_uwv, "test_trajectories.pdf")
    swarm_calculate_communication(SLS_test.F, T, n_uwv, tol=1e-3)
    sys.exit()



### SIMULATION #########################################################################################################

key = 'Reweighted Nuclear Norm'
start = time.time()
optimize_RTH_output = swarm_optimize_RTH(A_list, B_list, C_list, delay, n_uwv, Poly_x, Poly_u, Poly_w, N, delta, rank_eps, RTH_opt_eps)
t1 = time.time() - start
optimize_RTH_output.append(t1)



print("begin offdiag three phi optimization")
key = 'Offdiag Three Phis Constrain Phixx'
start = time.time()
optimize_offdiag_three_output = swarm_optimize_RTH_offdiag_three_phis_constrain_phixx(A_list, B_list, C_list, delay, n_uwv, Poly_x, Poly_u, Poly_w, N=8, delta=0.01, rank_eps=1e-7, opt_eps=1e-8)
t2 = time.time() - start
optimize_offdiag_three_output.append(t2)


data = {
'Reweighted Nuclear Norm': optimize_RTH_output,
'Offdiag Three Phis Constrain Phixx': optimize_offdiag_three_output
}

with open(file,"wb") as f:
    pickle.dump(data, f)
    


















### Print result #########################################################################################################


### Baseline Controller

simulation_data = pickle.load(open(file, "rb"))
optimize_RTH_data = simulation_data['Reweighted Nuclear Norm']

SLS_nuc_list= optimize_RTH_data[1]
SLS_nuc = SLS_nuc_list[-1]
Lambda_nuc = optimize_RTH_data[2]
Lambda_nuc_reopt = optimize_RTH_output[3]


print()
print("Delay time:", delay)
print("--- Nuclear Norm -----------------------------------------------------")
print("rank K:", SLS_nuc.rank_F_trunc)
print("band (D,E) = messages:", SLS_nuc.E.shape[0])
print("message times:", np.array(SLS_nuc.F_causal_row_basis)//2)

print("max |K - K_trunc|:", np.max( np.abs(SLS_nuc.F - SLS_nuc.F_trunc) ) )
print("max |Phi - Phi_trunc|:", np.max( np.abs(SLS_nuc.Phi_matrix.value - SLS_nuc.Phi_trunc) ) )
Poly_xu = Poly_x.cart(Poly_u)
print("Error true polytope constraint:", np.max(np.abs( Lambda_nuc.value.dot(Poly_w.H) - Poly_xu.H.dot(SLS_nuc.Phi_matrix.value))))
print("Error truncated polytope constraint:", np.max( np.abs(Lambda_nuc.value.dot(Poly_w.H) - Poly_xu.H.dot(SLS_nuc.Phi_trunc)) ) )
swarm_calculate_communication(SLS_nuc.F, T, n_uwv, tol=1e-3)
print()


Nuc_F_matrix = SLS_nuc.F
save_sparsity_pattern(Nuc_F_matrix, "Nuc_F_matrix")
Nuc_D_matrix = SLS_nuc.D
save_sparsity_pattern(Nuc_D_matrix, "Nuc_D_matrix")
Nuc_E_matrix = SLS_nuc.E
save_sparsity_pattern(Nuc_E_matrix, "Nuc_E_matrix")
Nuc_Phi_matrix = SLS_nuc.Phi_matrix.value
save_sparsity_pattern(Nuc_Phi_matrix, "Nuc_Phi_matrix")
Nuc_Phi_trunc = SLS_nuc.Phi_trunc
save_sparsity_pattern(Nuc_Phi_trunc, "Nuc_Phi_trunc")






### Proposed Controller

simulation_data = pickle.load(open(file, "rb"))
offdiag_three_phis_data = simulation_data['Offdiag Three Phis Constrain Phixx']


SLS_offdiag_three_Phi = offdiag_three_phis_data[1]
time_three_Phi = offdiag_three_phis_data[-1]
print()
print("--- Proposed Controller ------------------------------------")
print("Com time diagI:", time_three_Phi)
SLS_offdiag_three_Phi.calculate_dependent_variables("Reweighted Nuclear Norm")
SLS_offdiag_three_Phi.causal_factorization(rank_eps=1e-7)



print("rank F:", SLS_offdiag_three_Phi.rank_F_trunc)
print("band (D,E) = messages:", SLS_offdiag_three_Phi.E.shape[0])
print("Error true F and truncated F:", np.max( np.abs(SLS_offdiag_three_Phi.F - SLS_offdiag_three_Phi.F_trunc) ) )
print("Error true Phi and truncated Phi:", np.max( np.abs(SLS_offdiag_three_Phi.Phi_matrix.value - SLS_offdiag_three_Phi.Phi_trunc) ) )

plot_swarm_trajectory(
    SLS_data=SLS_offdiag_three_Phi,
    Poly_x=Poly_x,
    Poly_w=Poly_w,
    center_times=center_times,
    radius_times=radius_times,
    T=5,
    n_uwv=4,
    save_path="proposed_controller_trajectories.pdf"
)

swarm_calculate_communication(SLS_offdiag_three_Phi.F, T, n_uwv, tol=1e-3)