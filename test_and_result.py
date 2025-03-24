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
import sys


matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
np.random.seed(1)



### Relative Measurement for 2 agent ######################################################################################### 
print("### Task1 : 2 agents with relative measurement #################################################\n")
T = 10
dt = 1
delay = 0

A_0 = np.block([[np.zeros([2, 2]), np.eye(2)], [np.zeros([2, 2]), np.zeros([2, 2])]])
B_0 = np.block([[np.zeros([2, 2])], [np.eye(2)]])
A = sp.linalg.expm(A_0 * dt)
B = np.sum([np.linalg.matrix_power(A_0 * dt, i) / math.factorial(i + 1) for i in np.arange(100)], axis=0).dot(B_0)
C = np.block([[np.eye(2), np.zeros([2, 2])]])

C_test = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [-1, 0, 0, 0, 1, 0, 0, 0],
    [0, -1, 0, 0, 0, 1, 0, 0],
])

A_joint = sp.linalg.block_diag(A, A)
B_joint = sp.linalg.block_diag(B, B)
C_joint = C_test

A_list = (T + 1) * [A_joint]
B_list = (T + 1) * [B_joint]
C_list = (T + 1) * [C_joint]

max_v1 = 2
max_v2 = 2
max_x0 = 1
max_v0 = 0
max_xe = 1.75
max_ve = 1.5
box_x = 9

comm_dist = 15

center_times_uav1 = (T + 1) * [[0, 0, 0, 0]]
center_times_uav2 = (T + 1) * [[0, 0, 0, 0]]
radius_times_uav1 = (T + 1) * [[box_x, box_x, max_v1, max_v1]]
radius_times_uav2 = (T + 1) * [[box_x, box_x, max_v2, max_v2]]

box_check = [0,5,10]

center_times_uav1[box_check[0]] = [-3, -3, 0, 0]
radius_times_uav1[box_check[0]] = [max_x0, max_x0, max_v0, max_v0]
center_times_uav1[box_check[1]] = [3, -3, 0, 0]
radius_times_uav1[box_check[1]] = [max_xe, max_xe, max_v1, max_v1]
center_times_uav1[box_check[2]] = [3, 3, 0, 0]
radius_times_uav1[box_check[2]] = [max_xe, max_xe, max_ve, max_ve]

center_times_uav2[box_check[0]] = [-3, -3, 0, 0]
radius_times_uav2[box_check[0]] = [max_x0, max_x0, max_v0, max_v0]
center_times_uav2[box_check[1]] = [-3, 3, 0, 0]
radius_times_uav2[box_check[1]] = [max_xe, max_xe, max_v2, max_v2]
center_times_uav2[box_check[2]] = [3, 3, 0, 0]
radius_times_uav2[box_check[2]] = [max_xe, max_xe, max_ve, max_ve]

center_times = [center_times_uav1[i] + center_times_uav2[i] for i in range(T + 1)]
radius_times = [radius_times_uav1[i] + radius_times_uav2[i] for i in range(T + 1)]

Poly_x = cart_H_cube(center_times, radius_times)


uav1_x_max = 2
uav1_y_max = 2
uav2_x_max = 2
uav2_y_max = 2
Poly_u = H_cube([0, 0, 0, 0],[uav1_x_max, uav1_y_max, uav2_x_max, uav2_y_max]).cartpower(T + 1)


wx_1_scale = 0.05
wy_1_scale = 0.05
wx_2_scale = 0.05
wy_2_scale = 0.05
wxdot_1_scale = 0.05
wydot_1_scale = 0.05
wxdot_2_scale = 0.05
wydot_2_scale = 0.05

v_scale = 0.05
delta = 0.01
RTH_opt_eps = 1e-11
sparse_opt_eps = 1e-10
rank_eps = 1e-7
N = 8

Poly_w = H_cube(center_times[0], radius_times[0]).cart(
    H_cube([0, 0, 0, 0, 0, 0, 0, 0], [wx_1_scale,wy_1_scale,wx_2_scale,wy_2_scale,wxdot_1_scale,wydot_1_scale,wxdot_2_scale,wydot_2_scale]).cartpower(T)
).cart(H_cube([0, 0, 0, 0], [v_scale] * 4).cartpower(T + 1))



# Constraints for communication distance
H_comm = np.array([
    [ 1,  1,  0,  0, -1, -1,  0,  0],  # (x1 - x2) + (y1 - y2) <= comm_dist
    [ 1, -1,  0,  0, -1,  1,  0,  0],
    [-1,  1,  0,  0,  1, -1,  0,  0],
    [-1, -1,  0,  0,  1,  1,  0,  0],
])
h_comm = comm_dist * np.ones(4)
Poly_comm_single = Polytope(H_comm, h_comm)
Poly_comm = Poly_comm_single.cartpower(T + 1)

Poly_x = poly_intersect(Poly_x, Poly_comm)




file = "simulation_results/simulationT10_joint_relative.pickle"
save = True


simulation_data = pickle.load(open(file, "rb"))
optimize_RTH_data = simulation_data['Reweighted Nuclear Norm']
offdiag_phiuy_only_data = simulation_data['Offdiag Phiuy Only']
offdiag_phixx_constrained_data = simulation_data['Offdiag Phiuy Constrain Phixx']
offdiag_three_phis_data = simulation_data['Offdiag Three Phis Constrain Phixx']
no_comm_data = simulation_data['No Communication']




Poly_xu = Poly_x.cart(Poly_u)



print()
print("--- Baseline Communication Messages ----------------------------------------------------")

SLS_nuc_list= optimize_RTH_data[1]
SLS_nuc = SLS_nuc_list[-1]
Lambda_nuc = optimize_RTH_data[2]

print("rank K:", SLS_nuc.rank_F_trunc)
print("band (D,E) = messages:", SLS_nuc.E.shape[0])
print("message times:", np.array(SLS_nuc.F_causal_row_basis)//2)

print("max |K - K_trunc|:", np.max( np.abs(SLS_nuc.F - SLS_nuc.F_trunc) ) )
print("max |Phi - Phi_trunc|:", np.max( np.abs(SLS_nuc.Phi_matrix.value - SLS_nuc.Phi_trunc) ) )
print("Error true polytope constraint:", np.max(np.abs( Lambda_nuc.value.dot(Poly_w.H) - Poly_xu.H.dot(SLS_nuc.Phi_matrix.value))))
print("Error truncated polytope constraint:", np.max( np.abs(Lambda_nuc.value.dot(Poly_w.H) - Poly_xu.H.dot(SLS_nuc.Phi_trunc)) ) )

SLS_nuc.calculate_dependent_variables(key="Reweighted Nuclear Norm")
msg_21, msg_12 = SLS_nuc.compute_communication_messages(rank_eps=1e-3)
print("UAV2->UAV1 messages:", msg_21)
print("UAV1->UAV2 messages:", msg_12)
plot_matrices_sparcity(SLS_nuc, save_path="simulation_results/relative_controller4_sparcity.pdf")
SLS_nuc.display_message_time(save_file="baseline_results.npz")




SLS_offdiag = offdiag_phiuy_only_data[1]
Lambda_offdiag = offdiag_phiuy_only_data[2]
time_offdiag = offdiag_phiuy_only_data[-1]
print()
print("--- Offdiag Phi_uy Only -----------------------------------------------------")
print("Com time:", time_offdiag)
Poly_xu = Poly_x.cart(Poly_u)
diff = Lambda_offdiag.value.dot(Poly_w.H) - Poly_xu.H.dot(SLS_offdiag.Phi_trunc)
print("Error truncated polytope constraint (Offdiag):", np.max(np.abs(diff)))
print("rank K (offdiag):", SLS_offdiag.rank_F_trunc)
print("band (D,E) = messages (offdiag):", SLS_offdiag.E.shape[0])
SLS_offdiag.calculate_dependent_variables("Reweighted Nuclear Norm")
plot_matrices_sparcity(SLS_offdiag, save_path="simulation_results/relative_controller2_sparcity.pdf")
L21 = SLS_offdiag.extract_sub_communication_matrix(direction='21')
L12 = SLS_offdiag.extract_sub_communication_matrix(direction='12')
msg_21_offdiag, msg_12_offdiag = SLS_offdiag.compute_communication_messages(rank_eps=1e-3)
rank_L21 = np.linalg.matrix_rank(L21, tol=1e-7)
rank_L12 = np.linalg.matrix_rank(L12, tol=1e-7)

print("UAV2->UAV1 messages (offdiag):", msg_21_offdiag)
print("UAV1->UAV2 messages (offdiag):", msg_12_offdiag)
print("messages from rank:", rank_L21, rank_L12)
rank_L1, rank_L2 = SLS_offdiag.compute_offdiag_rank_of_Phi()
print("Supposed Rank of L1 and L2:")
print("Rank(L1), Rank(L2) in Phi_uy:", rank_L1, rank_L2)

np.set_printoptions(precision=10, suppress=True)
np.set_printoptions(threshold=np.inf)
with open("OffdiagMatrices.txt","w") as f:
    f.write("=== F matrix (closed-loop) ===\n")
    f.write(str(SLS_offdiag.F) + "\n\n")

    f.write("=== L21 matrix (UAV2->UAV1) ===\n")
    f.write(str(L21) + "\n\n")

    f.write("=== L12 matrix (UAV1->UAV2) ===\n")
    f.write(str(L12) + "\n\n")
print("F, L21, L12 have been written to 'OffdiagMatrices.txt' for inspection.")
SLS_offdiag.display_message_time(save_file="SLS_offdiag_results.npz")



SLS_offdiag_diagI = offdiag_phixx_constrained_data[1]
time_diagI = offdiag_phixx_constrained_data[-1]
print()
print("--- Offdiag Communication diagI ------------------------------------")
print("Com time diagI:", time_diagI)
SLS_offdiag_diagI.calculate_dependent_variables("Reweighted Nuclear Norm")
plot_matrices_sparcity(SLS_offdiag_diagI, save_path="simulation_results/relative_controller3_sparcity.pdf")
SLS_offdiag_diagI.causal_factorization(rank_eps=1e-7)
msg_21_diagI, msg_12_diagI = SLS_offdiag_diagI.compute_communication_messages(rank_eps=1e-3)
print("UAV2->UAV1 messages diagI:", msg_21_diagI)
print("UAV1->UAV2 messages diagI:", msg_12_diagI)
rank_L1, rank_L2 = SLS_offdiag_diagI.compute_offdiag_rank_of_Phi()
print("Supposed Rank of L1 and L2:")
print("Rank(L1), Rank(L2) in Phi_uy:", rank_L1, rank_L2)
SLS_offdiag_diagI.display_message_time(save_file="SLS_offdiag_diagI_results.npz")



SLS_offdiag_three_Phi = offdiag_three_phis_data[1]
time_three_Phi = offdiag_three_phis_data[-1]
print()
print("--- Offdiag Communication diagI ------------------------------------")
print("Com time diagI:", time_three_Phi)
SLS_offdiag_three_Phi.calculate_dependent_variables("Reweighted Nuclear Norm")
plot_matrices_sparcity(SLS_offdiag_three_Phi, save_path="simulation_results/relative_controller4_sparcity.pdf")
SLS_offdiag_three_Phi.causal_factorization(rank_eps=1e-7)
msg_21_diagI, msg_12_diagI = SLS_offdiag_three_Phi.compute_communication_messages(rank_eps=1e-3)
print("UAV2->UAV1 messages diagI:", msg_21_diagI)
print("UAV1->UAV2 messages diagI:", msg_12_diagI)
rank_L1, rank_L2 = SLS_offdiag_three_Phi.compute_offdiag_rank_of_Phi()
print("Supposed Rank of L1 and L2:")
print("Rank(L1), Rank(L2) in Phi_uy:", rank_L1, rank_L2)
plot_two_trajectory(SLS_offdiag_three_Phi, Poly_x, Poly_w, center_times, radius_times, T, "simulation_results/relative_uwv2_controller4_trajectories.pdf")
SLS_offdiag_three_Phi.display_message_time(save_file="SLS_offdiag_three_Phi_results.npz")

plt.show()






### 2 agent with one agent noisy ######################################################################################### 
print("### Task2 : 2 agents with one noisy #################################################\n")











### 2 agent with delay (mutral detectioin) ######################################################################################### 
print("### Task3 : 2 agents with delay #################################################\n")
T = 10
dt = 1
delay = 2

A_0 = np.block([[np.zeros([2, 2]), np.eye(2)], [np.zeros([2, 2]), np.zeros([2, 2])]])
B_0 = np.block([[np.zeros([2, 2])], [np.eye(2)]])
A = sp.linalg.expm(A_0 * dt)
B = np.sum([np.linalg.matrix_power(A_0 * dt, i) / math.factorial(i + 1) for i in np.arange(100)], axis=0).dot(B_0)
C = np.block([[np.eye(2), np.zeros([2, 2])]])

C_test = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0],
])

A_joint = sp.linalg.block_diag(A, A)
B_joint = sp.linalg.block_diag(B, B)
C_joint = C_test

A_list = (T + 1) * [A_joint]
B_list = (T + 1) * [B_joint]
C_list = (T + 1) * [C_joint]

max_v1 = 2
max_v2 = 2
max_x0 = 1
max_v0 = 0
box_x = 9

comm_dist = 14

center_times_uav1 = (T + 1) * [[0, 0, 0, 0]]
center_times_uav2 = (T + 1) * [[0, 0, 0, 0]]
radius_times_uav1 = (T + 1) * [[box_x, box_x, max_v1, max_v1]]
radius_times_uav2 = (T + 1) * [[box_x, box_x, max_v2, max_v2]]

box_check = [0,5,10]

center_times_uav1[box_check[0]] = [-3, -3, 0, 0]
radius_times_uav1[box_check[0]] = [max_x0, max_x0, max_v0, max_v0]
center_times_uav1[box_check[1]] = [3, -3, 0, 0]
radius_times_uav1[box_check[1]] = [2, 2, max_v1, max_v1]
center_times_uav1[box_check[2]] = [3, 3, 0, 0]
radius_times_uav1[box_check[2]] = [2.5, 2.5, 1, 1]

center_times_uav2[box_check[0]] = [-3, -3, 0, 0]
radius_times_uav2[box_check[0]] = [max_x0, max_x0, max_v0, max_v0]
center_times_uav2[box_check[1]] = [-3, 3, 0, 0]
radius_times_uav2[box_check[1]] = [2, 2, max_v2, max_v2]
center_times_uav2[box_check[2]] = [3, 3, 0, 0]
radius_times_uav2[box_check[2]] = [2.5, 2.5, 1, 1]

center_times = [center_times_uav1[i] + center_times_uav2[i] for i in range(T + 1)]
radius_times = [radius_times_uav1[i] + radius_times_uav2[i] for i in range(T + 1)]

Poly_x = cart_H_cube(center_times, radius_times)


uav1_x_max = 2
uav1_y_max = 2
uav2_x_max = 2
uav2_y_max = 2
Poly_u = H_cube([0, 0, 0, 0],[uav1_x_max, uav1_y_max, uav2_x_max, uav2_y_max]).cartpower(T + 1)



wx_1_scale = 0.05
wy_1_scale = 0.05
wx_2_scale = 0.05
wy_2_scale = 0.05
wxdot_1_scale = 0.05
wydot_1_scale = 0.05
wxdot_2_scale = 0.05
wydot_2_scale = 0.05

v_scale = 0.05
delta = 0.01
RTH_opt_eps = 1e-11
sparse_opt_eps = 1e-10
rank_eps = 1e-7
N = 8

Poly_w = H_cube(center_times[0], radius_times[0]).cart(
    H_cube([0, 0, 0, 0, 0, 0, 0, 0], [wx_1_scale,wy_1_scale,wx_2_scale,wy_2_scale,wxdot_1_scale,wydot_1_scale,wxdot_2_scale,wydot_2_scale]).cartpower(T)
).cart(H_cube([0, 0, 0, 0], [v_scale] * 4).cartpower(T + 1))

# Constraints for communication distance
H_comm = np.array([
    [ 1,  1,  0,  0, -1, -1,  0,  0],  # (x1 - x2) + (y1 - y2) <= comm_dist
    [ 1, -1,  0,  0, -1,  1,  0,  0],
    [-1,  1,  0,  0,  1, -1,  0,  0],
    [-1, -1,  0,  0,  1,  1,  0,  0],
])
h_comm = comm_dist * np.ones(4)
Poly_comm_single = Polytope(H_comm, h_comm)
Poly_comm = Poly_comm_single.cartpower(T + 1)

Poly_x = poly_intersect(Poly_x, Poly_comm)

file = "simulation_results/simulationT10_joint_mutral_detection.pickle"
save = True

simulation_data = pickle.load(open(file, "rb"))
optimize_RTH_data = simulation_data['Reweighted Nuclear Norm']
offdiag_phiuy_only_data = simulation_data['Offdiag Phiuy Only']
offdiag_phixx_constrained_data = simulation_data['Offdiag Phiuy Constrain Phixx']
offdiag_three_phis_data = simulation_data['Offdiag Three Phis Constrain Phixx']
no_comm_data = simulation_data['No Communication']


SLS_nuc_list= optimize_RTH_data[1]
SLS_nuc = SLS_nuc_list[-1]
Lambda_nuc = optimize_RTH_data[2]


Poly_xu = Poly_x.cart(Poly_u)
print()
print("Delay time:", delay)
print()
print("--- Baseline Communication Messages ----------------------------------------------------")
print("rank K:", SLS_nuc.rank_F_trunc)
print("band (D,E) = messages:", SLS_nuc.E.shape[0])
print("message times:", np.array(SLS_nuc.F_causal_row_basis)//2)

print("max |K - K_trunc|:", np.max( np.abs(SLS_nuc.F - SLS_nuc.F_trunc) ) )
print("max |Phi - Phi_trunc|:", np.max( np.abs(SLS_nuc.Phi_matrix.value - SLS_nuc.Phi_trunc) ) )
print("Error true polytope constraint:", np.max(np.abs( Lambda_nuc.value.dot(Poly_w.H) - Poly_xu.H.dot(SLS_nuc.Phi_matrix.value))))
print("Error truncated polytope constraint:", np.max( np.abs(Lambda_nuc.value.dot(Poly_w.H) - Poly_xu.H.dot(SLS_nuc.Phi_trunc)) ) )

SLS_nuc.calculate_dependent_variables(key="Reweighted Nuclear Norm")
msg_21, msg_12 = SLS_nuc.compute_communication_messages(rank_eps=1e-7)
print("UAV2->UAV1 messages:", msg_21)
print("UAV1->UAV2 messages:", msg_12)
plot_matrices_sparcity(SLS_nuc, save_path="simulation_results/controller1_sparcity_delay.pdf")
SLS_nuc.display_message_time(save_file="baseline_results.npz")




SLS_offdiag = offdiag_phiuy_only_data[1]
Lambda_offdiag = offdiag_phiuy_only_data[2]
time_offdiag = offdiag_phiuy_only_data[-1]
print()
print("--- Offdiag Phi_uy Only -----------------------------------------------------")
print("Com time:", time_offdiag)
Poly_xu = Poly_x.cart(Poly_u)
diff = Lambda_offdiag.value.dot(Poly_w.H) - Poly_xu.H.dot(SLS_offdiag.Phi_trunc)
print("Error truncated polytope constraint (Offdiag):", np.max(np.abs(diff)))
print("rank K (offdiag):", SLS_offdiag.rank_F_trunc)
print("band (D,E) = messages (offdiag):", SLS_offdiag.E.shape[0])
SLS_offdiag.calculate_dependent_variables("Reweighted Nuclear Norm")
plot_matrices_sparcity(SLS_offdiag, save_path="simulation_results/controller2_sparcity_delay.pdf")
L21 = SLS_offdiag.extract_sub_communication_matrix(direction='21')
L12 = SLS_offdiag.extract_sub_communication_matrix(direction='12')
msg_21_offdiag, msg_12_offdiag = SLS_offdiag.compute_communication_messages(rank_eps=1e-9)
rank_L21 = np.linalg.matrix_rank(L21, tol=1e-7)
rank_L12 = np.linalg.matrix_rank(L12, tol=1e-7)

print("UAV2->UAV1 messages (offdiag):", msg_21_offdiag)
print("UAV1->UAV2 messages (offdiag):", msg_12_offdiag)
print("messages from rank:", rank_L21, rank_L12)
rank_L1, rank_L2 = SLS_offdiag.compute_offdiag_rank_of_Phi()
print("Supposed Rank of L1 and L2:")
print("Rank(L1), Rank(L2) in Phi_uy:", rank_L1, rank_L2)

np.set_printoptions(precision=10, suppress=True)
np.set_printoptions(threshold=np.inf)
with open("OffdiagMatrices.txt","w") as f:
    f.write("=== F matrix (closed-loop) ===\n")
    f.write(str(SLS_offdiag.F) + "\n\n")

    f.write("=== L21 matrix (UAV2->UAV1) ===\n")
    f.write(str(L21) + "\n\n")

    f.write("=== L12 matrix (UAV1->UAV2) ===\n")
    f.write(str(L12) + "\n\n")
print("F, L21, L12 have been written to 'OffdiagMatrices.txt' for inspection.")
SLS_offdiag.display_message_time(save_file="SLS_offdiag_results.npz")



SLS_offdiag_diagI = offdiag_phixx_constrained_data[1]
time_diagI = offdiag_phixx_constrained_data[-1]
print()
print("--- Offdiag Communication diagI ------------------------------------")
print("Com time diagI:", time_diagI)
SLS_offdiag_diagI.calculate_dependent_variables("Reweighted Nuclear Norm")
plot_matrices_sparcity(SLS_offdiag_diagI, save_path="simulation_results/controller3_sparcity_delay.pdf")
SLS_offdiag_diagI.causal_factorization(rank_eps=1e-7)
msg_21_diagI, msg_12_diagI = SLS_offdiag_diagI.compute_communication_messages(rank_eps=1e-7)
print("UAV2->UAV1 messages diagI:", msg_21_diagI)
print("UAV1->UAV2 messages diagI:", msg_12_diagI)
rank_L1, rank_L2 = SLS_offdiag_diagI.compute_offdiag_rank_of_Phi()
print("Supposed Rank of L1 and L2:")
print("Rank(L1), Rank(L2) in Phi_uy:", rank_L1, rank_L2)
SLS_offdiag_diagI.display_message_time(save_file="SLS_offdiag_diagI_results.npz")



SLS_offdiag_three_Phi = offdiag_three_phis_data[1]
time_three_Phi = offdiag_three_phis_data[-1]
print()
print("--- Offdiag Communication diagI ------------------------------------")
print("Com time diagI:", time_three_Phi)
SLS_offdiag_three_Phi.calculate_dependent_variables("Reweighted Nuclear Norm")
plot_matrices_sparcity(SLS_offdiag_three_Phi, save_path="simulation_results/controller4_sparcity_delay.pdf")
SLS_offdiag_three_Phi.causal_factorization(rank_eps=1e-7)
msg_21_diagI, msg_12_diagI = SLS_offdiag_three_Phi.compute_communication_messages(rank_eps=1e-7)
print("UAV2->UAV1 messages diagI:", msg_21_diagI)
print("UAV1->UAV2 messages diagI:", msg_12_diagI)
rank_L1, rank_L2 = SLS_offdiag_three_Phi.compute_offdiag_rank_of_Phi()
print("Supposed Rank of L1 and L2:")
print("Rank(L1), Rank(L2) in Phi_uy:", rank_L1, rank_L2)
plot_two_trajectory(SLS_offdiag_three_Phi, Poly_x, Poly_w, center_times, radius_times, T, "simulation_results/uwv2_controller4_delay_trajectories.pdf")
SLS_offdiag_three_Phi.display_message_time(save_file="SLS_offdiag_three_Phi_results.npz")








### Relative Measurement for 4 agent ######################################################################################### 
print("### Task4 : 4 agents with relative measurement #################################################\n")

T = 10
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
C_test[1, 1] = 1.0 

C_test[2, 0] = -1.0
C_test[2, 4] = 1.0
C_test[3, 1] = -1.0
C_test[3, 5] = 1.0

C_test[4, 4]  = -1.0
C_test[4, 8]  = 1.0
C_test[5, 5] = -1.0
C_test[5, 9] = 1.0

C_test[6, 0] = -1.0
C_test[6, 12] = 1.0
C_test[7, 1]  = -1.0
C_test[7, 13]  = 1.0

C_joint = C_test


A_list = (T+1)*[A_joint]
B_list = (T+1)*[B_joint]
C_list = (T+1)*[C_joint]


max_v = 2.0
max_x0 = 1.0 
max_xe = 1.75
max_v0 = 0.0
max_ve = 1.5
box_x = 6.0   
box_check = [0, 5, 10]

center_uav1 = (T+1)*[[0,0,0,0]]
radius_uav1 = (T+1)*[[box_x,box_x,max_v,max_v]]
center_uav1[box_check[0]] = [-3, -3, 0, 0]
radius_uav1[box_check[0]] = [max_x0, max_x0, max_v0, max_v0]
center_uav1[box_check[1]] = [3, -3, 0, 0]
radius_uav1[box_check[1]] = [max_xe, max_xe, max_ve, max_ve]
center_uav1[box_check[2]] = [3, 3, 0, 0]
radius_uav1[box_check[2]] = [max_xe, max_xe, max_ve, max_ve]

center_uav2 = (T+1)*[[0,0,0,0]]
radius_uav2 = (T+1)*[[box_x,box_x,max_v,max_v]]
center_uav2[box_check[0]] = [-3, 3, 0, 0]
radius_uav2[box_check[0]] = [max_x0, max_x0, max_v0, max_v0]
center_uav2[box_check[1]] = [-3, -3, 0, 0] 
radius_uav2[box_check[1]] = [max_xe, max_xe, max_ve, max_ve]
center_uav2[box_check[2]] = [3, -3, 0, 0]
radius_uav2[box_check[2]] = [max_xe, max_xe, max_ve, max_ve]

center_uav3 = (T+1)*[[0,0,0,0]]
radius_uav3 = (T+1)*[[box_x,box_x,max_v,max_v]]
center_uav3[box_check[0]] = [3, 3, 0, 0]
radius_uav3[box_check[0]] = [max_x0, max_x0, max_v0, max_v0]
center_uav3[box_check[1]] = [-3, 3, 0, 0]
radius_uav3[box_check[1]] = [max_xe, max_xe, max_ve, max_ve]
center_uav3[box_check[2]] = [-3, -3, 0, 0]
radius_uav3[box_check[2]] = [max_xe, max_xe, max_ve, max_ve]

center_uav4 = (T+1)*[[0,0,0,0]]
radius_uav4 = (T+1)*[[box_x,box_x,max_v,max_v]]
center_uav4[box_check[0]] = [3, -3, 0, 0]
radius_uav4[box_check[0]] = [max_x0, max_x0, max_v0, max_v0]
center_uav4[box_check[1]] = [3, 3, 0, 0]
radius_uav4[box_check[1]] = [max_xe, max_xe, max_ve, max_ve]
center_uav4[box_check[2]] = [-3, 3, 0, 0]
radius_uav4[box_check[2]] = [max_xe, max_xe, max_ve, max_ve]

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
RTH_opt_eps = 1e-8
delta = 0.01
rank_eps = 1e-5
N = 8

w_scale = [pos_noise, pos_noise, vel_noise, vel_noise]*n_uwv
v_scale = [ctrl_noise]*8

Poly_w = H_cube(center_times[0], radius_times[0]).cart(
    H_cube([0]*(16), w_scale).cartpower(T)
).cart(H_cube([0]*8, v_scale).cartpower(T+1))


comm_dist = 10
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



file = "simulation_results/simulationT10_4drones_relative.pickle"
save = True

simulation_data = pickle.load(open(file, "rb"))
optimize_RTH_data = simulation_data['Reweighted Nuclear Norm']

SLS_nuc_list= optimize_RTH_data[1]
SLS_nuc = SLS_nuc_list[-1]
Lambda_nuc = optimize_RTH_data[2]


print()
print("Delay time:", delay)
print("--- Nuclear Norm -----------------------------------------------------")
print("rank F:", np.linalg.matrix_rank(SLS_nuc.F_trunc, tol=rank_eps))
print("band (D,E) = messages:", SLS_nuc.E.shape[0])
print("message times:", np.array(SLS_nuc.F_causal_row_basis)//2)

print("max |K - K_trunc|:", np.max( np.abs(SLS_nuc.F - SLS_nuc.F_trunc) ) )
print("max |Phi - Phi_trunc|:", np.max( np.abs(SLS_nuc.Phi_matrix.value - SLS_nuc.Phi_trunc) ) )
Poly_xu = Poly_x.cart(Poly_u)
print("Error true polytope constraint:", np.max(np.abs( Lambda_nuc.value.dot(Poly_w.H) - Poly_xu.H.dot(SLS_nuc.Phi_matrix.value))))
print("Error truncated polytope constraint:", np.max( np.abs(Lambda_nuc.value.dot(Poly_w.H) - Poly_xu.H.dot(SLS_nuc.Phi_trunc)) ) )
swarm_communication_message_num(SLS_nuc.F, T, n_uwv)
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
SLS_offdiag_three_Phi.causal_factorization(rank_eps=rank_eps)
print("rank F:", np.linalg.matrix_rank(SLS_offdiag_three_Phi.F_trunc, tol=rank_eps))
print("band (D,E) = messages:", SLS_offdiag_three_Phi.E.shape[0])
print("Error true F and truncated F:", np.max( np.abs(SLS_offdiag_three_Phi.F - SLS_offdiag_three_Phi.F_trunc) ) )
print("Error true Phi and truncated Phi:", np.max( np.abs(SLS_offdiag_three_Phi.Phi_matrix.value - SLS_offdiag_three_Phi.Phi_trunc) ) )

plot_swarm_trajectory(
    SLS_data=SLS_offdiag_three_Phi,
    Poly_x=Poly_x,
    Poly_w=Poly_w,
    center_times=center_times,
    radius_times=radius_times,
    T=10,
    n_uwv=4,
    save_path="controller4_T=10_UWV=4.pdf"
)
swarm_communication_message_num(SLS_offdiag_three_Phi.F, T, n_uwv)






