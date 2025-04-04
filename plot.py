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


matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
np.random.seed(1)



############################ A: Asymmetric Control and Noise #################################
print("===================== A: Asymmetric Control and Noise =====================")
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
file = "simulation_results/simulationT10_joint_noisy_agent.pickle"
simulation_data = pickle.load(open(file, "rb"))
optimize_RTH_data = simulation_data['Reweighted Nuclear Norm']
proposed_data = simulation_data['Proposed Offdiag Result']
SLS_nuc_list= optimize_RTH_data[1]
SLS_nuc = SLS_nuc_list[-1]
Lambda_nuc = optimize_RTH_data[2]
Poly_xu = Poly_x.cart(Poly_u)
print("--- Baseline Communication Messages ----------------------------------------------------")
SLS_nuc.calculate_dependent_variables(key="Reweighted Nuclear Norm")
msg_21, msg_12 = SLS_nuc.compute_communication_messages(rank_eps=1e-7)
print("UAV2->UAV1 messages:", msg_21)
print("UAV1->UAV2 messages:", msg_12)
SLS_proposed = proposed_data[1]
print("--- Proposed Controller Communication Messages ------------------------------------")
SLS_proposed.calculate_dependent_variables("Reweighted Nuclear Norm")
msg_21_proposed, msg_12_proposed = SLS_proposed.compute_communication_messages(rank_eps=1e-7)
print("UAV2->UAV1 messages:", msg_21_proposed)
print("UAV1->UAV2 messages:", msg_12_proposed)
plot_two_noisy(SLS_proposed, Poly_x, Poly_w, center_times, radius_times, T, "simulation_results/noisy_proposed_trajectories.pdf")
print()
print()


############################ B: Decoupled Measurements #################################
print("===================== B: Decoupled Measurements =====================")
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
    [0, 0, 0, 0, 1, 0, 0, 0],
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
rank_eps = 1e-7
N = 8
Poly_w = H_cube(center_times[0], radius_times[0]).cart(
    H_cube([0, 0, 0, 0, 0, 0, 0, 0], [wx_1_scale,wy_1_scale,wx_2_scale,wy_2_scale,wxdot_1_scale,wydot_1_scale,wxdot_2_scale,wydot_2_scale]).cartpower(T)
).cart(H_cube([0, 0, 0, 0], [v_scale] * 4).cartpower(T + 1))
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
file = "simulation_results/simulationT10_joint_normal.pickle"
simulation_data = pickle.load(open(file, "rb"))
optimize_RTH_data = simulation_data['Reweighted Nuclear Norm']
proposed_method_data = simulation_data['Proposed Offdiag Result']
decentral_data = simulation_data['No Communication']
SLS_nuc_list= optimize_RTH_data[1]
SLS_nuc = SLS_nuc_list[-1]
Lambda_nuc = optimize_RTH_data[2]
Poly_xu = Poly_x.cart(Poly_u)
print("--- Baseline Communication Messages ----------------------------------------------------")
SLS_nuc.calculate_dependent_variables(key="Reweighted Nuclear Norm")
msg_21, msg_12 = SLS_nuc.compute_communication_messages(rank_eps=1e-7)
print("UAV2->UAV1 messages:", msg_21)
print("UAV1->UAV2 messages:", msg_12)
SLS_proposed_method = proposed_method_data[1]
print("--- Proposed Controller Communication Messages ------------------------------------")
SLS_proposed_method.calculate_dependent_variables("Reweighted Nuclear Norm")
msg_21_proposed, msg_12_proposed = SLS_proposed_method.compute_communication_messages(rank_eps=1e-7)
print("UAV2->UAV1 messages:", msg_21_proposed)
print("UAV1->UAV2 messages:", msg_12_proposed)
plot_two_trajectory(SLS_proposed_method, Poly_x, Poly_w, center_times, radius_times, T, "simulation_results/normal_proposed_trajectories.pdf")
decentral_data = simulation_data['No Communication']
decentral_data = simulation_data['No Communication']
if decentral_data is not None:
    SLS_decentral = decentral_data[1]
    print("--- Decentral Controller Communication Messages ------------------------------------")
    SLS_decentral.calculate_dependent_variables("Reweighted Nuclear Norm")
    msg_21_decentral, msg_12_decentral = SLS_decentral.compute_communication_messages(rank_eps=1e-7)
    print("UAV2->UAV1 messages:", msg_21_decentral)
    print("UAV1->UAV2 messages:", msg_12_decentral)
print()

############################ B: Relative Measurements #################################
print("===================== B: Relative Measurements =====================")
T = 10
dt = 1
A_0 = np.block([[np.zeros([2, 2]), np.eye(2)], [np.zeros([2, 2]), np.zeros([2, 2])]])
B_0 = np.block([[np.zeros([2, 2])], [np.eye(2)]])
A = sp.linalg.expm(A_0 * dt)
B = np.sum([np.linalg.matrix_power(A_0 * dt, i) / math.factorial(i + 1) for i in np.arange(100)], axis=0).dot(B_0)
A_joint = sp.linalg.block_diag(A, A)
B_joint = sp.linalg.block_diag(B, B)
C_joint = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [-1, 0, 0, 0, 1, 0, 0, 0],
    [0, -1, 0, 0, 0, 1, 0, 0],
])
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
rank_eps = 1e-7
N = 8
Poly_w = H_cube(center_times[0], radius_times[0]).cart(
    H_cube([0, 0, 0, 0, 0, 0, 0, 0], [wx_1_scale,wy_1_scale,wx_2_scale,wy_2_scale,wxdot_1_scale,wydot_1_scale,wxdot_2_scale,wydot_2_scale]).cartpower(T)
).cart(H_cube([0, 0, 0, 0], [v_scale] * 4).cartpower(T + 1))
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
simulation_data = pickle.load(open(file, "rb"))
optimize_RTH_data = simulation_data['Reweighted Nuclear Norm']
proposed_data = simulation_data['Proposed Offdiag Result']
SLS_nuc_list= optimize_RTH_data[1]
SLS_nuc = SLS_nuc_list[-1]
Lambda_nuc = optimize_RTH_data[2]
Poly_xu = Poly_x.cart(Poly_u)
print("--- Baseline Communication Messages ----------------------------------------------------")
SLS_nuc.calculate_dependent_variables(key="Reweighted Nuclear Norm")
msg_21, msg_12 = SLS_nuc.compute_communication_messages(rank_eps=1e-7)
print("UAV2->UAV1 messages:", msg_21)
print("UAV1->UAV2 messages:", msg_12)
SLS_proposed = proposed_data[1]
print("--- Proposed Controller Communication Messages ------------------------------------")
SLS_proposed.calculate_dependent_variables("Reweighted Nuclear Norm")
msg_21_proposed, msg_12_proposed = SLS_proposed.compute_communication_messages(rank_eps=1e-7)
print("UAV2->UAV1 messages:", msg_21_proposed)
print("UAV1->UAV2 messages:", msg_12_proposed)
plot_two_trajectory(SLS_proposed, Poly_x, Poly_w, center_times, radius_times, T, "simulation_results/relative2_proposed_trajectories.pdf")
print()

############################ B: Heterogeneous Sensors #################################
print("===================== B: Relative Measurements: Delay = 0 =====================")
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
C_joint = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0],
])
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
file = f"simulation_results/simulationT10_joint_mutral_detection_delay_{delay}.pickle"
simulation_data = pickle.load(open(file, "rb"))
optimize_RTH_data = simulation_data['Reweighted Nuclear Norm']
proposed_data = simulation_data['Proposed Offdiag Result']
SLS_nuc_list= optimize_RTH_data[1]
SLS_nuc = SLS_nuc_list[-1]
Lambda_nuc = optimize_RTH_data[2]
Poly_xu = Poly_x.cart(Poly_u)
print("Delay time:", delay)
print("--- Baseline Communication Messages ----------------------------------------------------")
SLS_nuc.calculate_dependent_variables(key="Reweighted Nuclear Norm")
msg_21, msg_12 = SLS_nuc.compute_communication_messages(rank_eps=1e-7)
print("UAV2->UAV1 messages:", msg_21)
print("UAV1->UAV2 messages:", msg_12)
SLS_proposed = proposed_data[1]
print("--- Proposed Controller Communication Messages ------------------------------------")
SLS_proposed.calculate_dependent_variables("Reweighted Nuclear Norm")
msg_21_proposed, msg_12_proposed = SLS_proposed.compute_communication_messages(rank_eps=1e-7)
print("UAV2->UAV1 messages:", msg_21_proposed)
print("UAV1->UAV2 messages:", msg_12_proposed)
plot_two_trajectory(SLS_proposed, Poly_x, Poly_w, center_times, radius_times, T, "simulation_results/heterogeneus_proposed_trajectories.pdf")
print()


print("===================== B: Relative Measurements: Delay = 1 =====================")
T = 10
dt = 1
delay = 1
A_0 = np.block([[np.zeros([2, 2]), np.eye(2)], [np.zeros([2, 2]), np.zeros([2, 2])]])
B_0 = np.block([[np.zeros([2, 2])], [np.eye(2)]])
A = sp.linalg.expm(A_0 * dt)
B = np.sum([np.linalg.matrix_power(A_0 * dt, i) / math.factorial(i + 1) for i in np.arange(100)], axis=0).dot(B_0)
C = np.block([[np.eye(2), np.zeros([2, 2])]])
A_joint = sp.linalg.block_diag(A, A)
B_joint = sp.linalg.block_diag(B, B)
C_joint = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0],
])
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
file = f"simulation_results/simulationT10_joint_mutral_detection_delay_{delay}.pickle"
simulation_data = pickle.load(open(file, "rb"))
optimize_RTH_data = simulation_data['Reweighted Nuclear Norm']
proposed_data = simulation_data['Proposed Offdiag Result']
SLS_nuc_list= optimize_RTH_data[1]
SLS_nuc = SLS_nuc_list[-1]
Lambda_nuc = optimize_RTH_data[2]
Poly_xu = Poly_x.cart(Poly_u)
print("Delay time:", delay)
print("--- Baseline Communication Messages ----------------------------------------------------")
SLS_nuc.calculate_dependent_variables(key="Reweighted Nuclear Norm")
msg_21, msg_12 = SLS_nuc.compute_communication_messages(rank_eps=1e-7)
print("UAV2->UAV1 messages:", msg_21)
print("UAV1->UAV2 messages:", msg_12)
SLS_proposed = proposed_data[1]
print("--- Proposed Controller Communication Messages ------------------------------------")
SLS_proposed.calculate_dependent_variables("Reweighted Nuclear Norm")
msg_21_proposed, msg_12_proposed = SLS_proposed.compute_communication_messages(rank_eps=1e-7)
print("UAV2->UAV1 messages:", msg_21_proposed)
print("UAV1->UAV2 messages:", msg_12_proposed)
plot_two_trajectory(SLS_proposed, Poly_x, Poly_w, center_times, radius_times, T, "simulation_results/heterogeneus_proposed_trajectories.pdf")
print()


print("===================== B: Relative Measurements: Delay = 2 =====================")
T = 10
dt = 1
delay = 2
A_0 = np.block([[np.zeros([2, 2]), np.eye(2)], [np.zeros([2, 2]), np.zeros([2, 2])]])
B_0 = np.block([[np.zeros([2, 2])], [np.eye(2)]])
A = sp.linalg.expm(A_0 * dt)
B = np.sum([np.linalg.matrix_power(A_0 * dt, i) / math.factorial(i + 1) for i in np.arange(100)], axis=0).dot(B_0)
C = np.block([[np.eye(2), np.zeros([2, 2])]])
A_joint = sp.linalg.block_diag(A, A)
B_joint = sp.linalg.block_diag(B, B)
C_joint = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0],
])
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
file = f"simulation_results/simulationT10_joint_mutral_detection_delay_{delay}.pickle"
simulation_data = pickle.load(open(file, "rb"))
optimize_RTH_data = simulation_data['Reweighted Nuclear Norm']
proposed_data = simulation_data['Proposed Offdiag Result']
SLS_nuc_list= optimize_RTH_data[1]
SLS_nuc = SLS_nuc_list[-1]
Lambda_nuc = optimize_RTH_data[2]
Poly_xu = Poly_x.cart(Poly_u)
print("Delay time:", delay)
print("--- Baseline Communication Messages ----------------------------------------------------")
SLS_nuc.calculate_dependent_variables(key="Reweighted Nuclear Norm")
msg_21, msg_12 = SLS_nuc.compute_communication_messages(rank_eps=1e-7)
print("UAV2->UAV1 messages:", msg_21)
print("UAV1->UAV2 messages:", msg_12)
SLS_proposed = proposed_data[1]
print("--- Proposed Controller Communication Messages ------------------------------------")
SLS_proposed.calculate_dependent_variables("Reweighted Nuclear Norm")
msg_21_proposed, msg_12_proposed = SLS_proposed.compute_communication_messages(rank_eps=1e-7)
print("UAV2->UAV1 messages:", msg_21_proposed)
print("UAV1->UAV2 messages:", msg_12_proposed)
plot_two_trajectory(SLS_proposed, Poly_x, Poly_w, center_times, radius_times, T, "simulation_results/heterogeneus_proposed_trajectories.pdf")
print()
print()



############################ C: Four Vehicles #################################
print("===================== C: Four Vehicles =====================")
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
n_uwv = 4
A_joint = sp.linalg.block_diag(*(A for _ in range(n_uwv)))
B_joint = sp.linalg.block_diag(*(B for _ in range(n_uwv)))
C_joint = np.zeros((8, 16)) 
for i in range(n_uwv):
    C_joint[2*i:2*i+2, 4*i:4*i+2] = np.eye(2)
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
radius_uav1[box_check[0]] = [max_x0, 1, max_v0, max_v0]
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
rank_eps = 1e-7
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
H_comm_23 = np.zeros((4,16))
H_comm_23[:,4]  = [ 1,  1, -1, -1]  # x2
H_comm_23[:,5]  = [ 1, -1,  1, -1]  # y2
H_comm_23[:,8]  = [-1, -1,  1,  1]  # x3
H_comm_23[:,9]  = [-1,  1, -1,  1]  # y3
h_comm_23 = comm_dist * np.ones(4)
Poly_comm_23 = Polytope(H_comm_23, h_comm_23)
H_comm_34 = np.zeros((4,16))
H_comm_34[:,8]  = [ 1,  1, -1, -1]  
H_comm_34[:,9]  = [ 1, -1,  1, -1]  
H_comm_34[:,12] = [-1, -1,  1,  1]  
H_comm_34[:,13] = [-1,  1, -1,  1]
h_comm_34 = comm_dist * np.ones(4)
Poly_comm_34 = Polytope(H_comm_34, h_comm_34)
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
simulation_data = pickle.load(open(file, "rb"))
optimize_RTH_data = simulation_data['Reweighted Nuclear Norm']
SLS_nuc_list= optimize_RTH_data[1]
SLS_nuc = SLS_nuc_list[-1]
Lambda_nuc = optimize_RTH_data[2]
print("--- Baseline Communication Messages -----------------------------------------------------")
SLS_nuc.calculate_dependent_variables("Reweighted Nuclear Norm")
swarm_communication_message_num(SLS_nuc.F, T, n_uwv)
proposed_data = simulation_data['Proposed Offdiag Result']
SLS_proposed = proposed_data[1]
print("--- Proposed Controller Communication Messages------------------------------------")
SLS_proposed.calculate_dependent_variables("Reweighted Nuclear Norm")
plot_swarm_trajectory_4sub(
    SLS_data=SLS_proposed,
    center_times=center_times,
    radius_times=radius_times,
    T=10,
    n_uwv=4,
    save_path="relative_4_proposed_trajectories.pdf"
)
swarm_communication_message_num(SLS_proposed.F, T, n_uwv)