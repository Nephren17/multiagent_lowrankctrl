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




test_feas = True
file = "simulation_results/simulationT10_joint.pickle"
save = True


if test_feas:
    [result, SLS_test, Lambda] = optimize(A_list, B_list, C_list, delay, Poly_x, Poly_u, Poly_w, RTH_opt_eps)
    plot_two_trajectory(SLS_test, Poly_x, Poly_w, center_times, radius_times, T, "simulation_results/test_2_trajectories.pdf")
    # sys.exit()




### SIMULATION #########################################################################################################


key = 'Reweighted Nuclear Norm'
start = time.time()
optimize_RTH_output = optimize_RTH(A_list, B_list, C_list, delay, Poly_x, Poly_u, Poly_w, N, delta, rank_eps, RTH_opt_eps)
t1 = time.time() - start
optimize_RTH_output.append(t1)



# multi-agent communication matrix optimization
print("begin offdiag no constraint optimization")
key = 'Offdiag Phiuy Only'
start = time.time()
optimize_offdiag_output = optimize_RTH_offdiag_no_constraint(A_list, B_list, C_list, delay, Poly_x, Poly_u, Poly_w, N=8, delta=0.01, rank_eps=1e-7, opt_eps=1e-8)
t2 = time.time() - start
optimize_offdiag_output.append(t2)


print("begin offdiag optimization phixx constrained")
key = 'Offdiag Phiuy Constrain Phixx'
start = time.time()
optimize_offdiag_xx_output = optimize_RTH_offdiag_constrain_phixx(A_list, B_list, C_list, delay, Poly_x, Poly_u, Poly_w, N=8, delta=0.01, rank_eps=1e-7, opt_eps=1e-8)
t3 = time.time() - start
optimize_offdiag_xx_output.append(t3)

print("begin offdiag three phi optimization")
key = 'Offdiag Three Phis Constrain Phixx'
start = time.time()
optimize_offdiag_three_output = optimize_RTH_offdiag_three_phis_constrain_phixx(A_list, B_list, C_list, delay, Poly_x, Poly_u, Poly_w, N=8, delta=0.01, rank_eps=1e-7, opt_eps=1e-8)
t4 = time.time() - start
optimize_offdiag_three_output.append(t4)


data = {
'Reweighted Nuclear Norm': optimize_RTH_output,
'Offdiag Phiuy Only': optimize_offdiag_output,
'Offdiag Phiuy Constrain Phixx': optimize_offdiag_xx_output,
'Offdiag Three Phis Constrain Phixx': optimize_offdiag_three_output,
'No Communication': None
}


# multi-agent without any communication
# this is just a case for testing whether no communication is possible
key_diag = 'No Communication'
if comm_dist >= 200:
    start = time.time()
    optimize_no_comm_output = optimize_no_comm_both_ways(A_list, B_list, C_list, delay, Poly_x, Poly_u, Poly_w,opt_eps=1e-11)
    t7 = time.time() - start
    optimize_no_comm_output.append(t7)
    data['No Communication'] = optimize_no_comm_output
else:
    data['No Communication'] = None



with open(file,"wb") as f:
    pickle.dump(data, f)
    
simulation_data = pickle.load(open(file, "rb"))
optimize_RTH_data = simulation_data['Reweighted Nuclear Norm']
offdiag_phiuy_only_data = simulation_data['Offdiag Phiuy Only']
offdiag_phixx_constrained_data = simulation_data['Offdiag Phiuy Constrain Phixx']
offdiag_three_phis_data = simulation_data['Offdiag Three Phis Constrain Phixx']
no_comm_data = simulation_data['No Communication']


SLS_nuc_list= optimize_RTH_data[1]
SLS_nuc = SLS_nuc_list[-1]
Lambda_nuc = optimize_RTH_data[2]





print("Poly_w.H shape:", Poly_w.H.shape)
print("Poly_x.H shape:", Poly_x.H.shape)
print("Poly_u.H shape:", Poly_u.H.shape)
print("SLS_nuc.nx (state dimension):", SLS_nuc.nx)
print("SLS_nuc.ny (measurement noise dimension):", SLS_nuc.ny)





### MAKE PLOTS ############################################################################################
textsize=10
# Trajectory plots
def plot_trajectory(w, Phi_row, n_dim, checkpoints, ax, uav_index=1):
    traj = Phi_row @ w
    traj = traj.reshape((-1, n_dim))
    color = next(ax._get_lines.prop_cycler)['color']

    if uav_index == 1:
        index_offset = 0
    else:
        index_offset = 4

    ax.plot(traj[:, index_offset], traj[:, index_offset + 1], color=color, linewidth=0.5)
    for i in checkpoints:
        ax.plot(traj[i, index_offset], traj[i, index_offset + 1], '.', color=color)
    return



fig1, ax1 = plt.subplots()
colors = [(255/255,114/255,118/255,0.5), (153/255,186/255,221/255,0.5), (204/255,255/255,204/255,0.5)]
counter = 0

for idx in box_check:
    rect_uav1 = patches.Rectangle(
        (center_times[idx][0] - radius_times[idx][0], center_times[idx][1] - radius_times[idx][1]), 2 * radius_times[idx][0], 2 * radius_times[idx][1], 
        linewidth=1, edgecolor='k', facecolor=colors[counter],label=f'$\\mathcal{{X}}_{{1,{idx}}}$')
    ax1.add_patch(rect_uav1)

    rect_uav2 = patches.Rectangle(
        (center_times[idx][4] - radius_times[idx][4], center_times[idx][5] - radius_times[idx][5]),
        2 * radius_times[idx][4], 2 * radius_times[idx][5], linewidth=1, edgecolor='k', facecolor=colors[counter],  label=f'$\\mathcal{{X}}_{{2,{idx}}}$')
    ax1.add_patch(rect_uav2)

    counter += 1

ax1.legend(fontsize=15)
sign = [1, -1]



N_corner = 2
for i in sign:
    for j in sign:
        for k in range(N_corner):
            w_corner = np.array([])
            
            for _ in range(T+1):
                w_corner = np.hstack([
                    w_corner,
                    wx_1_scale   * np.random.choice([-1, 1], 1),
                    wy_1_scale   * np.random.choice([-1, 1], 1),
                    wx_2_scale   * np.random.choice([-1, 1], 1),
                    wy_2_scale   * np.random.choice([-1, 1], 1),
                    wxdot_1_scale* np.random.choice([-1, 1], 1),
                    wydot_1_scale* np.random.choice([-1, 1], 1),
                    wxdot_2_scale* np.random.choice([-1, 1], 1),
                    wydot_2_scale* np.random.choice([-1, 1], 1)
                ])
            
            w_corner = np.hstack([
                w_corner,
                v_scale * np.random.choice([-1, 1], (T+1) * SLS_nuc.ny)
            ])
            
            w_corner[0:SLS_nuc.nx] = center_times[0]
            
            w_corner[0:2] += np.array([i * radius_times[0][0], j * radius_times[0][1]])
            w_corner[4:6] += np.array([i * radius_times[0][2], j * radius_times[0][3]])
            
            plot_trajectory(w_corner, 
                            SLS_nuc.Phi_trunc[0:(T+1)*SLS_nuc.nx, 0:(T+1)*(SLS_nuc.nx + SLS_nuc.ny)],
                            SLS_nuc.nx,
                            box_check,
                            ax1,
                            uav_index=1)

            plot_trajectory(w_corner, 
                            SLS_nuc.Phi_trunc[0:(T+1)*SLS_nuc.nx, 0:(T+1)*(SLS_nuc.nx + SLS_nuc.ny)],
                            SLS_nuc.nx,
                            box_check,
                            ax1,
                            uav_index=2)


N_samples = 2
for i in range(N_samples):
    w = np.array([])
    for _ in range(T+1):
        w = np.hstack([
            w,
            wx_1_scale   * np.random.choice([-1, 1], 1),
            wy_1_scale   * np.random.choice([-1, 1], 1),
            wx_2_scale   * np.random.choice([-1, 1], 1),
            wy_2_scale   * np.random.choice([-1, 1], 1),
            wxdot_1_scale* np.random.choice([-1, 1], 1),
            wydot_1_scale* np.random.choice([-1, 1], 1),
            wxdot_2_scale* np.random.choice([-1, 1], 1),
            wydot_2_scale* np.random.choice([-1, 1], 1)
        ])

    w = np.hstack([w, v_scale*np.random.uniform(-1, 1, (T+1)*(SLS_nuc.ny))])
    w[0:SLS_nuc.nx] = center_times[0]
    w[0] += np.random.uniform(-radius_times[0][0], radius_times[0][0])
    w[1] += np.random.uniform(-radius_times[0][1], radius_times[0][1])
    w[2] += np.random.uniform(-radius_times[0][2], radius_times[0][2])
    w[3] += np.random.uniform(-radius_times[0][3], radius_times[0][3])
    plot_trajectory(w, SLS_nuc.Phi_trunc[0:(T+1)*SLS_nuc.nx, 0:(T+1)*(SLS_nuc.nx + SLS_nuc.ny)], SLS_nuc.nx, box_check, ax1)
ax1.set_xlim([-10,10])
ax1.set_ylim([-10,10])
ax1.locator_params(axis='both', nbins=5)
ax1.tick_params(axis='both', labelsize=textsize)
ax1.grid()
if save:
    fig1.savefig("simulation_results/NuclearNormTrajPlot.pdf", bbox_inches="tight")
#plt.show()


# sparsity plots causal

epsilon = 0
assert np.isclose(np.max(np.abs(SLS_nuc.F_trunc[-2:, :])), 0)
assert np.isclose(np.max(np.abs(SLS_nuc.F_trunc[:, -2:])), 0)
assert np.isclose(np.max(np.abs(SLS_nuc.D[-2:, :])), 0)
assert np.isclose(np.max(np.abs(SLS_nuc.E[:, -2:])), 0)
F = SLS_nuc.F_trunc[0:-2, 0:-2]
D = SLS_nuc.D[0:-2, :]
E = SLS_nuc.E[:, 0:-2]
gs2 = gridspec.GridSpec(2,3, width_ratios=[T*SLS_nuc.nu/E.shape[0],1,T*SLS_nuc.nu/E.shape[0]])
fig2 = plt.figure()
axs20 = plt.subplot(gs2[0,0])
axs21 = plt.subplot(gs2[0,1])
axs22 = plt.subplot(gs2[0,2])
axs20.spy(F, epsilon, markersize=1, color='b', label='$\mathbf{K}$')
axs21.spy(D, epsilon, markersize=1, color='b', label='$\mathbf{D}$')
axs22.spy(E, epsilon, markersize=1, color='b', label='$\mathbf{E}$')
axs20.set_xticks(np.arange(4,T*SLS_nuc.ny,5), np.arange(5,T*SLS_nuc.ny+1,5))
axs20.set_yticks(np.arange(4,T*SLS_nuc.nu,5), np.arange(5,T*SLS_nuc.nu+1,5))
axs20.tick_params(axis='both', labelsize=textsize)
axs21.set_xticks(np.arange(4,D.shape[1],5), np.arange(5,D.shape[1]+1,5))
axs21.set_yticks(np.arange(4,T*SLS_nuc.nu,5), np.arange(5,T*SLS_nuc.nu+1,5))
axs21.tick_params(axis='both', labelsize=textsize)
axs22.set_yticks(np.arange(4,E.shape[0],5), np.arange(5,E.shape[0]+1,5))
axs22.set_xticks(np.arange(4,T*SLS_nuc.ny,5), np.arange(5,T*SLS_nuc.ny+1,5))
axs22.tick_params(axis='both', labelsize=textsize)
axs20.grid()
axs21.grid()
axs22.grid()
axs20.legend(markerscale=0, handlelength=-0.8)
axs21.legend(markerscale=0, handlelength=-0.8)
axs22.legend(markerscale=0, handlelength=-0.8)
#axs22.locator_params(axis='y', nbins=3)
fig2.tight_layout()
if save:
    fig2.savefig("simulation_results/SparsityPlotCausal.pdf", bbox_inches="tight")














### Print Results ############################################################################################



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
plot_matrices_sparcity(SLS_nuc, save_path="simulation_results/controller4_sparcity.pdf")
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
plot_matrices_sparcity(SLS_offdiag, save_path="simulation_results/controller2_sparcity.pdf")
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
plot_matrices_sparcity(SLS_offdiag_diagI, save_path="simulation_results/controller3_sparcity.pdf")
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
plot_matrices_sparcity(SLS_offdiag_three_Phi, save_path="simulation_results/controller4_sparcity.pdf")
SLS_offdiag_three_Phi.causal_factorization(rank_eps=1e-7)
msg_21_diagI, msg_12_diagI = SLS_offdiag_three_Phi.compute_communication_messages(rank_eps=1e-7)
print("UAV2->UAV1 messages diagI:", msg_21_diagI)
print("UAV1->UAV2 messages diagI:", msg_12_diagI)
rank_L1, rank_L2 = SLS_offdiag_three_Phi.compute_offdiag_rank_of_Phi()
print("Supposed Rank of L1 and L2:")
print("Rank(L1), Rank(L2) in Phi_uy:", rank_L1, rank_L2)
plot_two_trajectory(SLS_offdiag_three_Phi, Poly_x, Poly_w, center_times, radius_times, T, "simulation_results/uwv2_controller4_trajectories.pdf")
SLS_offdiag_three_Phi.display_message_time(save_file="SLS_offdiag_three_Phi_results.npz")




no_comm_data = simulation_data['No Communication']
if no_comm_data is not None:
    SLS_no_comm = no_comm_data[1]
    time_no_comm = no_comm_data[-1]
    print()
    print("--- No Communication ------------------------------------")
    print("Com time no comm:", time_no_comm)
    SLS_no_comm.calculate_dependent_variables("Reweighted Nuclear Norm")
    SLS_no_comm.causal_factorization(rank_eps=1e-7)
    print("Rank F no-comm:", SLS_no_comm.rank_F_trunc)
    msg_21_nocomm, msg_12_nocomm = SLS_no_comm.compute_communication_messages()
    print("UAV2->UAV1 messages (no-comm):", msg_21_nocomm)
    print("UAV1->UAV2 messages (no-comm):", msg_12_nocomm)
    rank_L1, rank_L2 = SLS_no_comm.compute_offdiag_rank_of_Phi()
    print("Supposed Rank of L1 and L2:")
    print("Rank(L1), Rank(L2) in Phi_uy:", rank_L1, rank_L2)
    SLS_no_comm.display_message_time(save_file="SLS_no_comm_results.npz")


plt.show()

def run():
    t = ""
    with open('simulation.py') as f:
        t = f.read()
    return t