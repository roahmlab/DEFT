"""Visualize a specific BDLO5 training sample — runs sim and saves per-timestep PNGs with GT vs prediction."""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import numpy as np
import argparse

torch.set_default_dtype(torch.float64)
torch.manual_seed(1)

from deft.utils.util import DEFT_initialization, construct_b_DLOs, clamp_index, index_init, Train_DEFTData
from deft.core.DEFT_sim import DEFT_sim

parser = argparse.ArgumentParser()
parser.add_argument("--sample", type=int, default=44, help="Sample index to visualize")
args = parser.parse_args()
SAMPLE_IDX = args.sample

# ---- BDLO5 parameters ----
n_branch = 3
n_parent_vertices = 12
n_child1_vertices = 4
n_child2_vertices = 4
cs_n_vert = (4, 4)
n_vert = 12
n_edge = 11
device = "cpu"
total_time = 500
rigid_body_coupling_index = [5, 1]
bdlo5 = True

undeformed_BDLO = torch.tensor([
    [[0.188293, 0.172336, 0.13665, 0.099796, 0.073642, 0.0541, 0.049203,
      0.067681, 0.098314, 0.144249, 0.194441, 0.215592, -0.004642,
      -0.050726, -0.081415, -0.026287, -0.100634, -0.133205]],
    [[0.000706, -0.001975, -0.005219, -0.006773, -0.005287, -0.003288,
      -0.008895, -0.009546, -0.009834, -0.00937, -0.01009, -0.009817,
      -0.00524, -0.006016, -0.003387, -0.008853, -0.009468, -0.006654]],
    [[0.259138, 0.231036, 0.158627, 0.080946, -0.004805, -0.095316,
      -0.183408, -0.272979, -0.348157, -0.413674, -0.475076, -0.499779,
      -0.091729, -0.088754, -0.094614, -0.182658, -0.219473, -0.229931]]
]).permute(1, 2, 0)

# ---- Build sim ----
bend_stiffness_parent = nn.Parameter(1e-3 * torch.ones((1, 1, n_edge), device=device, dtype=torch.float64))
bend_stiffness_child1 = nn.Parameter(1e-3 * torch.ones((1, 1, n_edge), device=device, dtype=torch.float64))
bend_stiffness_child2 = nn.Parameter(1e-3 * torch.ones((1, 1, n_edge), device=device, dtype=torch.float64))
twist_stiffness = nn.Parameter(1e-4 * torch.ones((1, n_branch, n_edge), device=device, dtype=torch.float64))
damping = nn.Parameter(torch.tensor((3., 8., 8.), device=device, dtype=torch.float64))
learning_weight = nn.Parameter(torch.tensor(0.00, device=device, dtype=torch.float64))

parent_clamped_selection = torch.tensor((0, 1, -2, -1))
child1_clamped_selection = torch.tensor((2))
child2_clamped_selection = torch.tensor((2))

parent_v = undeformed_BDLO[:, :n_parent_vertices]
child1_v = undeformed_BDLO[:, n_parent_vertices:n_parent_vertices + n_child1_vertices - 1]
child2_v = undeformed_BDLO[:, n_parent_vertices + n_child1_vertices - 1:]

b_DLO_mass, parent_MOI, children_MOI, _, _, _ = DEFT_initialization(
    parent_v, child1_v, child2_v, n_branch, n_parent_vertices, cs_n_vert,
    rigid_body_coupling_index, 1., 10., (0.5, 0.5), (1, 1), 0.1, bdlo5=True)

b, _ = construct_b_DLOs(1, rigid_body_coupling_index, n_parent_vertices, cs_n_vert, n_branch,
    parent_v, parent_v, child1_v, child1_v, child2_v, child2_v, bdlo5=True)

step1 = torch.zeros_like(b)
step1[:, :, :, 0] = -b[:, :, :, 2]; step1[:, :, :, 1] = -b[:, :, :, 0]; step1[:, :, :, 2] = b[:, :, :, 1]
t = torch.zeros_like(step1)
t[:, :, :, 0] = -step1[:, :, :, 0]; t[:, :, :, 1] = step1[:, :, :, 2]; t[:, :, :, 2] = -step1[:, :, :, 1]
b_undeformed_vert = t[0].view(n_branch, -1, 3)

index_selection1, index_selection2, parent_MOI_index1, parent_MOI_index2 = index_init(
    rigid_body_coupling_index, n_branch)

batch = 1
eval_time_horizon = total_time - 2

clamped_index, parent_theta_clamp, child1_theta_clamp, child2_theta_clamp = clamp_index(
    batch, parent_clamped_selection, child1_clamped_selection, child2_clamped_selection,
    n_branch, n_parent_vertices, True, False, False)

sim = DEFT_sim(
    batch=batch, n_branch=n_branch, n_vert=n_vert, cs_n_vert=cs_n_vert,
    b_init_n_vert=b_undeformed_vert, n_edge=n_edge, b_undeformed_vert=b_undeformed_vert,
    b_DLO_mass=b_DLO_mass, parent_DLO_MOI=parent_MOI, children_DLO_MOI=children_MOI,
    device=device, clamped_index=clamped_index, rigid_body_coupling_index=rigid_body_coupling_index,
    parent_MOI_index1=parent_MOI_index1, parent_MOI_index2=parent_MOI_index2,
    parent_clamped_selection=parent_clamped_selection,
    child1_clamped_selection=child1_clamped_selection,
    child2_clamped_selection=child2_clamped_selection,
    clamp_parent=True, clamp_child1=False, clamp_child2=False,
    index_selection1=index_selection1, index_selection2=index_selection2,
    bend_stiffness_parent=bend_stiffness_parent, bend_stiffness_child1=bend_stiffness_child1,
    bend_stiffness_child2=bend_stiffness_child2, twist_stiffness=twist_stiffness,
    damping=damping, learning_weight=learning_weight,
    use_orientation_constraints=True, use_attachment_constraints=True,
    bdlo5=True, use_child2_orientation=True)

# ---- Load training data ----
print("Loading training data...")
train_dataset = Train_DEFTData(
    5, n_parent_vertices, (n_child1_vertices, n_child2_vertices), n_branch,
    rigid_body_coupling_index, 76, total_time, eval_time_horizon, device,
    bdlo5=True)

total_subs = len(train_dataset)
step_size = max(1, total_subs // 76)
sample_sub_idx = SAMPLE_IDX * step_size
print(f"Sample {SAMPLE_IDX} -> subsequence index {sample_sub_idx}/{total_subs}")

prev, curr, target, mu = train_dataset[sample_sub_idx]
prev = prev.unsqueeze(0)
curr = curr.unsqueeze(0)
target = target.unsqueeze(0)

# ---- Visualize undeformed state ----
print(f"\nUndeformed state (b_undeformed_vert) used in DEFT_sim:")
branch_names = ['Parent', 'Child1', 'Child2']
branch_n = [12, 4, 4]
for br in range(3):
    print(f"  {branch_names[br]} ({branch_n[br]} verts):")
    for vi in range(branch_n[br]):
        v = b_undeformed_vert[br, vi]
        print(f"    [{vi}]: ({v[0]:.4f}, {v[1]:.4f}, {v[2]:.4f})")

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
colors = ['red', 'green', 'blue']
undef_np = b_undeformed_vert.detach().numpy()
for br in range(3):
    nv = branch_n[br]
    pts = undef_np[br, :nv]
    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], '-o', color=colors[br], markersize=6, linewidth=2, label=branch_names[br])
    for vi in range(nv):
        ax.text(pts[vi, 0], pts[vi, 1], pts[vi, 2], f'{branch_names[br][0]}{vi}', fontsize=7, color=colors[br])

all_pts = np.concatenate([undef_np[br, :branch_n[br]] for br in range(3)])
mid = all_pts.mean(0)
r = (all_pts.max(0) - all_pts.min(0)).max() / 2 + 0.05
ax.set_xlim(mid[0]-r, mid[0]+r); ax.set_ylim(mid[1]-r, mid[1]+r); ax.set_zlim(mid[2]-r, mid[2]+r)
ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
ax.set_title('BDLO5 Undeformed State (used in DEFT_sim)')
ax.legend()
plt.show()

matplotlib.use('Agg')  # switch back to non-interactive for saving PNGs

# ---- Run full sim with vis=True ----
print(f"\nRunning simulation (vis=True, saving PNGs to sanity_check/DEFT_5_sample{SAMPLE_IDX}/0/)...")
loss_func = torch.nn.MSELoss()

with torch.no_grad():
    traj_loss, _ = sim.iterative_sim(
        eval_time_horizon, curr, prev, target,
        loss_func, 0.01,
        parent_theta_clamp, child1_theta_clamp, child2_theta_clamp,
        False, vis_type=f"DEFT_5_sample{SAMPLE_IDX}", vis=True)

rmse = np.sqrt(traj_loss.cpu().numpy() / total_time)
print(f"\nFinal RMSE: {rmse:.4f}")
print(f"PNGs saved to: sanity_check/DEFT_5_sample{SAMPLE_IDX}/0/")
