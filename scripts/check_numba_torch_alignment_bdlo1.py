"""
Numerical alignment check for BDLO1: numba vs pytorch paths for both
`iterative_predict` and `iterative_sim`. Mirrors the BDLO5 check but with
BDLO1's setup so we can compare drift between bdlo5=True and bdlo5=False
code paths.
"""
import os
import sys
import pickle

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import pandas as pd

from deft.utils.util import (
    DEFT_initialization, construct_b_DLOs, construct_BDLOs_data,
    clamp_index, index_init,
)
from deft.core.DEFT_sim import DEFT_sim


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------
torch.set_default_dtype(torch.float64)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(1)
np.random.seed(1)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


# ---------------------------------------------------------------------------
# BDLO1 problem setup (matches scripts/DEFT_train.py BDLO_type==1, ends clamp)
# ---------------------------------------------------------------------------
device = "cpu"
n_branch = 3
dt = 0.01

n_parent_vertices = 13
n_child1_vertices = 5
n_child2_vertices = 4
cs_n_vert = (n_child1_vertices, n_child2_vertices)
n_vert = n_parent_vertices
n_edge = n_vert - 1
rigid_body_coupling_index = [4, 8]
bdlo5 = False

undeformed_BDLO = torch.tensor([[[-0.6790, -0.6355, -0.5595, -0.4539, -0.3688, -0.2776, -0.1857,
                                  -0.0991, 0.0102, 0.0808, 0.1357, 0.2081, 0.2404, -0.4279,
                                  -0.4880, -0.5394, -0.5559, 0.0698, 0.0991, 0.1125]],
                                [[0.0035, -0.0066, -0.0285, -0.0349, -0.0704, -0.0663, -0.0744,
                                  -0.0957, -0.0702, -0.0592, -0.0452, -0.0236, -0.0134, -0.0813,
                                  -0.1233, -0.1875, -0.2178, -0.1044, -0.1858, -0.2165]],
                                [[0.0108, 0.0104, 0.0083, 0.0104, 0.0083, 0.0145, 0.0133,
                                  0.0198, 0.0155, 0.0231, 0.0199, 0.0154, 0.0169, 0.0160,
                                  0.0153, 0.0090, 0.0121, 0.0205, 0.0155, 0.0148]]]).permute(1, 2, 0)

parent_vertices_undeform = undeformed_BDLO[:, :n_parent_vertices]
child1_vertices_undeform = undeformed_BDLO[:, n_parent_vertices: n_parent_vertices + n_child1_vertices - 1]
child2_vertices_undeform = undeformed_BDLO[:, n_parent_vertices + n_child1_vertices - 1:]

parent_mass_scale = 1.
parent_moment_scale = 10.
moment_ratio = 0.1
children_moment_scale = (0.5, 0.5)
children_mass_scale = (1, 1)

parent_clamped_selection = torch.tensor((0, 1, -2, -1))
child1_clamped_selection = torch.tensor((2))
child2_clamped_selection = torch.tensor((2))
clamp_parent = True
clamp_child1 = False
clamp_child2 = False
batch = 1


def build_sim():
    bend_stiffness_parent = nn.Parameter(4e-3 * torch.ones((1, 1, n_edge), device=device, dtype=torch.float64))
    bend_stiffness_child1 = nn.Parameter(4e-3 * torch.ones((1, 1, n_edge), device=device, dtype=torch.float64))
    bend_stiffness_child2 = nn.Parameter(4e-3 * torch.ones((1, 1, n_edge), device=device, dtype=torch.float64))
    twist_stiffness = nn.Parameter(1e-4 * torch.ones((1, n_branch, n_edge), device=device, dtype=torch.float64))
    damping = nn.Parameter(torch.tensor((2.5, 2.5, 2.5), device=device, dtype=torch.float64))
    learning_weight = nn.Parameter(torch.tensor(0.0, device=device, dtype=torch.float64))

    b_DLO_mass, parent_MOI, children_MOI, _, _, _ = DEFT_initialization(
        parent_vertices_undeform, child1_vertices_undeform, child2_vertices_undeform,
        n_branch, n_parent_vertices, cs_n_vert, rigid_body_coupling_index,
        parent_mass_scale, parent_moment_scale, children_moment_scale, children_mass_scale,
        moment_ratio, bdlo5=bdlo5,
    )

    b_DLOs_vertices_undeform, _ = construct_b_DLOs(
        batch, rigid_body_coupling_index, n_parent_vertices, cs_n_vert, n_branch,
        parent_vertices_undeform, parent_vertices_undeform,
        child1_vertices_undeform, child1_vertices_undeform,
        child2_vertices_undeform, child2_vertices_undeform,
        bdlo5=bdlo5,
    )

    # Same coord transform as DEFT_train.py for non-bdlo5 case:
    # (x,y,z) -> (-z,-x,y)
    b_DLOs_vertices_undeform_transform = torch.zeros_like(b_DLOs_vertices_undeform)
    b_DLOs_vertices_undeform_transform[:, :, :, 0] = -b_DLOs_vertices_undeform[:, :, :, 2]
    b_DLOs_vertices_undeform_transform[:, :, :, 1] = -b_DLOs_vertices_undeform[:, :, :, 0]
    b_DLOs_vertices_undeform_transform[:, :, :, 2] = b_DLOs_vertices_undeform[:, :, :, 1]
    b_undeformed_vert = b_DLOs_vertices_undeform_transform[0].view(n_branch, -1, 3)

    index_selection1, index_selection2, parent_MOI_index1, parent_MOI_index2 = index_init(
        rigid_body_coupling_index, n_branch
    )
    clamped_index, parent_theta_clamp, child1_theta_clamp, child2_theta_clamp = clamp_index(
        batch, parent_clamped_selection, child1_clamped_selection, child2_clamped_selection,
        n_branch, n_parent_vertices, clamp_parent, clamp_child1, clamp_child2,
    )

    sim = DEFT_sim(
        batch=batch, n_branch=n_branch, n_vert=n_vert, cs_n_vert=cs_n_vert,
        b_init_n_vert=b_undeformed_vert, n_edge=n_edge,
        b_undeformed_vert=b_undeformed_vert,
        b_DLO_mass=b_DLO_mass, parent_DLO_MOI=parent_MOI, children_DLO_MOI=children_MOI,
        device=device, clamped_index=clamped_index,
        rigid_body_coupling_index=rigid_body_coupling_index,
        parent_MOI_index1=parent_MOI_index1, parent_MOI_index2=parent_MOI_index2,
        parent_clamped_selection=parent_clamped_selection,
        child1_clamped_selection=child1_clamped_selection,
        child2_clamped_selection=child2_clamped_selection,
        clamp_parent=clamp_parent, clamp_child1=clamp_child1, clamp_child2=clamp_child2,
        index_selection1=index_selection1, index_selection2=index_selection2,
        bend_stiffness_parent=bend_stiffness_parent,
        bend_stiffness_child1=bend_stiffness_child1,
        bend_stiffness_child2=bend_stiffness_child2,
        twist_stiffness=twist_stiffness, damping=damping, learning_weight=learning_weight,
        use_orientation_constraints=True, use_attachment_constraints=True, bdlo5=bdlo5,
    )
    sim.eval()
    return sim, parent_theta_clamp, child1_theta_clamp, child2_theta_clamp


# ---------------------------------------------------------------------------
# Load real input frames (uses Eval_DEFTData-style loading)
# ---------------------------------------------------------------------------
def load_real_input(path, n_frames_use, total_time=500):
    verts = torch.tensor(pd.read_pickle(r'%s' % path)).view(3, total_time, -1).permute(1, 2, 0)
    parent_vertices = verts[:, :n_parent_vertices]
    child1_vertices = verts[:, n_parent_vertices: n_parent_vertices + n_child1_vertices - 1]
    child2_vertices = verts[:, n_parent_vertices + n_child1_vertices - 1:]
    BDLO_vert_no_trans = construct_BDLOs_data(
        total_time, rigid_body_coupling_index,
        n_parent_vertices, cs_n_vert,
        n_branch, parent_vertices, child1_vertices, child2_vertices,
        bdlo5=bdlo5,
    )
    BDLO_vert = torch.zeros_like(BDLO_vert_no_trans)
    BDLO_vert[:, :, :, 0] = -BDLO_vert_no_trans[:, :, :, 2]
    BDLO_vert[:, :, :, 1] = -BDLO_vert_no_trans[:, :, :, 0]
    BDLO_vert[:, :, :, 2] = BDLO_vert_no_trans[:, :, :, 1]
    return BDLO_vert[:n_frames_use]


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------
def diff_stats(a, b):
    d = (a - b).abs()
    return float(d.max().item()), float(d.pow(2).mean().sqrt().item())


def per_step_diff(a, b):
    out = []
    for t in range(a.shape[0]):
        out.append(diff_stats(a[t], b[t]))
    return out


def run_iterative_predict(sim, parent_clamp, child1_clamp, child2_clamp,
                          BDLO_vert, time_horizon, inference_1_batch):
    clamped_positions = torch.zeros(1, time_horizon, n_branch, n_vert, 3, dtype=torch.float64)
    if clamp_parent:
        clamped_positions[0, :, 0, parent_clamped_selection] = BDLO_vert[2:2 + time_horizon, 0, parent_clamped_selection]
    curr_traj = BDLO_vert[1:2].unsqueeze(0)
    prev_traj = BDLO_vert[0:1].unsqueeze(0)
    with torch.no_grad():
        predicted_vertices, _ = sim.iterative_predict(
            time_horizon,
            curr_traj, prev_traj,
            clamped_positions,
            dt,
            parent_clamp, child1_clamp, child2_clamp,
            inference_1_batch, vis_type="DEFT_1", vis=False,
        )
    return predicted_vertices[0]


def run_iterative_sim(sim, parent_clamp, child1_clamp, child2_clamp,
                      BDLO_vert, time_horizon, inference_1_batch):
    curr_traj = BDLO_vert[1:1 + time_horizon].unsqueeze(0)
    prev_traj = BDLO_vert[0:time_horizon].unsqueeze(0)
    target_traj = BDLO_vert[2:2 + time_horizon].unsqueeze(0)
    loss_func = torch.nn.MSELoss()
    with torch.no_grad():
        traj_loss, _ = sim.iterative_sim(
            time_horizon,
            curr_traj, prev_traj, target_traj,
            loss_func, dt,
            parent_clamp, child1_clamp, child2_clamp,
            inference_1_batch, vis_type="DEFT_1", vis=False,
        )
    return float(traj_loss.item())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
INPUT_PKL = "dataset/BDLO1/eval/BDLO_data_hand_and_panda_track1234_block5_frames4189to4688.pkl"
TIME_HORIZON = 100

print("Loading input frames from", INPUT_PKL)
BDLO_vert = load_real_input(INPUT_PKL, n_frames_use=TIME_HORIZON + 4)
print(f"  loaded shape: {tuple(BDLO_vert.shape)}\n")

# ---- iterative_predict comparison ----
print(f"[BDLO1 iterative_predict]  horizon={TIME_HORIZON}")
print("  building NUMBA sim...")
torch.manual_seed(1); np.random.seed(1)
sim_n, p_n, c1_n, c2_n = build_sim()
pred_numba = run_iterative_predict(sim_n, p_n, c1_n, c2_n, BDLO_vert, TIME_HORIZON, True)

print("  building TORCH sim...")
torch.manual_seed(1); np.random.seed(1)
sim_t, p_t, c1_t, c2_t = build_sim()
pred_torch = run_iterative_predict(sim_t, p_t, c1_t, c2_t, BDLO_vert, TIME_HORIZON, False)

print(f"  numba pred shape: {tuple(pred_numba.shape)}")
print(f"  torch pred shape: {tuple(pred_torch.shape)}")
max_d, rms_d = diff_stats(pred_numba, pred_torch)
print(f"  overall: max_abs_diff={max_d:.3e}  rms_diff={rms_d:.3e}")

per_step = per_step_diff(pred_numba, pred_torch)
print(f"  per-step (max_abs, rms) — printing every 10th step plus first 5 and last:")
for t, (m, r) in enumerate(per_step):
    if t < 5 or t == TIME_HORIZON - 1 or t % 10 == 0:
        print(f"    t={t:3d}: max={m:.3e}  rms={r:.3e}")
print()

# ---- iterative_sim comparison ----
print(f"[BDLO1 iterative_sim]  horizon={TIME_HORIZON}")
print("  running NUMBA sim...")
torch.manual_seed(1); np.random.seed(1)
sim_n2, p_n2, c1_n2, c2_n2 = build_sim()
loss_numba = run_iterative_sim(sim_n2, p_n2, c1_n2, c2_n2, BDLO_vert, TIME_HORIZON, True)

print("  running TORCH sim...")
torch.manual_seed(1); np.random.seed(1)
sim_t2, p_t2, c1_t2, c2_t2 = build_sim()
loss_torch = run_iterative_sim(sim_t2, p_t2, c1_t2, c2_t2, BDLO_vert, TIME_HORIZON, False)

print(f"  numba traj_loss = {loss_numba:.10e}")
print(f"  torch traj_loss = {loss_torch:.10e}")
print(f"  abs diff = {abs(loss_numba - loss_torch):.3e}")
print(f"  rel diff = {abs(loss_numba - loss_torch) / max(abs(loss_numba), 1e-30):.3e}")
print()

print("Done.")
