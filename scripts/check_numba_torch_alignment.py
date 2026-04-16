"""
Numerical alignment check: numba (inference_1_batch=True) vs pytorch
(inference_1_batch=False) paths for both `iterative_predict` and
`iterative_sim` on BDLO5.

We build identical sims, feed them identical inputs, and report per-step
max-abs and RMS differences over the predicted vertices for a short
horizon (long enough to exercise the constraint loop, short enough to
finish quickly).
"""
import os
import sys
import pickle

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn

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
# BDLO5 problem setup (matches visualizations/predict_bdlo5_biased.py)
# ---------------------------------------------------------------------------
device = "cpu"
n_branch = 3
dt = 0.01

n_parent_vertices = 12
n_child1_vertices = 4
n_child2_vertices = 4
cs_n_vert = (n_child1_vertices, n_child2_vertices)
n_vert = n_parent_vertices
n_edge = n_vert - 1
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

parent_vertices_undeform = undeformed_BDLO[:, :n_parent_vertices]
child1_vertices_undeform = undeformed_BDLO[:, n_parent_vertices: n_parent_vertices + n_child1_vertices - 1]
child2_vertices_undeform = undeformed_BDLO[:, n_parent_vertices + n_child1_vertices - 1:]

parent_mass_scale = 1e6
parent_moment_scale = 1e6
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
    """Construct a fresh BDLO5 sim with the predict-script overrides applied."""
    bend_stiffness_parent = nn.Parameter(2e-3 * torch.ones((1, 1, n_edge), device=device, dtype=torch.float64))
    bend_stiffness_child1 = nn.Parameter(2e-3 * torch.ones((1, 1, n_edge), device=device, dtype=torch.float64))
    bend_stiffness_child2 = nn.Parameter(3e-3 * torch.ones((1, 1, n_edge), device=device, dtype=torch.float64))
    twist_stiffness = nn.Parameter(1e-4 * torch.ones((1, n_branch, n_edge), device=device, dtype=torch.float64))
    damping = nn.Parameter(torch.tensor((3., 3., 3.), device=device, dtype=torch.float64))
    learning_weight = nn.Parameter(torch.tensor(0.1, device=device, dtype=torch.float64))

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
    step1 = torch.zeros_like(b_DLOs_vertices_undeform)
    step1[:, :, :, 0] = -b_DLOs_vertices_undeform[:, :, :, 2]
    step1[:, :, :, 1] = -b_DLOs_vertices_undeform[:, :, :, 0]
    step1[:, :, :, 2] = b_DLOs_vertices_undeform[:, :, :, 1]
    b_DLOs_vertices_undeform_transform = torch.zeros_like(step1)
    b_DLOs_vertices_undeform_transform[:, :, :, 0] = -step1[:, :, :, 0]
    b_DLOs_vertices_undeform_transform[:, :, :, 1] = step1[:, :, :, 2]
    b_DLOs_vertices_undeform_transform[:, :, :, 2] = -step1[:, :, :, 1]
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
        use_child2_orientation=True,
    )
    sim.eval()
    sim.child2_coupling_mass_scale = torch.tensor(
        [[[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
          [[-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]]]],
        dtype=torch.float64,
    )
    return sim, parent_theta_clamp, child1_theta_clamp, child2_theta_clamp


# ---------------------------------------------------------------------------
# Load real input frames (so the dynamics are realistic)
# ---------------------------------------------------------------------------
def load_real_input(path, n_frames_use):
    with open(path, "rb") as f:
        data = pickle.load(f)
    n_raw = 18
    n_frames = 500
    x = np.array(data[0]).reshape(n_frames, n_raw)
    y = np.array(data[1]).reshape(n_frames, n_raw)
    z = np.array(data[2]).reshape(n_frames, n_raw)
    raw = torch.tensor(np.stack([x, y, z], axis=-1), dtype=torch.float64)
    raw = raw[:n_frames_use]
    n_child1_raw = n_child1_vertices - 1
    n_child2_raw = n_child2_vertices - 1
    parent_v = raw[:, :n_parent_vertices]
    child1_v = raw[:, n_parent_vertices: n_parent_vertices + n_child1_raw]
    child2_v = raw[:, n_parent_vertices + n_child1_raw:]
    vert_no_trans = construct_BDLOs_data(
        n_frames_use, rigid_body_coupling_index,
        n_parent_vertices, cs_n_vert,
        n_branch, parent_v, child1_v, child2_v,
        bdlo5=bdlo5,
    )
    vert = torch.zeros_like(vert_no_trans)
    vert[:, :, :, 0] = -vert_no_trans[:, :, :, 2]
    vert[:, :, :, 1] = -vert_no_trans[:, :, :, 0]
    vert[:, :, :, 2] = vert_no_trans[:, :, :, 1]
    return vert  # (n_frames_use, n_branch, n_vert, 3)


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------
def diff_stats(a, b):
    d = (a - b).abs()
    return float(d.max().item()), float(d.pow(2).mean().sqrt().item())


def per_step_diff(a, b):
    """a, b: (T, n_branch, n_vert, 3) — return list of (max, rms) per step."""
    out = []
    for t in range(a.shape[0]):
        out.append(diff_stats(a[t], b[t]))
    return out


# ---------------------------------------------------------------------------
# Run the comparison
# ---------------------------------------------------------------------------
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
            inference_1_batch, vis_type="DEFT_5", vis=False,
        )
    return predicted_vertices[0]  # (T, n_branch, n_vert, 3)


def run_iterative_sim(sim, parent_clamp, child1_clamp, child2_clamp,
                      BDLO_vert, time_horizon, inference_1_batch):
    """iterative_sim takes (curr, prev, target) trajectories."""
    curr_traj = BDLO_vert[1:1 + time_horizon].unsqueeze(0)   # (1, T, n_branch, n_vert, 3)
    prev_traj = BDLO_vert[0:time_horizon].unsqueeze(0)
    target_traj = BDLO_vert[2:2 + time_horizon].unsqueeze(0)
    loss_func = torch.nn.MSELoss()
    with torch.no_grad():
        traj_loss, _ = sim.iterative_sim(
            time_horizon,
            curr_traj, prev_traj, target_traj,
            loss_func, dt,
            parent_clamp, child1_clamp, child2_clamp,
            inference_1_batch, vis_type="DEFT_5", vis=False,
        )
    return float(traj_loss.item())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
INPUT_PKL = "dataset/BDLO5/eval/kv1_pd123_block5_frames3400to3899.pkl"
TIME_HORIZON = 100   # 100 steps to characterize drift growth

print("Loading input frames from", INPUT_PKL)
BDLO_vert = load_real_input(INPUT_PKL, n_frames_use=TIME_HORIZON + 4)
print(f"  loaded shape: {tuple(BDLO_vert.shape)}\n")

# ---- iterative_predict comparison ----
# IMPORTANT: re-seed before each build_sim() so both sims get IDENTICAL random
# GNN weights. Without this, the two sims would have different parameters and
# the comparison would mostly measure that, not the path divergence.
print(f"[iterative_predict]  horizon={TIME_HORIZON}")
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

# ---- iterative_sim comparison (returns scalar loss only) ----
print(f"[iterative_sim]  horizon={TIME_HORIZON}")
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
