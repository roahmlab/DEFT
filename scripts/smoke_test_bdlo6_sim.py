"""
Stage 3a smoke test: build a DEFT_sim instance for BDLO6 (4 branches) without
running any forward pass, and verify the constructor returns without error.

This wires up:
    load_bdlo6_undeformed -> [4, 12, 3] padded layout
    DEFT_initialization_BDLO6 -> mass, MOI, etc.
    construct_b_DLOs_BDLO6 -> batched undeformed
    clamp_index, index_init, DEFT_sim(bdlo6=True, use_orientation_constraints=False)

Run:
    python3 smoke_test_bdlo6_sim.py
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from deft.utils.util import (
    load_bdlo6_undeformed,
    DEFT_initialization_BDLO6,
    construct_b_DLOs_BDLO6,
    clamp_index, index_init,
    BDLO6_RIGID_BODY_COUPLING_INDEX,
    _BDLO6_N_C1, _BDLO6_N_C2, _BDLO6_N_C3,
)
from deft.core.DEFT_sim import DEFT_sim


# ---- BDLO6 hyperparameters (mirroring the BDLO5 block in DEFT_train.py) ----
n_parent_vertices = 12
n_child1_vertices = _BDLO6_N_C1   # 5
n_child2_vertices = _BDLO6_N_C2   # 3
n_child3_vertices = _BDLO6_N_C3   # 4
n_branch = 4
cs_n_vert = (n_child1_vertices, n_child2_vertices, n_child3_vertices)
n_vert = n_parent_vertices
n_edge = n_vert - 1

rigid_body_coupling_index = list(BDLO6_RIGID_BODY_COUPLING_INDEX)   # [2, 7, 7]

parent_mass_scale     = 1.
parent_moment_scale   = 10.
moment_ratio          = 0.1
children_moment_scale = (0.5, 0.5, 0.5)
children_mass_scale   = (1, 1, 1)

device = "cpu"
torch.set_default_dtype(torch.float64)

# Stiffness and damping placeholders (one per branch where applicable)
bend_stiffness_parent = nn.Parameter(2e-3 * torch.ones((1, 1, n_edge), device=device))
bend_stiffness_child1 = nn.Parameter(2e-3 * torch.ones((1, 1, n_edge), device=device))
bend_stiffness_child2 = nn.Parameter(2e-3 * torch.ones((1, 1, n_edge), device=device))
bend_stiffness_child3 = nn.Parameter(2e-3 * torch.ones((1, 1, n_edge), device=device))
twist_stiffness       = nn.Parameter(1e-4 * torch.ones((1, n_branch, n_edge), device=device))
damping               = nn.Parameter(torch.tensor((3., 3., 3., 3.), device=device))
learning_weight       = nn.Parameter(torch.tensor(0.0, device=device))

# Clamp selections (placeholder — we just want the constructor to run)
parent_clamped_selection = torch.tensor((0, 1, -2, -1))
child1_clamped_selection = torch.tensor((2,))
child2_clamped_selection = torch.tensor((2,))
child3_clamped_selection = torch.tensor((2,))

# ---- Build the BDLO6 undeformed pose and per-branch tensors ----
print("Loading BDLO6 undeformed pose...")
b_undeformed_vert = load_bdlo6_undeformed("../dataset/BDLO6_undeformed.pkl")
print(f"  shape: {tuple(b_undeformed_vert.shape)}  (expect (4, 12, 3))")

print("DEFT_initialization_BDLO6...")
b_DLO_mass, parent_MOI, children_MOI, parent_rod_orientation, children_rod_orientation, b_nominal_length = \
    DEFT_initialization_BDLO6(
        b_undeformed_vert,
        parent_mass_scale, parent_moment_scale,
        children_moment_scale, children_mass_scale,
        moment_ratio,
    )
print(f"  b_DLO_mass: {tuple(b_DLO_mass.shape)}")
print(f"  parent_MOI: {tuple(parent_MOI.shape)}  (expect (6, 3) — 2 entries per coupling × 3 couplings)")
print(f"  children_MOI: {tuple(children_MOI.shape)}  (expect (3, 3))")
print(f"  b_nominal_length: {tuple(b_nominal_length.shape)}")

batch = 1
print(f"\nconstruct_b_DLOs_BDLO6(batch={batch})...")
b_DLOs_vertices_undeform, prev_b_DLOs_vertices_undeform = construct_b_DLOs_BDLO6(batch, b_undeformed_vert)
print(f"  b_DLOs_vertices_undeform: {tuple(b_DLOs_vertices_undeform.shape)}  (expect ({batch}, 4, 12, 3))")

# ---- Index init / clamp ----
print("\nindex_init / clamp_index...")
index_selection1, index_selection2, parent_MOI_index1, parent_MOI_index2 = index_init(
    rigid_body_coupling_index, n_branch)
print(f"  index_selection1: {index_selection1.tolist()}")
print(f"  index_selection2: {index_selection2.tolist()}")
print(f"  parent_MOI_index1: {parent_MOI_index1.tolist()}")
print(f"  parent_MOI_index2: {parent_MOI_index2.tolist()}")

clamped_index, parent_theta_clamp, child1_theta_clamp, child2_theta_clamp = clamp_index(
    batch,
    parent_clamped_selection, child1_clamped_selection, child2_clamped_selection,
    n_branch, n_parent_vertices,
    True, False, False,
)
print(f"  clamped_index: {tuple(clamped_index.shape)}")

# ---- Build DEFT_sim with bdlo6=True ----
print("\nBuilding DEFT_sim(bdlo6=True, use_orientation_constraints=False)...")
b_undeformed_vert_for_sim = b_DLOs_vertices_undeform[0].view(n_branch, -1, 3)
sim = DEFT_sim(
    batch=batch,
    n_branch=n_branch,
    n_vert=n_vert,
    cs_n_vert=cs_n_vert,
    b_init_n_vert=b_undeformed_vert_for_sim,
    n_edge=n_edge,
    b_undeformed_vert=b_undeformed_vert_for_sim,
    b_DLO_mass=b_DLO_mass,
    parent_DLO_MOI=parent_MOI,
    children_DLO_MOI=children_MOI,
    device=device,
    clamped_index=clamped_index,
    rigid_body_coupling_index=rigid_body_coupling_index,
    parent_MOI_index1=parent_MOI_index1,
    parent_MOI_index2=parent_MOI_index2,
    parent_clamped_selection=parent_clamped_selection,
    child1_clamped_selection=child1_clamped_selection,
    child2_clamped_selection=child2_clamped_selection,
    clamp_parent=True, clamp_child1=False, clamp_child2=False,
    index_selection1=index_selection1,
    index_selection2=index_selection2,
    bend_stiffness_parent=bend_stiffness_parent,
    bend_stiffness_child1=bend_stiffness_child1,
    bend_stiffness_child2=bend_stiffness_child2,
    bend_stiffness_child3=bend_stiffness_child3,
    twist_stiffness=twist_stiffness,
    damping=damping,
    learning_weight=learning_weight,
    use_orientation_constraints=False,
    use_attachment_constraints=True,
    bdlo5=False,
    bdlo6=True,
)
print("DEFT_sim constructed OK.")
print(f"  sim.batch={sim.batch}, sim.n_branch={sim.n_branch}, sim.n_vert={sim.n_vert}")
print(f"  sim.bdlo6={sim.bdlo6}")
print(f"  sim.selected_parent_index={sim.selected_parent_index.tolist()}")
print(f"  sim.selected_child1_index={sim.selected_child1_index.tolist()}")
print(f"  sim.selected_child2_index={sim.selected_child2_index.tolist()}")
print(f"  sim.selected_child3_index={sim.selected_child3_index.tolist()}")
print(f"  sim.coupling_mass_scale.shape={tuple(sim.coupling_mass_scale.shape)}")
print("\nSTAGE 3A SMOKE TEST PASSED")

# ----------------------------------------------------------------------------
# Stage 3b smoke test: try one iterative_sim forward pass for BDLO6.
# ----------------------------------------------------------------------------
from deft.utils.util import Eval_DEFTData_BDLO6
print("\n=== Stage 3b smoke: iterative_sim forward ===")
print("Loading Eval_DEFTData_BDLO6...")
eval_ds = Eval_DEFTData_BDLO6(total_time=500, eval_time_horizon=5, device="cpu")
print(f"  num eval samples: {len(eval_ds)}")
prev, curr, tgt = eval_ds[0]
print(f"  prev shape: {tuple(prev.shape)}  (expect (5, 4, 12, 3))")

# Add a batch dim of 1 (DEFT_sim expects [batch, ...] input)
prev_b   = prev.unsqueeze(0)
curr_b   = curr.unsqueeze(0)
tgt_b    = tgt.unsqueeze(0)
print(f"  with batch dim: {tuple(curr_b.shape)}")

loss_func = torch.nn.MSELoss()
print("\nCalling sim.iterative_sim(time_horizon=5) under no_grad...")
with torch.no_grad():
    try:
        traj_loss, total_loss = sim.iterative_sim(
            5,
            curr_b,
            prev_b,
            tgt_b,
            loss_func,
            0.01,
            parent_theta_clamp,
            child1_theta_clamp,
            child2_theta_clamp,
            False,             # inference_1_batch
            vis_type="bdlo6_smoke",
            vis=False,
        )
        import numpy as np
        rmse = float(np.sqrt(traj_loss.cpu().numpy() / 5))
        print(f"  iterative_sim returned. traj_loss={traj_loss.item():.6f}, RMSE={rmse:.6f}")
        print("\nSTAGE 3B SMOKE TEST PASSED")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\nSTAGE 3B SMOKE TEST FAILED: {type(e).__name__}: {e}")
