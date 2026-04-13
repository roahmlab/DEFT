"""
Numerical alignment test: run DEFT simulation for 5 steps with both
  inference_1_batch=False (torch constraints) and
  inference_1_batch=True  (numba constraints)
then compare the output vertices.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Change to scripts/ dir so relative paths in dataset loaders work
os.chdir(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

torch.set_default_dtype(torch.float64)

from deft.utils.util import (
    DEFT_initialization, construct_b_DLOs, clamp_index, index_init,
    Eval_DEFTData
)
from deft.core.DEFT_sim import DEFT_sim


def main():
    device = "cpu"
    n_branch = 3
    BDLO_type = 5
    clamp_type = "ends"
    clamp_parent = True
    clamp_child1 = False
    clamp_child2 = False
    time_horizon = 5  # run 5 steps

    # ---- BDLO5 config (copied from DEFT_train.py) ----
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

    n_parent_vertices = 12
    n_child1_vertices = 4
    n_child2_vertices = 4
    eval_set_number = 26
    total_time = 500  # match DEFT_train default

    cs_n_vert = (n_child1_vertices, n_child2_vertices)
    n_vert = n_parent_vertices
    n_edge = n_vert - 1

    bend_stiffness_parent = nn.Parameter(2e-3 * torch.ones((1, 1, n_edge), device=device, dtype=torch.float64))
    bend_stiffness_child1 = nn.Parameter(2e-3 * torch.ones((1, 1, n_edge), device=device, dtype=torch.float64))
    bend_stiffness_child2 = nn.Parameter(2e-3 * torch.ones((1, 1, n_edge), device=device, dtype=torch.float64))
    twist_stiffness = nn.Parameter(1e-4 * torch.ones((1, n_branch, n_edge), device=device, dtype=torch.float64))
    damping = nn.Parameter(torch.tensor((3., 3., 3.), device=device, dtype=torch.float64))
    learning_weight = nn.Parameter(torch.tensor(0.1, device=device, dtype=torch.float64))
    rigid_body_coupling_index = [5, 1]

    parent_mass_scale = 1.
    parent_moment_scale = 10.
    moment_ratio = 0.1
    children_moment_scale = (0.5, 0.5)
    children_mass_scale = (1, 1)
    bdlo5 = True

    parent_clamped_selection = torch.tensor((0, 1, -2, -1))
    child1_clamped_selection = torch.tensor((2))
    child2_clamped_selection = torch.tensor((2))

    # ---- Initialize ----
    batch = 1  # single batch for comparison
    eval_batch = 1

    n_children_vertices = (n_child1_vertices, n_child2_vertices)
    parent_vertices_undeform = undeformed_BDLO[:, :n_parent_vertices]
    child1_vertices_undeform = undeformed_BDLO[:, n_parent_vertices: n_parent_vertices + n_children_vertices[0] - 1]
    child2_vertices_undeform = undeformed_BDLO[:, n_parent_vertices + n_children_vertices[0] - 1:]

    b_DLO_mass, parent_MOI, children_MOI, parent_rod_orientation, children_rod_orientation, b_nominal_length = DEFT_initialization(
        parent_vertices_undeform,
        child1_vertices_undeform,
        child2_vertices_undeform,
        n_branch,
        n_parent_vertices,
        cs_n_vert,
        rigid_body_coupling_index,
        parent_mass_scale,
        parent_moment_scale,
        children_moment_scale,
        children_mass_scale,
        moment_ratio,
        bdlo5=bdlo5
    )

    b_DLOs_vertices_undeform_untransform, _ = construct_b_DLOs(
        batch, rigid_body_coupling_index, n_parent_vertices, cs_n_vert, n_branch,
        parent_vertices_undeform, parent_vertices_undeform,
        child1_vertices_undeform, child1_vertices_undeform,
        child2_vertices_undeform, child2_vertices_undeform,
        bdlo5=bdlo5
    )

    # BDLO5 coordinate transform
    step1 = torch.zeros_like(b_DLOs_vertices_undeform_untransform)
    step1[:, :, :, 0] = -b_DLOs_vertices_undeform_untransform[:, :, :, 2]
    step1[:, :, :, 1] = -b_DLOs_vertices_undeform_untransform[:, :, :, 0]
    step1[:, :, :, 2] = b_DLOs_vertices_undeform_untransform[:, :, :, 1]
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
        n_branch, n_parent_vertices, clamp_parent, clamp_child1, clamp_child2
    )

    dt = 0.01

    # Create two identical DEFT_sim instances
    sim_kwargs = dict(
        batch=batch, n_branch=n_branch, n_vert=n_vert, cs_n_vert=cs_n_vert,
        b_init_n_vert=b_undeformed_vert, n_edge=n_vert - 1,
        b_undeformed_vert=b_undeformed_vert, b_DLO_mass=b_DLO_mass,
        parent_DLO_MOI=parent_MOI, children_DLO_MOI=children_MOI,
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
        twist_stiffness=twist_stiffness, damping=damping,
        learning_weight=learning_weight,
        use_orientation_constraints=True, use_attachment_constraints=True,
        bdlo5=bdlo5
    )

    sim_torch = DEFT_sim(**sim_kwargs)
    sim_numba = DEFT_sim(**sim_kwargs)

    # Load pretrained model
    pretrained_path = "../save_model/BDLO5/DEFT_ends_5_pretrained_full_model.pth"
    if os.path.exists(pretrained_path):
        state = torch.load(pretrained_path)
        sim_torch.load_state_dict(state)
        sim_numba.load_state_dict(state)
        print(f"Loaded model from {pretrained_path}")
    else:
        print(f"WARNING: {pretrained_path} not found, using default init")

    # Load eval data — only need 5 steps (+ 2 for prev/curr offset)
    eval_time_horizon = time_horizon
    eval_dataset = Eval_DEFTData(
        BDLO_type, n_parent_vertices, n_children_vertices, n_branch,
        rigid_body_coupling_index, eval_set_number, total_time,
        eval_time_horizon,
        device, bdlo5=bdlo5
    )
    eval_loader = DataLoader(eval_dataset, batch_size=batch, shuffle=False, drop_last=True)

    # Grab one batch
    data_loaded = False
    for eval_data in eval_loader:
        previous_b_DLOs_vertices_traj, b_DLOs_vertices_traj, target_b_DLOs_vertices_traj = eval_data
        data_loaded = True
        break

    if not data_loaded:
        print("ERROR: No eval data loaded! Check dataset path.")
        return 1.0

    loss_func = torch.nn.MSELoss()

    # Run step-by-step: 1 step at a time so we can compare per-step
    time_horizon_run = min(time_horizon, eval_time_horizon)
    print(f"\nComparing torch vs numba for {time_horizon_run} steps...")

    # ---- Run torch path (inference_1_batch=False) ----
    print("  Running TORCH path...")
    with torch.no_grad():
        traj_loss_torch, total_loss_torch = sim_torch.iterative_sim(
            time_horizon_run,
            b_DLOs_vertices_traj.clone(),
            previous_b_DLOs_vertices_traj.clone(),
            target_b_DLOs_vertices_traj.clone(),
            loss_func, dt,
            parent_theta_clamp, child1_theta_clamp, child2_theta_clamp,
            inference_1_batch=False,
            vis_type="DEFT_5", vis=False
        )

    # ---- Run numba path (inference_1_batch=True) ----
    print("  Running NUMBA path...")
    with torch.no_grad():
        traj_loss_numba, total_loss_numba = sim_numba.iterative_sim(
            time_horizon_run,
            b_DLOs_vertices_traj.clone(),
            previous_b_DLOs_vertices_traj.clone(),
            target_b_DLOs_vertices_traj.clone(),
            loss_func, dt,
            parent_theta_clamp, child1_theta_clamp, child2_theta_clamp,
            inference_1_batch=True,
            vis_type="DEFT_5", vis=False
        )

    # ---- Compare ----
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    tl_torch = traj_loss_torch.item()
    tl_numba = traj_loss_numba.item()
    tl_diff = abs(tl_torch - tl_numba)

    tot_torch = total_loss_torch.item()
    tot_numba = total_loss_numba.item()
    tot_diff = abs(tot_torch - tot_numba)

    print(f"\nTrajectory loss (position MSE summed over {time_horizon_run} steps):")
    print(f"  Torch: {tl_torch:.10f}")
    print(f"  Numba: {tl_numba:.10f}")
    print(f"  Diff:  {tl_diff:.2e}")

    print(f"\nTotal loss (pos + vel):")
    print(f"  Torch: {tot_torch:.10f}")
    print(f"  Numba: {tot_numba:.10f}")
    print(f"  Diff:  {tot_diff:.2e}")

    # Also compare per-sample RMSE if available
    rmse_torch = sim_torch.last_sample_traj_rmse
    rmse_numba = sim_numba.last_sample_traj_rmse
    if rmse_torch is not None and rmse_numba is not None:
        rmse_diff = (rmse_torch - rmse_numba).abs()
        print(f"\nPer-sample RMSE diff: max={rmse_diff.max().item():.2e}, mean={rmse_diff.mean().item():.2e}")

    max_diff = max(tl_diff, tot_diff)
    rel_diff = tl_diff / max(tl_torch, 1e-20)

    print(f"\nRelative traj_loss difference: {rel_diff:.2e}")

    if rel_diff < 1e-10:
        print("\nPASS — torch and numba are numerically aligned (relative diff < 1e-10)")
    elif rel_diff < 1e-6:
        print(f"\nACCEPTABLE — small floating-point differences (relative diff {rel_diff:.2e})")
    else:
        print(f"\nFAIL — significant differences detected (relative diff {rel_diff:.2e})")

    return rel_diff


if __name__ == "__main__":
    max_diff = main()
    sys.exit(0 if max_diff < 1e-6 else 1)
