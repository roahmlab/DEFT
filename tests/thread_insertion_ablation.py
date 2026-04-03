#!/usr/bin/env python3
"""
Thread insertion ablation: run thread insertion optimization with different DEFT variants.

Supports switching between DEFT (full model), DEFORM+ (no orientation/attachment constraints),
and DEFT_ABLATION (no orientation constraints, with attachment constraints).

Usage:
    python thread_insertion_ablation.py --config 1 --height 1 --hole A --ablation all
    python thread_insertion_ablation.py --config 1 --height 1 --hole A --ablation DEFT
"""

import torch
import torch.nn as nn
import sys
import os
import numpy as np
import argparse
import time

try:
    import cyipopt
    IPOPT_AVAILABLE = True
except ImportError:
    IPOPT_AVAILABLE = False
    print("Warning: cyipopt not available. Install with: pip install cyipopt")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deft.utils.util import DEFT_initialization, construct_b_DLOs, clamp_index, index_init
from deft.core.DEFT_sim import DEFT_sim

# Reuse data loading, visualization, and problem classes from thread_insertion
from thread_insertion import (
    N_PARENT, N_CHILD1, N_CHILD2, N_BRANCH, COUPLING_INDEX,
    SIM_TIME_HORIZON, WARMUP_STEPS,
    load_mocap_config, load_hole, compute_hole_center_and_normal,
    mocap_to_branched_tensor, visualize_initial_state,
    ThreadInsertionProblem, finite_difference_gradient,
    _run_warmup, _run_single_stage,
    visualize_result, animate_result,
)

# ---- Ablation configurations ----
ABLATION_CONFIGS = {
    "DEFT": {
        "checkpoint": "../save_model/BDLO1/DEFT_middle_1_pretrained_full_model.pth",
        "use_orientation": True,
        "use_attachment": True,
    },
    "DEFORM_PLUS": {
        "checkpoint": "../save_model/BDLO1/DEFT_middle_1_DEFORM_plus.pth",
        "use_orientation": False,
        "use_attachment": False,
    },
    "DEFT_ABLATION": {
        "checkpoint": "../save_model/BDLO1/DEFT_middle_1_DEFT_ablation.pth",
        "use_orientation": False,
        "use_attachment": True,
    },
}


def setup_deft_sim(parent_clamped_selection, load_checkpoint=None,
                   use_orientation_constraints=True, use_attachment_constraints=True):
    """Setup DEFT simulation with configurable constraint flags."""
    torch.set_default_dtype(torch.float64)
    device = "cpu"

    # BDLO1 undeformed shape
    undeformed_BDLO = torch.tensor([[[-0.6790, -0.6355, -0.5595, -0.4539, -0.3688, -0.2776, -0.1857,
                                      -0.0991, 0.0102, 0.0808, 0.1357, 0.2081, 0.2404, -0.4279,
                                      -0.4880, -0.5394, -0.5559, 0.0698, 0.0991, 0.1125]],
                                    [[0.0035, -0.0066, -0.0285, -0.0349, -0.0704, -0.0663, -0.0744,
                                      -0.0957, -0.0702, -0.0592, -0.0452, -0.0236, -0.0134, -0.0813,
                                      -0.1233, -0.1875, -0.2178, -0.1044, -0.1858, -0.2165]],
                                    [[0.0108, 0.0104, 0.0083, 0.0104, 0.0083, 0.0145, 0.0133,
                                      0.0198, 0.0155, 0.0231, 0.0199, 0.0154, 0.0169, 0.0160,
                                      0.0153, 0.0090, 0.0121, 0.0205, 0.0155, 0.0148]]]).permute(1, 2, 0)

    batch = 1
    child1_clamped_selection = torch.tensor((2))
    child2_clamped_selection = torch.tensor((2))

    parent_vertices = undeformed_BDLO[:, :N_PARENT]
    child1_vertices = undeformed_BDLO[:, N_PARENT:N_PARENT + N_CHILD1 - 1]
    child2_vertices = undeformed_BDLO[:, N_PARENT + N_CHILD1 - 1:]

    b_DLO_mass, parent_MOI, children_MOI, _, _, _ = DEFT_initialization(
        parent_vertices, child1_vertices, child2_vertices, N_BRANCH, N_PARENT,
        (N_CHILD1, N_CHILD2), COUPLING_INDEX, 1.0, 10.0, (0.5, 0.5), (1, 1), 0.1
    )

    b_DLOs_vertices, _ = construct_b_DLOs(
        batch, COUPLING_INDEX, N_PARENT, (N_CHILD1, N_CHILD2), N_BRANCH,
        parent_vertices, parent_vertices, child1_vertices, child1_vertices,
        child2_vertices, child2_vertices
    )

    # Coordinate transform
    b_DLOs_transform = torch.zeros_like(b_DLOs_vertices)
    b_DLOs_transform[:, :, :, 0] = -b_DLOs_vertices[:, :, :, 2]
    b_DLOs_transform[:, :, :, 1] = -b_DLOs_vertices[:, :, :, 0]
    b_DLOs_transform[:, :, :, 2] = b_DLOs_vertices[:, :, :, 1]
    b_undeformed_vert = b_DLOs_transform[0].view(N_BRANCH, -1, 3)

    index_selection1, index_selection2, parent_MOI_index1, parent_MOI_index2 = index_init(COUPLING_INDEX, N_BRANCH)
    clamped_index, parent_theta_clamp, child1_theta_clamp, child2_theta_clamp = clamp_index(
        batch, parent_clamped_selection, child1_clamped_selection, child2_clamped_selection,
        N_BRANCH, N_PARENT, True, False, False
    )

    bend_stiffness_parent = nn.Parameter(4e-3 * torch.ones((1, 1, N_PARENT - 1), device=device))
    bend_stiffness_child1 = nn.Parameter(4e-3 * torch.ones((1, 1, N_PARENT - 1), device=device))
    bend_stiffness_child2 = nn.Parameter(4e-3 * torch.ones((1, 1, N_PARENT - 1), device=device))
    twist_stiffness = nn.Parameter(1e-4 * torch.ones((1, N_BRANCH, N_PARENT - 1), device=device))
    damping = nn.Parameter(torch.tensor((2.5, 2.5, 2.5), device=device))
    learning_weight = nn.Parameter(torch.tensor(0.02, device=device))

    deft_sim = DEFT_sim(
        batch=batch, n_branch=N_BRANCH, n_vert=N_PARENT, cs_n_vert=(N_CHILD1, N_CHILD2),
        b_init_n_vert=b_undeformed_vert, n_edge=N_PARENT - 1, b_undeformed_vert=b_undeformed_vert,
        b_DLO_mass=b_DLO_mass, parent_DLO_MOI=parent_MOI, children_DLO_MOI=children_MOI, device=device,
        clamped_index=clamped_index, rigid_body_coupling_index=COUPLING_INDEX,
        parent_MOI_index1=parent_MOI_index1, parent_MOI_index2=parent_MOI_index2,
        parent_clamped_selection=parent_clamped_selection,
        child1_clamped_selection=child1_clamped_selection, child2_clamped_selection=child2_clamped_selection,
        clamp_parent=True, clamp_child1=False, clamp_child2=False,
        index_selection1=index_selection1, index_selection2=index_selection2,
        bend_stiffness_parent=bend_stiffness_parent, bend_stiffness_child1=bend_stiffness_child1,
        bend_stiffness_child2=bend_stiffness_child2, twist_stiffness=twist_stiffness,
        damping=damping, learning_weight=learning_weight,
        use_orientation_constraints=use_orientation_constraints,
        use_attachment_constraints=use_attachment_constraints
    )

    if load_checkpoint:
        checkpoint = torch.load(load_checkpoint, map_location=device)
        deft_sim.load_state_dict(checkpoint, strict=False)
        deft_sim.DEFT_func.bend_stiffness = torch.cat(
            (deft_sim.DEFT_func.bend_stiffness_parent,
             deft_sim.DEFT_func.bend_stiffness_child1,
             deft_sim.DEFT_func.bend_stiffness_child2),
            dim=1
        )
        print(f"Loaded checkpoint from {load_checkpoint}")

    return deft_sim, parent_theta_clamp, child1_theta_clamp, child2_theta_clamp


def run_ablation_insertion(ablation_name, ablation_cfg, config_id, height_id, hole_side,
                           initial_branched, target_align, target_insert,
                           hole_pts, hole_center, hole_normal, flat_pts,
                           offset_distance, align_distance):
    """Run 2-stage thread insertion for a single ablation configuration.

    Returns dict with results, or None if IPOPT is unavailable.
    """
    print(f"\n{'='*60}")
    print(f"  ABLATION: {ablation_name}")
    print(f"  checkpoint: {ablation_cfg['checkpoint']}")
    print(f"  orientation_constraints={ablation_cfg['use_orientation']}")
    print(f"  attachment_constraints={ablation_cfg['use_attachment']}")
    print(f"{'='*60}")

    if not IPOPT_AVAILABLE:
        print("IPOPT not available, skipping.")
        return None

    parent_clamped_selection = torch.tensor((1, 2, -2, -1))
    deft_sim, parent_theta_clamp, child1_theta_clamp, child2_theta_clamp = setup_deft_sim(
        parent_clamped_selection,
        load_checkpoint=ablation_cfg["checkpoint"],
        use_orientation_constraints=ablation_cfg["use_orientation"],
        use_attachment_constraints=ablation_cfg["use_attachment"],
    )

    # Warmup
    settled_state = _run_warmup(
        deft_sim, initial_branched, parent_theta_clamp,
        child1_theta_clamp, child2_theta_clamp, parent_clamped_selection
    )

    # Stage 1: Align
    print(f"\n{'='*60}")
    print(f"  [{ablation_name}] STAGE 1: ALIGN (tip -> {align_distance}m from hole center)")
    print(f"{'='*60}")
    total_start = time.time()
    sol1, info1, traj1, state_after_align, time1 = _run_single_stage(
        deft_sim, settled_state, parent_theta_clamp,
        child1_theta_clamp, child2_theta_clamp, target_align,
        parent_clamped_selection, f"{ablation_name}-Stage1-Align", run_grad_check=True
    )

    # Stage 2: Insert
    print(f"\n{'='*60}")
    print(f"  [{ablation_name}] STAGE 2: INSERT (tip -> through hole, {offset_distance}m past)")
    print(f"{'='*60}")
    sol2, info2, traj2, state_after_insert, time2 = _run_single_stage(
        deft_sim, state_after_align, parent_theta_clamp,
        child1_theta_clamp, child2_theta_clamp, target_insert,
        parent_clamped_selection, f"{ablation_name}-Stage2-Insert", run_grad_check=False
    )
    total_elapsed = time.time() - total_start

    combined_traj = torch.cat([traj1, traj2], dim=1)

    tip_final = combined_traj[0, -1, 0, 0].numpy()
    final_dist = np.linalg.norm(tip_final - target_insert)
    print(f"\n[{ablation_name}] Final tip position: {tip_final}")
    print(f"[{ablation_name}] Final tip-to-target distance: {final_dist:.6f}")
    print(f"[{ablation_name}] Total optimization time: {total_elapsed:.2f}s")

    return {
        "combined_traj": combined_traj,
        "final_dist": final_dist,
        "tip_final": tip_final,
        "total_time": total_elapsed,
        "stage1_time": time1,
        "stage2_time": time2,
        "stage1_cost": info1["obj_val"],
        "stage2_cost": info2["obj_val"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Thread insertion ablation study")
    parser.add_argument("--config", type=int, default=1, help="Mocap config ID (1-5)")
    parser.add_argument("--height", type=int, default=1, help="Height file ID (1-4)")
    parser.add_argument("--hole", type=str, default="A", choices=["A", "B"], help="Hole side")
    parser.add_argument("--offset", type=float, default=0.05, help="Target offset behind hole (stage 2)")
    parser.add_argument("--align", type=float, default=0.05, help="Alignment distance in front of hole (stage 1)")
    parser.add_argument("--ablation", type=str, default="all",
                        choices=["DEFT", "DEFORM_PLUS", "DEFT_ABLATION", "all"],
                        help="Which ablation config to run (default: all)")
    args = parser.parse_args()

    torch.set_default_dtype(torch.float64)
    out_dir = os.path.join(os.path.dirname(__file__), "thread_insertion_output")
    os.makedirs(out_dir, exist_ok=True)

    # Select ablation configs
    if args.ablation == "all":
        ablations = ABLATION_CONFIGS
    else:
        ablations = {args.ablation: ABLATION_CONFIGS[args.ablation]}

    # ---- Shared data loading (same across all ablations) ----
    print(f"\n{'='*60}")
    print(f"Thread insertion ablation: Config {args.config} -> Hole H{args.height}{args.hole}")
    print(f"  align_distance={args.align}, offset_distance={args.offset}")
    print(f"  ablations: {list(ablations.keys())}")
    print(f"{'='*60}")

    flat_pts = load_mocap_config(args.config)
    hole_pts = load_hole(args.height, args.hole)

    tip_position = flat_pts[0]
    hole_center, hole_normal = compute_hole_center_and_normal(hole_pts, tip_position)
    print(f"Hole center: {hole_center}")
    print(f"Hole normal: {hole_normal} (points toward tip)")
    print(f"Tip (v0): {tip_position}")

    target_align = hole_center + hole_normal * args.align
    target_insert = hole_center - hole_normal * args.offset
    target_insert[2] -= 0.02

    print(f"Stage 1 target (align): {target_align}")
    print(f"Stage 2 target (insert): {target_insert}")

    initial_branched = mocap_to_branched_tensor(flat_pts)
    tip_initial = initial_branched[0, 0, 0].numpy()
    print(f"Tip (vertex 0) initial: {tip_initial}")
    print(f"Initial tip-to-align distance: {np.linalg.norm(tip_initial - target_align):.4f}")
    print(f"Initial tip-to-insert distance: {np.linalg.norm(tip_initial - target_insert):.4f}")

    visualize_initial_state(initial_branched, target_insert, hole_pts, hole_center, hole_normal,
                            args.config, args.height, args.hole)

    # ---- Run each ablation ----
    results = {}
    for name, cfg in ablations.items():
        result = run_ablation_insertion(
            name, cfg, args.config, args.height, args.hole,
            initial_branched, target_align, target_insert,
            hole_pts, hole_center, hole_normal, flat_pts,
            args.offset, args.align,
        )
        if result is not None:
            results[name] = result

            # Save per-ablation outputs
            tag = f"config{args.config}_H{args.height}{args.hole}_{name}"
            fig_path = os.path.join(out_dir, f"{tag}.png")
            visualize_result(result["combined_traj"], initial_branched, target_insert,
                             hole_pts, flat_pts, args.config, args.height, args.hole,
                             save_path=fig_path)
            anim_path = os.path.join(out_dir, f"{tag}.mp4")
            animate_result(result["combined_traj"], initial_branched, target_insert,
                           hole_pts, args.config, args.height, args.hole,
                           save_path=anim_path)

    # ---- Summary ----
    if results:
        print(f"\n{'='*60}")
        print("  ABLATION SUMMARY")
        print(f"{'='*60}")
        print(f"  {'Config':<20s} {'Tip Dist':>10s} {'Stage1 Cost':>12s} {'Stage2 Cost':>12s} {'Time (s)':>10s}")
        print(f"  {'-'*20} {'-'*10} {'-'*12} {'-'*12} {'-'*10}")
        for name, r in results.items():
            print(f"  {name:<20s} {r['final_dist']:10.6f} {r['stage1_cost']:12.6f} "
                  f"{r['stage2_cost']:12.6f} {r['total_time']:10.2f}")
