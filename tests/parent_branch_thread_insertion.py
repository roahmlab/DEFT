#!/usr/bin/env python3
"""
Thread insertion optimization: control vertex 2 to guide vertex 0 (tip) through a target hole.
Uses IPOPT with DEFT simulation as the forward model.
"""

import torch
import torch.nn as nn
import sys
import os
import pickle
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import argparse
import time
import warnings
warnings.filterwarnings(
    "ignore",
    message="Vectorization is off by default",
    module="theseus.optimizer.optimizer",
)

try:
    import cyipopt
    IPOPT_AVAILABLE = True
except ImportError:
    IPOPT_AVAILABLE = False
    print("Warning: cyipopt not available. Install with: pip install cyipopt")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deft.utils.util import DEFT_initialization, construct_b_DLOs, construct_BDLOs_data, clamp_index, index_init
from deft.core.DEFT_sim import DEFT_sim

# ---- Configuration ----
SIM_TIME_HORIZON = 100
WARMUP_STEPS = 500  # 1 second at dt=0.01 — hold static to damp initial vibrations
DATASET_DIR = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'BDLO_main_branch_thread_insertion')

# BDLO1 parameters
N_PARENT = 13
N_CHILD1 = 5
N_CHILD2 = 4
N_BRANCH = 3
COUPLING_INDEX = [4, 8]


def load_mocap_config(config_id):
    """Load a mocap initial configuration (20 flat vertices in raw coords)."""
    fpath = os.path.join(DATASET_DIR, f'mocap_in_base_config{config_id}.pkl')
    with open(fpath, 'rb') as f:
        data = pickle.load(f)
    # data: list[1] -> tuple(3) -> list[N] of float64
    pts = np.array(data).squeeze().T.reshape(-1, 20, 3)
    return pts[0]  # (20, 3) - single frame


def load_hole(height_id, hole_side='A'):
    """Load hole points (4 vertices) from height file.
    hole_side: 'A' (points 0-3, x≈0.19) or 'B' (points 4-7, x≈0.35)
    """
    fpath = os.path.join(DATASET_DIR, f'height_{height_id}_0319_in_base.pkl')
    with open(fpath, 'rb') as f:
        data = pickle.load(f)
    pts = np.array(data).squeeze().T.reshape(-1, 8, 3)
    pts = pts[0]  # (8, 3)
    if hole_side == 'A':
        return pts[:4]  # (4, 3)
    else:
        return pts[4:]  # (4, 3)


def compute_hole_center_and_normal(hole_pts, bdlo_bulk_center):
    """Compute hole center and normal (facing TOWARD BDLO bulk).
    Target = center - normal * offset puts the target on the far side of the hole from the BDLO.
    hole_pts: (4, 3) raw coordinates
    bdlo_bulk_center: (3,) center of BDLO mass for orienting normal
        is not used in this version of the script 
    """
    center = hole_pts.mean(axis=0)
    # Normal from cross product of two edges of the triangle (first 3 points)
    v1 = hole_pts[1] - hole_pts[0]
    v2 = hole_pts[2] - hole_pts[0]
    normal = np.cross(v1, v2)
    normal = normal / np.linalg.norm(normal)

    # Orient normal towards the direction with positive x
    if normal[0] < 0:
        normal = -normal
    return center, normal


def mocap_to_branched_tensor(flat_vertices_raw):
    """Convert 20 flat mocap vertices to DEFT branched format.

    The mocap_in_base data is already preprocessed into the correct coordinate frame,
    so no (-z, -x, y) transform is needed. Only the undeformed shape requires that transform.

    Input: (20, 3) already in DEFT-compatible coords
    Output: (1, n_branch, max_vert, 3)
    """
    flat = torch.tensor(flat_vertices_raw, dtype=torch.float64)
    # Split into parent / child1 / child2
    parent = flat[:N_PARENT]           # (13, 3)
    child1 = flat[N_PARENT:N_PARENT + N_CHILD1 - 1]  # (4, 3) - exclude coupling point
    child2 = flat[N_PARENT + N_CHILD1 - 1:]           # (3, 3) - exclude coupling point

    # Add time dimension for construct_BDLOs_data: (1, n_vert, 3)
    parent_t = parent.unsqueeze(0)
    child1_t = child1.unsqueeze(0)
    child2_t = child2.unsqueeze(0)

    # Build branched format: (1, n_branch, max_vert, 3)
    b_verts = construct_BDLOs_data(
        1, COUPLING_INDEX, N_PARENT, (N_CHILD1, N_CHILD2), N_BRANCH,
        parent_t, child1_t, child2_t
    )

    return b_verts  # (1, n_branch, max_vert, 3)


def setup_deft_sim(parent_clamped_selection, load_checkpoint=None):
    """Setup DEFT simulation for thread insertion (only vertex 2 clamped)."""
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
        use_orientation_constraints=True, use_attachment_constraints=True
    )

    if load_checkpoint:
        checkpoint = torch.load(load_checkpoint, map_location=device)
        deft_sim.load_state_dict(checkpoint, strict=False)
        # Recompute derived stiffness tensors (bend_stiffness is concatenated in __init__
        # and not updated by load_state_dict)
        deft_sim.DEFT_func.bend_stiffness = torch.cat(
            (deft_sim.DEFT_func.bend_stiffness_parent,
             deft_sim.DEFT_func.bend_stiffness_child1,
             deft_sim.DEFT_func.bend_stiffness_child2),
            dim=1
        )
        print(f"Loaded checkpoint from {load_checkpoint}")

    return deft_sim, parent_theta_clamp, child1_theta_clamp, child2_theta_clamp


class ThreadInsertionProblem:
    """IPOPT problem: optimize shared v1/v2 displacement (3 DOF) to guide vertex 0 through a hole.

    Clamped vertices: 1, 2 (control pair, shared displacement), 11, 12 (hard-clamped fixed).
    Like move_bdlo_ablation.py: paired vertices share displacement to maintain relative offset.
    """

    def __init__(self, deft_sim, initial_branched, parent_theta_clamp,
                 child1_theta_clamp, child2_theta_clamp, target_pos,
                 parent_clamped_selection):
        self.deft_sim = deft_sim
        self.deft_sim.eval()
        for param in self.deft_sim.parameters():
            param.requires_grad = False

        # initial_branched: (1, n_branch, max_vert, 3) in DEFT coords
        self.initial_branched = initial_branched.detach().clone()
        self.parent_theta_clamp = parent_theta_clamp
        self.child1_theta_clamp = child1_theta_clamp
        self.child2_theta_clamp = child2_theta_clamp
        self.target_pos = torch.tensor(target_pos, dtype=torch.float64)  # (3,) in DEFT coords
        self.parent_clamped_selection = parent_clamped_selection

        # Initial positions of all 4 clamped vertices: v1, v2, v11, v12
        self.initial_clamped = initial_branched[0, 0, parent_clamped_selection].detach().clone()  # (4, 3)

        self.n = 3  # 3 DOF: shared displacement for v1 & v2
        self.iteration = 0
        self.converged = False
        self.best_solution = None

    def _expand_displacements(self, displacement):
        """Expand 3 DOF to per-clamped-vertex displacements (4, 3).
        displacement: (3,) — shared displacement for v1 & v2.
        v11 & v12 get zero displacement (hard-clamped).
        """
        zero = torch.zeros(3, dtype=torch.float64)
        return torch.stack([displacement, displacement, zero, zero])  # v1, v2, v11, v12

    def _build_clamped_trajectory(self, displacement):
        """Build clamped_positions tensor.
        displacement: (3,) torch tensor — shared displacement for v1 & v2.
        Vertices 11, 12 stay fixed at their initial positions.
        """
        final_displacements = self._expand_displacements(displacement)  # (4, 3)

        batch = 1
        clamped_full = torch.zeros(batch, SIM_TIME_HORIZON, N_BRANCH, N_PARENT, 3, dtype=torch.float64)
        for t in range(SIM_TIME_HORIZON):
            alpha = t / max(SIM_TIME_HORIZON - 1, 1)
            interpolated = self.initial_clamped + alpha * final_displacements
            for i, idx in enumerate(self.parent_clamped_selection):
                clamped_full[0, t, 0, idx] = interpolated[i]
        return clamped_full

    def _run_sim(self, clamped_full, grad_enabled=False):
        """Run DEFT forward simulation."""
        # previous = current (no initial velocity)
        b_current = self.initial_branched.unsqueeze(1)       # (1, 1, n_branch, max_vert, 3)
        b_previous = self.initial_branched.unsqueeze(1)      # same

        if grad_enabled:
            pred, _ = self.deft_sim.iterative_predict(
                time_horizon=SIM_TIME_HORIZON,
                b_DLOs_vertices_traj=b_current,
                previous_b_DLOs_vertices_traj=b_previous,
                clamped_positions=clamped_full,
                dt=0.01,
                parent_theta_clamp=self.parent_theta_clamp,
                child1_theta_clamp=self.child1_theta_clamp,
                child2_theta_clamp=self.child2_theta_clamp,
                inference_1_batch=False
            )
        else:
            with torch.no_grad():
                pred, _ = self.deft_sim.iterative_predict(
                    time_horizon=SIM_TIME_HORIZON,
                    b_DLOs_vertices_traj=b_current,
                    previous_b_DLOs_vertices_traj=b_previous,
                    clamped_positions=clamped_full,
                    dt=0.01,
                    parent_theta_clamp=self.parent_theta_clamp,
                    child1_theta_clamp=self.child1_theta_clamp,
                    child2_theta_clamp=self.child2_theta_clamp,
                    inference_1_batch=True
                )
        return pred

    def objective(self, x):
        """Cost: ||vertex_0_final - target||^2"""
        disp = torch.from_numpy(x).double()
        clamped = self._build_clamped_trajectory(disp)
        pred = self._run_sim(clamped, grad_enabled=False)
        # pred: (1, time, n_branch, max_vert, 3)
        tip_final = pred[0, -1, 0, 0]  # vertex 0 of parent branch at final time
        cost = torch.sum((tip_final - self.target_pos) ** 2).item()

        dist = torch.norm(tip_final - self.target_pos).item()
        print(f"  [Iter {self.iteration}] tip dist to target: {dist:.6f}, cost: {cost:.6f}")

        if dist < 0.01 and not self.converged:
            print(f"  *** Converged! Tip within 0.01 of target ***")
            self.converged = True
            self.best_solution = x.copy()

        self.iteration += 1
        return cost

    def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu,
                     d_norm, regularization_size, alpha_du, alpha_pr, ls_trials):
        """Intermediate callback for early stopping."""
        if self.converged:
            return False  # Stop optimization
        return True  # Continue

    def gradient(self, x):
        """Gradient via autograd."""
        disp = torch.from_numpy(x).double().requires_grad_(True)
        clamped = self._build_clamped_trajectory(disp)
        pred = self._run_sim(clamped, grad_enabled=True)
        tip_final = pred[0, -1, 0, 0]
        cost = torch.sum((tip_final - self.target_pos) ** 2)
        grad = torch.autograd.grad(cost, disp)[0]
        return grad.detach().numpy()



def visualize_initial_state(initial_branched, target, hole_pts, hole_center, hole_normal, config_id, height_id, hole_side):
    """Visualize the initial BDLO configuration with hole and target before optimization."""
    initial = initial_branched[0].numpy()  # (n_branch, max_vert, 3)

    fig = plt.figure(figsize=(14, 6))

    # -- 3D view --
    ax1 = fig.add_subplot(121, projection='3d')
    branch_colors = ['red', 'blue', 'green']
    branch_labels = ['Parent', 'Child1', 'Child2']
    for bi in range(N_BRANCH):
        v = initial[bi]
        mask = np.any(v != 0, axis=-1)
        if mask.any():
            ax1.plot(v[mask, 0], v[mask, 1], v[mask, 2], 'o-',
                     color=branch_colors[bi], linewidth=2, markersize=5, label=branch_labels[bi])
    # Highlight vertex 0 (tip) and vertices 1, 2 (control pair)
    ax1.scatter(*initial[0, 0], color='red', s=120, marker='D', edgecolors='black', zorder=5, label='v0 (tip)')
    ax1.scatter(*initial[0, 1], color='orange', s=120, marker='s', edgecolors='black', zorder=5, label='v1 (control)')
    ax1.scatter(*initial[0, 2], color='orange', s=120, marker='s', edgecolors='black', zorder=5, label='v2 (control)')

    # Hole triangle
    tri = [0, 1, 2, 0]
    ax1.plot(hole_pts[tri, 0], hole_pts[tri, 1], hole_pts[tri, 2],
             'g^-', linewidth=2, markersize=8, label='Hole')
    if len(hole_pts) >= 4:
        ax1.scatter(*hole_pts[3], color='lime', s=60, marker='*', edgecolors='black', zorder=5)

    # Target
    ax1.scatter(*target, color='magenta', s=100, marker='x', linewidths=3, label='Target')
    # Hole normal arrow (shows insertion direction)
    arrow_len = 0.05
    ax1.quiver(hole_center[0], hole_center[1], hole_center[2],
               hole_normal[0]*arrow_len, hole_normal[1]*arrow_len, hole_normal[2]*arrow_len,
               color='purple', arrow_length_ratio=0.3, linewidth=2, label='Normal')

    ax1.set_title(f'Initial State: Config {config_id} + H{height_id}{hole_side}')
    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
    ax1.legend(fontsize=7)
    ax1.view_init(elev=20, azim=-60)

    # -- X-Z view --
    ax2 = fig.add_subplot(122)
    for bi in range(N_BRANCH):
        v = initial[bi]
        mask = np.any(v != 0, axis=-1)
        if mask.any():
            ax2.plot(v[mask, 0], v[mask, 2], 'o-', color=branch_colors[bi], linewidth=2,
                     markersize=5, label=branch_labels[bi])
    ax2.scatter(initial[0, 0, 0], initial[0, 0, 2], color='red', s=120, marker='D',
                edgecolors='black', zorder=5, label='v0 (tip)')
    ax2.scatter(initial[0, 1, 0], initial[0, 1, 2], color='orange', s=120, marker='s',
                edgecolors='black', zorder=5, label='v1 (control)')
    ax2.scatter(initial[0, 2, 0], initial[0, 2, 2], color='orange', s=120, marker='s',
                edgecolors='black', zorder=5, label='v2 (control)')
    ax2.plot(hole_pts[tri, 0], hole_pts[tri, 2], 'g^-', linewidth=2, label='Hole')
    ax2.scatter(target[0], target[2], color='magenta', s=100, marker='x', zorder=5, label='Target')

    ax2.set_title(f'X-Z View: Initial State')
    ax2.set_xlabel('X'); ax2.set_ylabel('Z')
    ax2.set_aspect('equal')
    ax2.legend(fontsize=7)
    ax2.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    out_dir = os.path.join(os.path.dirname(__file__), 'visualization', 'parent_branch_thread_insertion')
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, f'initial_config{config_id}_H{height_id}{hole_side}.png'), dpi=150)
    print("Saved initial state visualization.")
    plt.close()


def finite_difference_gradient(problem_obj, x, eps=1e-6):
    """Compute gradient using finite differences for verification."""
    n = len(x)
    fd_grad = np.zeros(n)
    for i in range(n):
        x_plus = x.copy()
        x_plus[i] += eps
        x_minus = x.copy()
        x_minus[i] -= eps
        # Temporarily suppress printing in objective
        save_iter = problem_obj.iteration
        cost_plus = problem_obj.objective(x_plus)
        cost_minus = problem_obj.objective(x_minus)
        problem_obj.iteration = save_iter
        fd_grad[i] = (cost_plus - cost_minus) / (2 * eps)
    return fd_grad


def _run_warmup(deft_sim, initial_branched, parent_theta_clamp,
                child1_theta_clamp, child2_theta_clamp, parent_clamped_selection):
    """Run a warmup phase with all clamped vertices held static at initial positions.
    Returns the settled final state as (1, n_branch, max_vert, 3).
    """
    print(f"\n--- Warmup: {WARMUP_STEPS} steps (static hold) to damp initial vibrations ---")
    batch = 1
    clamped_full = torch.zeros(batch, WARMUP_STEPS, N_BRANCH, N_PARENT, 3, dtype=torch.float64)
    # Hold all clamped vertices at their initial positions for every timestep
    for t in range(WARMUP_STEPS):
        for idx in parent_clamped_selection:
            clamped_full[0, t, 0, idx] = initial_branched[0, 0, idx]

    b_current = initial_branched.unsqueeze(1)
    b_previous = initial_branched.unsqueeze(1)
    with torch.no_grad():
        pred, _ = deft_sim.iterative_predict(
            time_horizon=WARMUP_STEPS,
            b_DLOs_vertices_traj=b_current,
            previous_b_DLOs_vertices_traj=b_previous,
            clamped_positions=clamped_full,
            dt=0.01,
            parent_theta_clamp=parent_theta_clamp,
            child1_theta_clamp=child1_theta_clamp,
            child2_theta_clamp=child2_theta_clamp,
            inference_1_batch=True
        )
    settled = pred[:, -1].detach().clone()  # (1, n_branch, max_vert, 3)
    tip_before = initial_branched[0, 0, 0].numpy()
    tip_after = settled[0, 0, 0].numpy()
    print(f"  Tip before warmup: {tip_before}")
    print(f"  Tip after warmup:  {tip_after}")
    print(f"  Tip drift: {np.linalg.norm(tip_after - tip_before):.6f}")
    return settled


def _run_single_stage(deft_sim, initial_branched, parent_theta_clamp,
                      child1_theta_clamp, child2_theta_clamp, target,
                      parent_clamped_selection, stage_name, run_grad_check=True):
    """Run a single IPOPT optimization stage.
    Returns (solution, optimized_traj, final_state_branched).
    final_state_branched: (1, n_branch, max_vert, 3) — the last frame, used as init for next stage.
    """
    problem_obj = ThreadInsertionProblem(
        deft_sim, initial_branched, parent_theta_clamp,
        child1_theta_clamp, child2_theta_clamp, target,
        parent_clamped_selection
    )

    # Gradient check
    if run_grad_check:
        print(f"\n--- {stage_name}: Gradient Check ---")
        x_test = np.zeros(problem_obj.n)
        autograd = problem_obj.gradient(x_test)
        fd_grad = finite_difference_gradient(problem_obj, x_test)
        print(f"  Autograd:     {autograd}")
        print(f"  Finite diff:  {fd_grad}")
        print(f"  Abs diff:     {np.abs(autograd - fd_grad)}")
        print(f"  Max abs diff: {np.max(np.abs(autograd - fd_grad)):.8f}")
        problem_obj.iteration = 0

    # IPOPT solve
    n = problem_obj.n
    x0 = np.zeros(n)
    problem = cyipopt.Problem(
        n=n, m=0,
        problem_obj=problem_obj,
        lb=[-1e20] * n, ub=[1e20] * n
    )
    problem.add_option("print_level", 5)
    problem.add_option("tol", 1e-4)
    problem.add_option("max_iter", 100)

    print(f"\n{stage_name}: Starting IPOPT optimization...")
    start_time = time.time()
    solution, info = problem.solve(x0)
    elapsed = time.time() - start_time

    if problem_obj.best_solution is not None:
        print(f"{stage_name}: Using early-stopped solution (tip dist < 0.01)")
        solution = problem_obj.best_solution

    print(f"\n{stage_name} IPOPT Status: {info['status_msg']}")
    print(f"{stage_name} Final objective: {info['obj_val']:.6f}")
    print(f"{stage_name} Optimized displacement: {solution}")
    print(f"{stage_name} Optimization time: {elapsed:.2f}s")

    # Generate trajectory
    disp = torch.from_numpy(solution).double()
    clamped_full = problem_obj._build_clamped_trajectory(disp)
    with torch.no_grad():
        b_current = initial_branched.unsqueeze(1)
        b_previous = initial_branched.unsqueeze(1)
        optimized_traj, _ = deft_sim.iterative_predict(
            time_horizon=SIM_TIME_HORIZON,
            b_DLOs_vertices_traj=b_current,
            previous_b_DLOs_vertices_traj=b_previous,
            clamped_positions=clamped_full,
            dt=0.01,
            parent_theta_clamp=parent_theta_clamp,
            child1_theta_clamp=child1_theta_clamp,
            child2_theta_clamp=child2_theta_clamp,
            inference_1_batch=False
        )

    tip_final = optimized_traj[0, -1, 0, 0].numpy()
    final_dist = np.linalg.norm(tip_final - target)
    print(f"{stage_name} Final tip position: {tip_final}")
    print(f"{stage_name} Final tip-to-target distance: {final_dist:.6f}")

    # Extract final frame as the initial state for the next stage
    final_state_branched = optimized_traj[:, -1].detach().clone()  # (1, n_branch, max_vert, 3)

    return solution, info, optimized_traj, final_state_branched, elapsed


def run_thread_insertion(config_id, height_id, hole_side, checkpoint_path=None,
                         offset_distance=0.05, align_distance=0.05, gradient_check=False):
    """Run 2-stage thread insertion optimization.

    Stage 1 (Align): Move tip to a point on the tip's side of the hole,
                      align_distance away along the normal.
    Stage 2 (Insert): From aligned state, push tip through the hole to
                       offset_distance on the far side.
    """
    torch.set_default_dtype(torch.float64)

    if not IPOPT_AVAILABLE:
        raise ImportError("cyipopt required. Install with: pip install cyipopt")

    # 1. Load data
    print(f"\n{'='*60}")
    print(f"Thread insertion (2-stage): Config {config_id} -> Hole H{height_id}{hole_side}")
    print(f"  align_distance={align_distance}, offset_distance={offset_distance}")
    print(f"{'='*60}")

    flat_pts = load_mocap_config(config_id)          # (20, 3) already preprocessed
    hole_pts = load_hole(height_id, hole_side)       # (4, 3) already preprocessed


    # Compute hole center & normal oriented toward the TIP (vertex 0) (not oriented toward TIP in this version)
    tip_position = flat_pts[0]  # vertex 0
    hole_center, hole_normal = compute_hole_center_and_normal(hole_pts, tip_position)
    print(f"Hole center: {hole_center}")
    print(f"Hole normal: {hole_normal}")
    print(f"Tip (v0): {tip_position}")

    # Stage 1 target: on the tip's side of the hole, align_distance away
    # Normal points toward tip, so center + normal * align_distance = toward tip side
    target_align = hole_center + hole_normal * align_distance
    # Stage 2 target: on the far side of the hole from the tip, with z offset
    target_insert = hole_center - hole_normal * offset_distance
    target_insert[2] -= 0.02  # small downward z offset

    print(f"\nStage 1 target (align):  {target_align}")
    print(f"Stage 2 target (insert): {target_insert}")

    # 2. Build initial state
    initial_branched = mocap_to_branched_tensor(flat_pts)  # (1, n_branch, max_vert, 3)
    tip_initial = initial_branched[0, 0, 0].numpy()
    print(f"Tip (vertex 0) initial: {tip_initial}")
    print(f"Initial tip-to-align distance: {np.linalg.norm(tip_initial - target_align):.4f}")
    print(f"Initial tip-to-insert distance: {np.linalg.norm(tip_initial - target_insert):.4f}")

    # 3. Visualize initial state (show both targets)
    visualize_initial_state(initial_branched, target_insert, hole_pts, hole_center, hole_normal,
                            config_id, height_id, hole_side)

    # 4. Setup DEFT sim
    parent_clamped_selection = torch.tensor((1, 2, -2, -1))
    deft_sim, parent_theta_clamp, child1_theta_clamp, child2_theta_clamp = setup_deft_sim(
        parent_clamped_selection, load_checkpoint=checkpoint_path
    )

    # ===== WARMUP: Hold static to damp initial vibrations =====
    settled_state = _run_warmup(
        deft_sim, initial_branched, parent_theta_clamp,
        child1_theta_clamp, child2_theta_clamp, parent_clamped_selection
    )

    # ===== STAGE 1: Align =====
    print(f"\n{'='*60}")
    print(f"  STAGE 1: ALIGN (tip -> {align_distance}m from hole center)")
    print(f"{'='*60}")
    sol1, info1, traj1, state_after_align, time1 = _run_single_stage(
        deft_sim, settled_state, parent_theta_clamp,
        child1_theta_clamp, child2_theta_clamp, target_align,
        parent_clamped_selection, "Stage1-Align", run_grad_check=gradient_check
    )

    # ===== STAGE 2: Insert =====
    print(f"\n{'='*60}")
    print(f"  STAGE 2: INSERT (tip -> through hole, {offset_distance}m past)")
    print(f"{'='*60}")
    sol2, info2, traj2, state_after_insert, time2 = _run_single_stage(
        deft_sim, state_after_align, parent_theta_clamp,
        child1_theta_clamp, child2_theta_clamp, target_insert,
        parent_clamped_selection, "Stage2-Insert", run_grad_check=gradient_check
    )

    # Concatenate trajectories for visualization
    combined_traj = torch.cat([traj1, traj2], dim=1)  # (1, 2*time, n_branch, max_vert, 3)

    tip_final = combined_traj[0, -1, 0, 0].numpy()
    final_dist = np.linalg.norm(tip_final - target_insert)
    print(f"\n{'='*60}")
    print(f"  OVERALL RESULT")
    print(f"{'='*60}")
    print(f"Final tip position: {tip_final}")
    print(f"Final tip-to-insert-target distance: {final_dist:.6f}")
    print(f"Total optimization time: {time1 + time2:.2f}s")

    return combined_traj, initial_branched, target_insert, hole_pts, flat_pts, sol2, info2


def visualize_result(optimized_traj, initial_branched, target_pt, hole_pts, flat_pts,
                     config_id, height_id, hole_side, save_path=None):
    """Visualize the thread insertion result."""
    # Data is already in the correct coordinate frame — no conversion needed
    traj = optimized_traj[0].numpy()      # (time, n_branch, max_vert, 3)
    initial = initial_branched[0].numpy()  # (n_branch, max_vert, 3)
    target = target_pt

    # Wire connections (0-indexed, parent branch only for clarity)
    parent_conn = list(range(N_PARENT))
    child1_conn = list(range(N_CHILD1))
    child2_conn = list(range(N_CHILD2))

    fig = plt.figure(figsize=(14, 6))

    # -- Left: initial vs final --
    ax1 = fig.add_subplot(121, projection='3d')
    # Initial (faded)
    p_init = initial[0]
    mask_p = np.any(p_init != 0, axis=-1)
    ax1.plot(p_init[mask_p, 0], p_init[mask_p, 1], p_init[mask_p, 2],
             'o-', color='gray', alpha=0.4, label='Initial', linewidth=1.5)
    # Final
    p_final = traj[-1, 0]
    mask_f = np.any(p_final != 0, axis=-1)
    ax1.plot(p_final[mask_f, 0], p_final[mask_f, 1], p_final[mask_f, 2],
             'o-', color='red', linewidth=2, label='Final')
    # Child branches final
    for bi, color in [(1, 'blue'), (2, 'green')]:
        c = traj[-1, bi]
        mask = np.any(c != 0, axis=-1)
        if mask.any():
            ax1.plot(c[mask, 0], c[mask, 1], c[mask, 2], 'o-', color=color, linewidth=1.5)

    # Hole triangle
    tri = [0, 1, 2, 0]
    ax1.plot(hole_pts[tri, 0], hole_pts[tri, 1], hole_pts[tri, 2],
             'g^-', linewidth=2, markersize=8, label='Hole')
    if len(hole_pts) >= 4:
        ax1.scatter(*hole_pts[3], color='lime', s=60, marker='*', edgecolors='black', zorder=5)

    # Target point
    ax1.scatter(*target, color='magenta', s=100, marker='x', linewidths=3, label='Target')
    # Tip trajectory
    tips = traj[:, 0, 0]  # (time, 3) - vertex 0 over time
    ax1.plot(tips[:, 0], tips[:, 1], tips[:, 2], '--', color='orange', alpha=0.6, label='Tip path')

    ax1.set_title(f'Config {config_id} -> H{height_id}{hole_side}')
    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
    ax1.legend(fontsize=7)
    ax1.view_init(elev=20, azim=-60)

    # -- Right: X-Z view --
    ax2 = fig.add_subplot(122)
    ax2.plot(p_init[mask_p, 0], p_init[mask_p, 2], 'o-', color='gray', alpha=0.4, label='Initial')
    ax2.plot(p_final[mask_f, 0], p_final[mask_f, 2], 'o-', color='red', linewidth=2, label='Final')
    ax2.plot(hole_pts[tri, 0], hole_pts[tri, 2], 'g^-', linewidth=2, label='Hole')
    ax2.scatter(target[0], target[2], color='magenta', s=100, marker='x', zorder=5, label='Target')
    ax2.plot(tips[:, 0], tips[:, 2], '--', color='orange', alpha=0.6, label='Tip path')
    ax2.set_title(f'X-Z View: Config {config_id} -> H{height_id}{hole_side}')
    ax2.set_xlabel('X'); ax2.set_ylabel('Z')
    ax2.set_aspect('equal')
    ax2.legend(fontsize=7)
    ax2.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved figure to {save_path}")
    else:
        plt.show()


def animate_result(optimized_traj, initial_branched, target_pt, hole_pts,
                   config_id, height_id, hole_side, save_path=None, skip=2):
    """Animate the thread insertion."""
    traj = optimized_traj[0].numpy()
    target = target_pt

    # Precompute fixed axis limits from full trajectory + hole + target
    all_points = []
    for t in range(traj.shape[0]):
        for bi in range(traj.shape[1]):
            v = traj[t, bi]
            mask = np.any(v != 0, axis=-1)
            if mask.any():
                all_points.append(v[mask])
    all_points.append(hole_pts)
    all_points.append(np.array(target).reshape(1, 3))
    all_points = np.concatenate(all_points, axis=0)
    margin = 0.02
    x_min, x_max = all_points[:, 0].min() - margin, all_points[:, 0].max() + margin
    y_min, y_max = all_points[:, 1].min() - margin, all_points[:, 1].max() + margin
    z_min, z_max = all_points[:, 2].min() - margin, all_points[:, 2].max() + margin

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    colors = ['red', 'blue', 'green']

    def update(frame):
        ax.clear()
        # Current BDLO
        for bi, c in enumerate(colors):
            v = traj[frame, bi]
            mask = np.any(v != 0, axis=-1)
            if mask.any():
                ax.plot(v[mask, 0], v[mask, 1], v[mask, 2], 'o-', color=c, linewidth=2, markersize=4)
        # Hole
        tri = [0, 1, 2, 0]
        ax.plot(hole_pts[tri, 0], hole_pts[tri, 1], hole_pts[tri, 2],
                'g^-', linewidth=2, markersize=8)
        # Target
        ax.scatter(*target, color='magenta', s=100, marker='x', linewidths=3)
        # Tip trace up to current frame
        tips = traj[:frame+1, 0, 0]
        ax.plot(tips[:, 0], tips[:, 1], tips[:, 2], '--', color='orange', alpha=0.5)

        # Fixed axis limits
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)

        ax.set_title(f'Config {config_id}->H{height_id}{hole_side} Frame {frame}/{traj.shape[0]}')
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.view_init(elev=20, azim=-60)

    frames = range(0, traj.shape[0], skip)
    anim = FuncAnimation(fig, update, frames=frames, interval=50)

    if save_path:
        anim.save(save_path, writer='ffmpeg', fps=20)
        print(f"Saved animation to {save_path}")
    else:
        plt.show()
    return anim


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Thread insertion optimization')
    parser.add_argument('--config', type=int, default=1, help='Mocap config ID (1-5)')
    parser.add_argument('--height', type=int, default=1, help='Height file ID (1-4)')
    parser.add_argument('--hole', type=str, default='A', choices=['A', 'B'], help='Hole side')
    parser.add_argument('--checkpoint', type=str, default=None, help='Model checkpoint path')
    parser.add_argument('--offset', type=float, default=0.05, help='Target offset behind hole (stage 2)')
    parser.add_argument('--align', type=float, default=0.05, help='Alignment distance in front of hole (stage 1)')
    parser.add_argument('--gradient-check', action='store_true', default=False, help='Run gradient check before each IPOPT stage')
    args = parser.parse_args()

    result = run_thread_insertion(
        config_id=args.config,
        height_id=args.height,
        hole_side=args.hole,
        checkpoint_path=args.checkpoint,
        offset_distance=args.offset,
        align_distance=args.align,
        gradient_check=args.gradient_check,
    )
    optimized_traj, initial_branched, target_pt, hole_pts, flat_pts, solution, info = result

    vis_dir = os.path.join(os.path.dirname(__file__), 'visualization', 'parent_branch_thread_insertion')
    os.makedirs(vis_dir, exist_ok=True)
    tag = f"config{args.config}_H{args.height}{args.hole}"

    # Save visualization figures
    fig_path = os.path.join(vis_dir, f'{tag}.png')
    visualize_result(optimized_traj, initial_branched, target_pt, hole_pts, flat_pts,
                     args.config, args.height, args.hole, save_path=fig_path)
    
    # Save animation
    anim_path = os.path.join(vis_dir, f'{tag}.mp4')
    animate_result(optimized_traj, initial_branched, target_pt, hole_pts,
                   args.config, args.height, args.hole, save_path=anim_path)

    # Save controlled vertex's trajectory as pkl file
    out_dir = os.path.join(os.path.dirname(__file__), 'trajectories', 'parent_branch_thread_insertion')
    os.makedirs(out_dir, exist_ok=True)
    with open(f'{out_dir}/{tag}.pkl', 'wb') as f:
        pickle.dump(optimized_traj[0, :, 0, 2, :], f)
