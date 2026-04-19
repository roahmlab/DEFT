#!/usr/bin/env python3
"""
Child-branch thread insertion optimization: control parent vertices 0,1 to guide
the tip of child branch 2 (flat vertex 17, branched: branch 2, vertex 3) through a target hole.
Uses IPOPT with DEFT simulation as the forward model.

ABLATION: deft ablation
- attachment constraints:       ON  (kept)
- orientation constraints:      OFF (implies coplanar projection OFF — the c1
                                coplanar logic lives inside the orientation
                                block)
- extreme parent mass/MOI:      OFF (uses normal scales: 1., 10.)
- c2→c1 coupling/momentum:      default (no extreme overrides)

BDLO5 topology:
  - Parent: 12 vertices (indices 0-11)
  - Child1: 4 vertices, attaches at parent vertex 5
  - Child2: 4 vertices, attaches at child1 local vertex 1
  - Coupling index: [5, 1]
  - Flat layout: parent(12) + child1_excl_coupling(3) + child2_excl_coupling(3) = 18
  - Tip to insert: flat vertex 17 = branch 2, vertex 3 (last vertex of child2)

Clamped vertices on parent: 0, 1 (control pair, shared displacement), -2, -1 (hard-clamped fixed).
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
WARMUP_STEPS = 100  # 1 second at dt=0.01
DATASET_DIR = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'BDLO_child_branch_thread_insertion')

# BDLO5 parameters
N_PARENT = 12
N_CHILD1 = 4
N_CHILD2 = 4
N_BRANCH = 3
COUPLING_INDEX = [5, 1]
N_FLAT = N_PARENT + (N_CHILD1 - 1) + (N_CHILD2 - 1)  # 18

# Tip to insert: child2 tip = branch 2, vertex 3 in branched format
TIP_BRANCH = 2
TIP_VERTEX = N_CHILD2 - 1  # 3


def load_mocap_config(kinova_id, franka_id):
    """Load a mocap initial configuration (18 flat vertices in raw coords)."""
    fpath = os.path.join(DATASET_DIR,
                         f'child_insertion_kinova_{kinova_id}_franka_{franka_id}_transformed.pkl')
    with open(fpath, 'rb') as f:
        data = pickle.load(f)
    # data shape: (1, 3, 18)
    pts = np.array(data).squeeze().T  # (18, 3)
    pts[:, 0] = -pts[:, 0]   # flip x to match the DEFT BDLO5 frame
    return pts


def load_hole(target_id):
    """Load hole points (4 vertices) from target file."""
    fpath = os.path.join(DATASET_DIR, f'child_insertion_target_{target_id}_transformed.pkl')
    with open(fpath, 'rb') as f:
        data = pickle.load(f)
    # data shape: (1, 3, 4)
    pts = np.array(data).squeeze().T  # (4, 3)
    return pts


def compute_hole_center_and_normal(hole_pts, tip_position):
    """Compute hole center and normal (facing TOWARD the tip).
    hole_pts: (4, 3) raw coordinates
    tip_position: (3,) position of the tip vertex for orienting normal
    """
    center = hole_pts.mean(axis=0)
    # Normal from cross product of two edges (first 3 points)
    v1 = hole_pts[1] - hole_pts[0]
    v2 = hole_pts[2] - hole_pts[0]
    normal = np.cross(v1, v2)
    normal = normal / np.linalg.norm(normal)
    # Orient normal TOWARD the tip
    to_tip = tip_position - center
    if np.dot(normal, to_tip) < 0:
        normal = -normal
    return center, normal


def mocap_to_branched_tensor(flat_vertices_raw):
    """Convert 18 flat mocap vertices to DEFT branched format.

    Input: (18, 3) already in DEFT-compatible coords
    Output: (1, n_branch, max_vert, 3)
    """
    flat = torch.tensor(flat_vertices_raw, dtype=torch.float64)
    # Split into parent / child1 / child2
    parent = flat[:N_PARENT]                              # (12, 3)
    child1 = flat[N_PARENT:N_PARENT + N_CHILD1 - 1]      # (3, 3) - exclude coupling point
    child2 = flat[N_PARENT + N_CHILD1 - 1:]               # (3, 3) - exclude coupling point

    # Add batch dimension: (1, n_vert, 3)
    parent_t = parent.unsqueeze(0)
    child1_t = child1.unsqueeze(0)
    child2_t = child2.unsqueeze(0)

    # Build branched format: (1, n_branch, max_vert, 3)
    # bdlo5=True: child2 coupling point comes from child1, not parent
    b_verts = construct_BDLOs_data(
        1, COUPLING_INDEX, N_PARENT, (N_CHILD1, N_CHILD2), N_BRANCH,
        parent_t, child1_t, child2_t, bdlo5=True
    )

    return b_verts  # (1, n_branch, max_vert, 3)


def setup_deft_sim(parent_clamped_selection, load_checkpoint=None):
    """Setup DEFT simulation for child-branch thread insertion."""
    torch.set_default_dtype(torch.float64)
    device = "cpu"

    # BDLO5 undeformed shape
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

    batch = 1
    child1_clamped_selection = torch.tensor((2))
    child2_clamped_selection = torch.tensor((2))

    parent_vertices = undeformed_BDLO[:, :N_PARENT]
    child1_vertices = undeformed_BDLO[:, N_PARENT:N_PARENT + N_CHILD1 - 1]
    child2_vertices = undeformed_BDLO[:, N_PARENT + N_CHILD1 - 1:]

    # ABLATION (deft ablation): use normal mass/MOI scales (parent_mass_scale=1,
    # parent_moment_scale=10) — no "infinite parent" trick.
    b_DLO_mass, parent_MOI, children_MOI, _, _, _ = DEFT_initialization(
        parent_vertices, child1_vertices, child2_vertices, N_BRANCH, N_PARENT,
        (N_CHILD1, N_CHILD2), COUPLING_INDEX, 1., 10., (0.5, 0.5), (1, 1), 0.1,
        bdlo5=True
    )

    b_DLOs_vertices, _ = construct_b_DLOs(
        batch, COUPLING_INDEX, N_PARENT, (N_CHILD1, N_CHILD2), N_BRANCH,
        parent_vertices, parent_vertices, child1_vertices, child1_vertices,
        child2_vertices, child2_vertices, bdlo5=True
    )

    # BDLO5 coordinate transform: (-z,-x,y) then rotate -X 90° then negate x
    step1 = torch.zeros_like(b_DLOs_vertices)
    step1[:, :, :, 0] = -b_DLOs_vertices[:, :, :, 2]
    step1[:, :, :, 1] = -b_DLOs_vertices[:, :, :, 0]
    step1[:, :, :, 2] = b_DLOs_vertices[:, :, :, 1]
    b_DLOs_transform = torch.zeros_like(step1)
    b_DLOs_transform[:, :, :, 0] = -step1[:, :, :, 0]
    b_DLOs_transform[:, :, :, 1] = step1[:, :, :, 2]
    b_DLOs_transform[:, :, :, 2] = -step1[:, :, :, 1]
    b_undeformed_vert = b_DLOs_transform[0].view(N_BRANCH, -1, 3)

    index_selection1, index_selection2, parent_MOI_index1, parent_MOI_index2 = index_init(COUPLING_INDEX, N_BRANCH)
    clamped_index, parent_theta_clamp, child1_theta_clamp, child2_theta_clamp = clamp_index(
        batch, parent_clamped_selection, child1_clamped_selection, child2_clamped_selection,
        N_BRANCH, N_PARENT, True, False, False
    )

    bend_stiffness_parent = nn.Parameter(2e-3 * torch.ones((1, 1, N_PARENT - 1), device=device))
    bend_stiffness_child1 = nn.Parameter(2e-3 * torch.ones((1, 1, N_PARENT - 1), device=device))
    bend_stiffness_child2 = nn.Parameter(2e-3 * torch.ones((1, 1, N_PARENT - 1), device=device))
    twist_stiffness = nn.Parameter(1e-4 * torch.ones((1, N_BRANCH, N_PARENT - 1), device=device))
    damping = nn.Parameter(torch.tensor((3., 3., 3.), device=device))
    learning_weight = nn.Parameter(torch.tensor(0.0, device=device))

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
        use_orientation_constraints=False, use_attachment_constraints=True,
        bdlo5=True
    )

    # ABLATION (deft ablation): no c2↔c1 extreme coupling/momentum overrides;
    # the default initializer values are used. Attachment constraints are still on.

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


class ChildBranchInsertionProblem:
    """IPOPT problem: optimize shared v0/v1 displacement (3 DOF) to guide child2 tip through a hole.

    Clamped vertices: 0, 1 (control pair, shared displacement), -2, -1 (hard-clamped fixed).
    Tip to track: branch 2, vertex 3 (child2 tip, flat vertex 17).
    """

    def __init__(self, deft_sim, initial_branched, parent_theta_clamp,
                 child1_theta_clamp, child2_theta_clamp, target_pos,
                 parent_clamped_selection, hole_pts=None, out_dir=None):
        self.deft_sim = deft_sim
        self.deft_sim.eval()
        self.hole_pts = hole_pts if hole_pts is not None else np.zeros((4, 3))
        self.out_dir = out_dir
        self.last_x = None  # track current solution for animation
        for param in self.deft_sim.parameters():
            param.requires_grad = False

        self.initial_branched = initial_branched.detach().clone()
        self.parent_theta_clamp = parent_theta_clamp
        self.child1_theta_clamp = child1_theta_clamp
        self.child2_theta_clamp = child2_theta_clamp
        self.target_pos = torch.tensor(target_pos, dtype=torch.float64)  # (3,)
        self.parent_clamped_selection = parent_clamped_selection

        # Initial positions of all 4 clamped vertices: v0, v1, v-2, v-1
        self.initial_clamped = initial_branched[0, 0, parent_clamped_selection].detach().clone()  # (4, 3)

        self.n = 3  # 3 DOF: shared displacement for v0 & v1
        self.iteration = 0
        self.converged = False
        self.best_solution = None

    def _expand_displacements(self, displacement):
        """Expand 3 DOF to per-clamped-vertex displacements (4, 3).
        displacement: (3,) — shared displacement for v0 & v1.
        v-2 & v-1 get zero displacement (hard-clamped).
        """
        zero = torch.zeros(3, dtype=torch.float64)
        return torch.stack([displacement, displacement, zero, zero])  # v0, v1, v-2, v-1

    def _build_clamped_trajectory(self, displacement):
        """Build clamped_positions tensor.
        displacement: (3,) torch tensor — shared displacement for v0 & v1.
        Vertices -2, -1 stay fixed at their initial positions.
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
        b_current = self.initial_branched.unsqueeze(1)
        b_previous = self.initial_branched.unsqueeze(1)

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
        """Cost: ||child2_tip_final - target||^2"""
        self.last_x = x.copy()
        disp = torch.from_numpy(x).double()
        clamped = self._build_clamped_trajectory(disp)
        pred = self._run_sim(clamped, grad_enabled=False)
        # pred: (1, time, n_branch, max_vert, 3)
        tip_final = pred[0, -1, TIP_BRANCH, TIP_VERTEX]  # child2 tip at final time
        cost = torch.sum((tip_final - self.target_pos) ** 2).item()

        dist = torch.norm(tip_final - self.target_pos).item()
        print(f"  [Iter {self.iteration}] child2 tip dist to target: {dist:.6f}, cost: {cost:.6f}")

        if dist < 0.01 and not self.converged:
            print(f"  *** Converged! Tip within 0.01 of target ***")
            self.converged = True
            self.best_solution = x.copy()

        self.iteration += 1
        return cost

    def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu,
                     d_norm, regularization_size, alpha_du, alpha_pr, ls_trials):
        """Intermediate callback: save animation after each IPOPT iteration."""
        if self.last_x is not None:
            abs_pos = self.initial_clamped[0].numpy() + self.last_x
            print(f"  [IPOPT iter {iter_count}] end effector: {abs_pos}, obj: {obj_value:.6f}")
        if self.out_dir is not None and self.last_x is not None:
            disp = torch.from_numpy(self.last_x).double()
            clamped = self._build_clamped_trajectory(disp)
            with torch.no_grad():
                b_current = self.initial_branched.unsqueeze(1)
                b_previous = self.initial_branched.unsqueeze(1)
                pred, _ = self.deft_sim.iterative_predict(
                    time_horizon=SIM_TIME_HORIZON,
                    b_DLOs_vertices_traj=b_current,
                    previous_b_DLOs_vertices_traj=b_previous,
                    clamped_positions=clamped,
                    dt=0.01,
                    parent_theta_clamp=self.parent_theta_clamp,
                    child1_theta_clamp=self.child1_theta_clamp,
                    child2_theta_clamp=self.child2_theta_clamp,
                    inference_1_batch=False
                )
            anim_path = os.path.join(self.out_dir, f'ipopt_iter_{iter_count}.mp4')
            target_np = self.target_pos.numpy()
            animate_result(pred, self.initial_branched, target_np, self.hole_pts,
                           0, 0, 0, save_path=anim_path)
            print(f"  Saved animation: {anim_path}")

        if self.converged:
            return False
        return True

    def gradient(self, x):
        """Gradient via autograd."""
        disp = torch.from_numpy(x).double().requires_grad_(True)
        clamped = self._build_clamped_trajectory(disp)
        pred = self._run_sim(clamped, grad_enabled=True)
        tip_final = pred[0, -1, TIP_BRANCH, TIP_VERTEX]
        cost = torch.sum((tip_final - self.target_pos) ** 2)
        grad = torch.autograd.grad(cost, disp)[0]
        return grad.detach().numpy()


def visualize_initial_state(initial_branched, target, hole_pts, hole_center, hole_normal,
                            kinova_id, franka_id, target_id, target_align=None,
                            interactive=False):
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
    # Highlight child2 tip (the vertex we're inserting)
    ax1.scatter(*initial[TIP_BRANCH, TIP_VERTEX], color='magenta', s=120, marker='D',
                edgecolors='black', zorder=5, label='child2 tip (insert)')
    # Highlight control vertices v0, v1
    ax1.scatter(*initial[0, 0], color='orange', s=120, marker='s',
                edgecolors='black', zorder=5, label='v0 (control)')
    ax1.scatter(*initial[0, 1], color='orange', s=120, marker='s',
                edgecolors='black', zorder=5, label='v1 (control)')

    # Hole triangle
    tri = [0, 1, 2, 0]
    ax1.plot(hole_pts[tri, 0], hole_pts[tri, 1], hole_pts[tri, 2],
             'g^-', color='green', linewidth=2, markersize=8, label='Hole')
    if len(hole_pts) >= 4:
        ax1.scatter(*hole_pts[3], color='lime', s=60, marker='*', edgecolors='black', zorder=5)

    # Targets
    ax1.scatter(*target, color='cyan', s=100, marker='x', linewidths=3, label='Target (insert)')
    if target_align is not None:
        ax1.scatter(*target_align, color='yellow', s=100, marker='+', linewidths=3, label='Target (align)')
    # Hole normal arrow
    arrow_len = 0.05
    ax1.quiver(hole_center[0], hole_center[1], hole_center[2],
               hole_normal[0]*arrow_len, hole_normal[1]*arrow_len, hole_normal[2]*arrow_len,
               color='purple', arrow_length_ratio=0.3, linewidth=2, label='Normal')

    ax1.set_title(f'Initial: kinova{kinova_id}_franka{franka_id} -> target{target_id}')
    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
    ax1.legend(fontsize=7)
    ax1.set_box_aspect([1, 1, 1])
    # Equal axis scaling
    all_pts = []
    for bi in range(N_BRANCH):
        v = initial[bi]
        mask = np.any(v != 0, axis=-1)
        if mask.any():
            all_pts.append(v[mask])
    all_pts.append(hole_pts)
    all_pts.append(np.array(target).reshape(1, 3))
    all_pts = np.concatenate(all_pts, axis=0)
    mid = (all_pts.max(axis=0) + all_pts.min(axis=0)) / 2
    half_range = (all_pts.max(axis=0) - all_pts.min(axis=0)).max() / 2 * 1.1
    ax1.set_xlim(mid[0] - half_range, mid[0] + half_range)
    ax1.set_ylim(mid[1] - half_range, mid[1] + half_range)
    ax1.set_zlim(mid[2] - half_range, mid[2] + half_range)
    ax1.view_init(elev=20, azim=-60)

    # -- X-Z view --
    ax2 = fig.add_subplot(122)
    for bi in range(N_BRANCH):
        v = initial[bi]
        mask = np.any(v != 0, axis=-1)
        if mask.any():
            ax2.plot(v[mask, 0], v[mask, 2], 'o-', color=branch_colors[bi], linewidth=2,
                     markersize=5, label=branch_labels[bi])
    ax2.scatter(initial[TIP_BRANCH, TIP_VERTEX, 0], initial[TIP_BRANCH, TIP_VERTEX, 2],
                color='magenta', s=120, marker='D', edgecolors='black', zorder=5, label='child2 tip')
    ax2.scatter(initial[0, 0, 0], initial[0, 0, 2], color='orange', s=120, marker='s',
                edgecolors='black', zorder=5, label='v0 (control)')
    ax2.scatter(initial[0, 1, 0], initial[0, 1, 2], color='orange', s=120, marker='s',
                edgecolors='black', zorder=5, label='v1 (control)')
    ax2.plot(hole_pts[tri, 0], hole_pts[tri, 2], 'g^-', linewidth=2, label='Hole')
    ax2.scatter(target[0], target[2], color='cyan', s=100, marker='x', zorder=5, label='Target (insert)')
    if target_align is not None:
        ax2.scatter(target_align[0], target_align[2], color='yellow', s=100, marker='+', zorder=5, label='Target (align)')

    ax2.set_title(f'X-Z View: Initial State')
    ax2.set_xlabel('X'); ax2.set_ylabel('Z')
    ax2.set_aspect('equal')
    ax2.legend(fontsize=7)
    ax2.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    if interactive:
        print("Showing interactive plot — close the window to continue.")
        plt.show()
    else:
        out_dir = os.path.join(os.path.dirname(__file__), 'child_branch_insertion_output')
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(os.path.join(out_dir,
                    f'initial_kinova{kinova_id}_franka{franka_id}_target{target_id}.png'), dpi=150)
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
        save_iter = problem_obj.iteration
        cost_plus = problem_obj.objective(x_plus)
        cost_minus = problem_obj.objective(x_minus)
        problem_obj.iteration = save_iter
        fd_grad[i] = (cost_plus - cost_minus) / (2 * eps)
    return fd_grad


def _run_warmup(deft_sim, initial_branched, parent_theta_clamp,
                child1_theta_clamp, child2_theta_clamp, parent_clamped_selection):
    """Run a warmup phase with all clamped vertices held static at initial positions."""
    print(f"\n--- Warmup: {WARMUP_STEPS} steps (static hold) to damp initial vibrations ---")
    batch = 1
    clamped_full = torch.zeros(batch, WARMUP_STEPS, N_BRANCH, N_PARENT, 3, dtype=torch.float64)
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
    tip_before = initial_branched[0, TIP_BRANCH, TIP_VERTEX].numpy()
    tip_after = settled[0, TIP_BRANCH, TIP_VERTEX].numpy()
    print(f"  Child2 tip before warmup: {tip_before}")
    print(f"  Child2 tip after warmup:  {tip_after}")
    print(f"  Tip drift: {np.linalg.norm(tip_after - tip_before):.6f}")

    return settled


def _run_single_stage(deft_sim, initial_branched, parent_theta_clamp,
                      child1_theta_clamp, child2_theta_clamp, target,
                      parent_clamped_selection, stage_name, run_grad_check=True,
                      hole_pts=None, out_dir=None):
    """Run a single IPOPT optimization stage."""
    problem_obj = ChildBranchInsertionProblem(
        deft_sim, initial_branched, parent_theta_clamp,
        child1_theta_clamp, child2_theta_clamp, target,
        parent_clamped_selection, hole_pts=hole_pts, out_dir=out_dir
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
    problem.add_option("max_iter", 10)

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

    tip_final = optimized_traj[0, -1, TIP_BRANCH, TIP_VERTEX].numpy()
    final_dist = np.linalg.norm(tip_final - target)
    print(f"{stage_name} Final child2 tip position: {tip_final}")
    print(f"{stage_name} Final tip-to-target distance: {final_dist:.6f}")

    final_state_branched = optimized_traj[:, -1].detach().clone()

    return solution, info, optimized_traj, final_state_branched, elapsed


def run_child_branch_insertion(kinova_id, franka_id, target_id, checkpoint_path=None,
                               offset_distance=0.025, align_distance=0.025):
    """Run 2-stage child-branch thread insertion optimization.

    Stage 1 (Align): Move child2 tip to a point on the tip's side of the hole.
    Stage 2 (Insert): Push child2 tip through the hole to the far side.
    """
    torch.set_default_dtype(torch.float64)
    out_dir = os.path.join(os.path.dirname(__file__), 'child_branch_insertion_output')
    os.makedirs(out_dir, exist_ok=True)

    if not IPOPT_AVAILABLE:
        raise ImportError("cyipopt required. Install with: pip install cyipopt")

    # 1. Load data
    print(f"\n{'='*60}")
    print(f"Child-branch insertion (2-stage): kinova{kinova_id}_franka{franka_id} -> target{target_id}")
    print(f"  align_distance={align_distance}, offset_distance={offset_distance}")
    print(f"  Tip: child2 branch vertex {TIP_VERTEX} (flat vertex 17)")
    print(f"{'='*60}")

    flat_pts = load_mocap_config(kinova_id, franka_id)  # (18, 3)
    hole_pts = load_hole(target_id)                     # (4, 3)
    hole_pts[:, 0] += 0.09  # shift hole 9cm toward x positive
    hole_pts[:, 0] = -hole_pts[:, 0]  # flip x AFTER the shift to match the DEFT BDLO5 frame

    # Compute hole center & normal oriented toward the child2 tip
    # In flat format, tip is vertex 17
    tip_position = flat_pts[17]
    hole_center, hole_normal = compute_hole_center_and_normal(hole_pts, tip_position)
    print(f"Hole center: {hole_center}")
    print(f"Hole normal: {hole_normal} (points toward child2 tip)")
    print(f"Child2 tip (flat v17): {tip_position}")

    # Stage 1 target: on the far side of the hole (away from tip)
    target_align = hole_center - hole_normal * align_distance
    # Stage 2 target: on the opposite side of the hole (toward tip)
    target_insert = hole_center + hole_normal * offset_distance

    print(f"Stage 1 target (align): {target_align}")
    print(f"Stage 2 target (insert): {target_insert}")

    # 2. Build initial state
    initial_branched = mocap_to_branched_tensor(flat_pts)  # (1, n_branch, max_vert, 3)
    tip_initial = initial_branched[0, TIP_BRANCH, TIP_VERTEX].numpy()
    print(f"Child2 tip initial (branched): {tip_initial}")
    print(f"Initial tip-to-align distance: {np.linalg.norm(tip_initial - target_align):.4f}")
    print(f"Initial tip-to-insert distance: {np.linalg.norm(tip_initial - target_insert):.4f}")

    # 3. Visualize initial state
    visualize_initial_state(initial_branched, target_insert, hole_pts, hole_center, hole_normal,
                            kinova_id, franka_id, target_id)

    # 4. Setup DEFT sim
    parent_clamped_selection = torch.tensor((0, 1, -2, -1))
    deft_sim, parent_theta_clamp, child1_theta_clamp, child2_theta_clamp = setup_deft_sim(
        parent_clamped_selection, load_checkpoint=checkpoint_path
    )

    # Stage 1 starts directly from the raw mocap state (no warmup).
    settled_state = initial_branched

    # ===== STAGE 1: Align =====
    print(f"\n{'='*60}")
    print(f"  STAGE 1: ALIGN (child2 tip -> {align_distance}m in front of hole)")
    print(f"{'='*60}")
    sol1, info1, traj1, state_after_align, time1 = _run_single_stage(
        deft_sim, settled_state, parent_theta_clamp,
        child1_theta_clamp, child2_theta_clamp, target_align,
        parent_clamped_selection, "Stage1-Align", run_grad_check=False,
        hole_pts=hole_pts, out_dir=out_dir
    )

    tip_after_align = traj1[0, -1, TIP_BRANCH, TIP_VERTEX].numpy()
    align_dist = np.linalg.norm(tip_after_align - target_align)
    print(f"\n{'='*60}")
    print(f"  STAGE 1 RESULT (align)")
    print(f"{'='*60}")
    print(f"Final child2 tip position: {tip_after_align}")
    print(f"Final tip-to-align-target distance: {align_dist:.6f}")
    print(f"Optimization time: {time1:.2f}s")

    initial_for_insert = state_after_align  # (1, n_branch, max_vert, 3)

    # ===== STAGE 2: Insert =====
    print(f"\n{'='*60}")
    print(f"  STAGE 2: INSERT (child2 tip -> {offset_distance}m behind hole)")
    print(f"{'='*60}")
    sol2, info2, traj2, state_after_insert, time2 = _run_single_stage(
        deft_sim, initial_for_insert, parent_theta_clamp,
        child1_theta_clamp, child2_theta_clamp, target_insert,
        parent_clamped_selection, "Stage2-Insert", run_grad_check=False,
        hole_pts=hole_pts, out_dir=out_dir
    )

    tip_after_insert = traj2[0, -1, TIP_BRANCH, TIP_VERTEX].numpy()
    insert_dist = np.linalg.norm(tip_after_insert - target_insert)
    print(f"\n{'='*60}")
    print(f"  STAGE 2 RESULT (insert)")
    print(f"{'='*60}")
    print(f"Final child2 tip position: {tip_after_insert}")
    print(f"Final tip-to-insert-target distance: {insert_dist:.6f}")
    print(f"Optimization time: {time2:.2f}s")

    return traj2, initial_branched, target_insert, hole_pts, flat_pts, sol2, info2


def visualize_result(optimized_traj, initial_branched, target_pt, hole_pts, flat_pts,
                     kinova_id, franka_id, target_id, save_path=None):
    """Visualize the child-branch thread insertion result."""
    traj = optimized_traj[0].numpy()
    initial = initial_branched[0].numpy()
    target = target_pt

    fig = plt.figure(figsize=(14, 6))

    # -- Left: initial vs final (3D) --
    ax1 = fig.add_subplot(121, projection='3d')
    # Initial (faded)
    for bi, color in enumerate(['gray', 'gray', 'gray']):
        v = initial[bi]
        mask = np.any(v != 0, axis=-1)
        if mask.any():
            ax1.plot(v[mask, 0], v[mask, 1], v[mask, 2],
                     'o-', color=color, alpha=0.4, linewidth=1.5,
                     label='Initial' if bi == 0 else None)
    # Final
    branch_colors = ['red', 'blue', 'green']
    branch_labels = ['Parent (final)', 'Child1 (final)', 'Child2 (final)']
    for bi, (color, label) in enumerate(zip(branch_colors, branch_labels)):
        v = traj[-1, bi]
        mask = np.any(v != 0, axis=-1)
        if mask.any():
            ax1.plot(v[mask, 0], v[mask, 1], v[mask, 2],
                     'o-', color=color, linewidth=2, label=label)

    # Hole triangle
    tri = [0, 1, 2, 0]
    ax1.plot(hole_pts[tri, 0], hole_pts[tri, 1], hole_pts[tri, 2],
             'g^-', color='green', linewidth=2, markersize=8, label='Hole')
    if len(hole_pts) >= 4:
        ax1.scatter(*hole_pts[3], color='lime', s=60, marker='*', edgecolors='black', zorder=5)

    # Target point
    ax1.scatter(*target, color='cyan', s=100, marker='x', linewidths=3, label='Target')
    # Child2 tip trajectory
    tips = traj[:, TIP_BRANCH, TIP_VERTEX]  # (time, 3)
    ax1.plot(tips[:, 0], tips[:, 1], tips[:, 2], '--', color='orange', alpha=0.6, label='Child2 tip path')

    ax1.set_title(f'kinova{kinova_id}_franka{franka_id} -> target{target_id}')
    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
    ax1.legend(fontsize=7)
    ax1.view_init(elev=20, azim=-60)

    # -- Right: X-Z view --
    ax2 = fig.add_subplot(122)
    # Initial
    for bi in range(N_BRANCH):
        v = initial[bi]
        mask = np.any(v != 0, axis=-1)
        if mask.any():
            ax2.plot(v[mask, 0], v[mask, 2], 'o-', color='gray', alpha=0.4,
                     label='Initial' if bi == 0 else None)
    # Final
    for bi, (color, label) in enumerate(zip(branch_colors, branch_labels)):
        v = traj[-1, bi]
        mask = np.any(v != 0, axis=-1)
        if mask.any():
            ax2.plot(v[mask, 0], v[mask, 2], 'o-', color=color, linewidth=2, label=label)
    ax2.plot(hole_pts[tri, 0], hole_pts[tri, 2], 'g^-', linewidth=2, label='Hole')
    ax2.scatter(target[0], target[2], color='cyan', s=100, marker='x', zorder=5, label='Target')
    ax2.plot(tips[:, 0], tips[:, 2], '--', color='orange', alpha=0.6, label='Child2 tip path')

    ax2.set_title(f'X-Z View: kinova{kinova_id}_franka{franka_id} -> target{target_id}')
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
                   kinova_id, franka_id, target_id, save_path=None, skip=2, fps=None):
    """Animate the child-branch thread insertion."""
    traj = optimized_traj[0].numpy()
    target = target_pt

    # Precompute fixed axis limits
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
    # Equal axis scaling
    mid = (all_points.max(axis=0) + all_points.min(axis=0)) / 2
    half_range = (all_points.max(axis=0) - all_points.min(axis=0)).max() / 2 * 1.1

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    colors = ['red', 'blue', 'green']

    def update(frame):
        ax.clear()
        for bi, c in enumerate(colors):
            v = traj[frame, bi]
            mask = np.any(v != 0, axis=-1)
            if mask.any():
                ax.plot(v[mask, 0], v[mask, 1], v[mask, 2], 'o-', color=c, linewidth=2, markersize=4)
        # Hole
        tri = [0, 1, 2, 0]
        ax.plot(hole_pts[tri, 0], hole_pts[tri, 1], hole_pts[tri, 2],
                'g^-', color='green', linewidth=2, markersize=8)
        # Target
        ax.scatter(*target, color='cyan', s=100, marker='x', linewidths=3)
        # Child2 tip trace
        tips = traj[:frame+1, TIP_BRANCH, TIP_VERTEX]
        ax.plot(tips[:, 0], tips[:, 1], tips[:, 2], '--', color='orange', alpha=0.5)

        ax.set_xlim(mid[0] - half_range, mid[0] + half_range)
        ax.set_ylim(mid[1] - half_range, mid[1] + half_range)
        ax.set_zlim(mid[2] - half_range, mid[2] + half_range)
        ax.set_box_aspect([1, 1, 1])

        ax.set_title(f'kinova{kinova_id}_franka{franka_id}->target{target_id} '
                     f'Frame {frame}/{traj.shape[0]}')
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.view_init(elev=20, azim=-60)

    frames = range(0, traj.shape[0], skip)
    anim = FuncAnimation(fig, update, frames=frames, interval=50)

    if save_path:
        # Default: real-time at dt=0.01 (100 fps / skip)
        if fps is None:
            fps = int(100 / skip)
        anim.save(save_path, writer='ffmpeg', fps=fps)
        print(f"Saved animation to {save_path}")
    else:
        plt.show()
    return anim


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Child-branch thread insertion optimization (BDLO5)')
    parser.add_argument('--kinova', type=int, default=1, help='Kinova config ID (1-4)')
    parser.add_argument('--franka', type=int, default=1, help='Franka config ID (1-5)')
    parser.add_argument('--target', type=int, default=1, help='Target hole ID (1-2)')
    parser.add_argument('--checkpoint', type=str, default='',
                        help='Model checkpoint path (empty = no checkpoint, pure physics)')
    parser.add_argument('--offset', type=float, default=0.025, help='Target offset behind hole (stage 2)')
    parser.add_argument('--align', type=float, default=0.025, help='Alignment distance in front of hole (stage 1)')
    parser.add_argument('--vis-only', action='store_true', help='Only visualize initial shape, no optimization')
    parser.add_argument('--undeform-only', action='store_true',
                        help='Only visualize the undeformed BDLO5 reference pose, no data load or optimization')
    args = parser.parse_args()

    if args.undeform_only:
        torch.set_default_dtype(torch.float64)
        # Build the sim purely to grab the rest pose after the BDLO5 coord
        # transform. We don't need any of the loaded data.
        deft_sim, _, _, _ = setup_deft_sim(
            torch.tensor((0, 1, -2, -1)), load_checkpoint=None
        )
        undeformed = deft_sim.undeformed_vert.detach().cpu().numpy()  # (n_branch, n_vert, 3)
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        branch_colors = ['red', 'blue', 'green']
        branch_labels = ['Parent', 'Child1', 'Child2']
        branch_nv = [N_PARENT, N_CHILD1, N_CHILD2]
        for bi in range(N_BRANCH):
            v = undeformed[bi, :branch_nv[bi]]
            ax.plot(v[:, 0], v[:, 1], v[:, 2], 'o-', color=branch_colors[bi],
                    linewidth=2, markersize=5, label=branch_labels[bi])
        # Mark child2 tip
        ax.scatter(*undeformed[TIP_BRANCH, TIP_VERTEX], color='magenta', s=120,
                   marker='D', edgecolors='black', zorder=5, label='child2 tip')
        all_pts = np.concatenate([undeformed[bi, :branch_nv[bi]] for bi in range(N_BRANCH)], axis=0)
        mid = (all_pts.max(0) + all_pts.min(0)) / 2
        half = (all_pts.max(0) - all_pts.min(0)).max() / 2 * 1.1
        ax.set_xlim(mid[0] - half, mid[0] + half)
        ax.set_ylim(mid[1] - half, mid[1] + half)
        ax.set_zlim(mid[2] - half, mid[2] + half)
        ax.set_box_aspect([1, 1, 1])
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.set_title('BDLO5 Undeformed reference (post coord transform)')
        ax.legend(fontsize=8)
        ax.view_init(elev=20, azim=-60)
        print("Showing undeformed BDLO5 — close the window to exit.")
        plt.show()
        sys.exit(0)

    if args.vis_only:
        torch.set_default_dtype(torch.float64)
        out_dir = os.path.join(os.path.dirname(__file__), 'child_branch_insertion_output')
        os.makedirs(out_dir, exist_ok=True)

        flat_pts = load_mocap_config(args.kinova, args.franka)
        hole_pts = load_hole(args.target)
        hole_pts[:, 0] += 0.09  # shift hole 9cm toward x positive
        hole_pts[:, 0] = -hole_pts[:, 0]  # flip x AFTER the shift to match the DEFT BDLO5 frame
        initial_branched = mocap_to_branched_tensor(flat_pts)

        tip_position = flat_pts[17]
        hole_center, hole_normal = compute_hole_center_and_normal(hole_pts, tip_position)

        target_insert = hole_center + hole_normal * args.offset

        print(f"BDLO5 initial shape: {initial_branched.shape}")
        print(f"Parent (branch 0): vertices 0-{N_PARENT-1}")
        print(f"Child1 (branch 1): {N_CHILD1} vertices, couples at parent[{COUPLING_INDEX[0]}]")
        print(f"Child2 (branch 2): {N_CHILD2} vertices, couples at child1[{COUPLING_INDEX[1]}]")
        print(f"Child2 tip (branch {TIP_BRANCH}, vertex {TIP_VERTEX}): "
              f"{initial_branched[0, TIP_BRANCH, TIP_VERTEX].numpy()}")

        target_align = hole_center - hole_normal * args.align

        visualize_initial_state(initial_branched, target_insert, hole_pts, hole_center, hole_normal,
                                args.kinova, args.franka, args.target, target_align=target_align,
                                interactive=True)
        sys.exit(0)

    result = run_child_branch_insertion(
        kinova_id=args.kinova,
        franka_id=args.franka,
        target_id=args.target,
        checkpoint_path=args.checkpoint,
        offset_distance=args.offset,
        align_distance=args.align,
    )
    optimized_traj, initial_branched, target_pt, hole_pts, flat_pts, solution, info = result

    out_dir = os.path.join(os.path.dirname(__file__), 'child_branch_insertion_output')
    os.makedirs(out_dir, exist_ok=True)
    tag = f"kinova{args.kinova}_franka{args.franka}_target{args.target}"

    fig_path = os.path.join(out_dir, f'{tag}.png')
    visualize_result(optimized_traj, initial_branched, target_pt, hole_pts, flat_pts,
                     args.kinova, args.franka, args.target, save_path=fig_path)

    anim_path = os.path.join(out_dir, f'{tag}.mp4')
    animate_result(optimized_traj, initial_branched, target_pt, hole_pts,
                   args.kinova, args.franka, args.target, save_path=anim_path)
