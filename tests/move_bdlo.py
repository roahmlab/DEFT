#!/usr/bin/env python3
"""
Load BDLO1 data and run iterative_predict
"""

import torch
import torch.nn as nn
import sys
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
try:
    import cyipopt
    IPOPT_AVAILABLE = True
except ImportError:
    IPOPT_AVAILABLE = False
    print("Warning: cyipopt not available. Install with: pip install cyipopt")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deft.utils.util import DEFT_initialization, construct_b_DLOs, clamp_index, index_init, Eval_DEFTData
from deft.core.DEFT_sim import DEFT_sim
from torch.utils.data import DataLoader

# Global simulation time configuration
SIM_TIME_HORIZON = 100  # Adjust this to change simulation duration

def setup_deft_sim(load_checkpoint=None):
    """Setup DEFT simulation with BDLO1 (from DEFT_train.py)"""
    torch.set_default_dtype(torch.float64)
    device = "cpu"
    
    # BDLO1 configuration (from DEFT_train.py)
    undeformed_BDLO = torch.tensor([[[-0.6790, -0.6355, -0.5595, -0.4539, -0.3688, -0.2776, -0.1857,
                                      -0.0991, 0.0102, 0.0808, 0.1357, 0.2081, 0.2404, -0.4279,
                                      -0.4880, -0.5394, -0.5559, 0.0698, 0.0991, 0.1125]],
                                    [[0.0035, -0.0066, -0.0285, -0.0349, -0.0704, -0.0663, -0.0744,
                                      -0.0957, -0.0702, -0.0592, -0.0452, -0.0236, -0.0134, -0.0813,
                                      -0.1233, -0.1875, -0.2178, -0.1044, -0.1858, -0.2165]],
                                    [[0.0108, 0.0104, 0.0083, 0.0104, 0.0083, 0.0145, 0.0133,
                                      0.0198, 0.0155, 0.0231, 0.0199, 0.0154, 0.0169, 0.0160,
                                      0.0153, 0.0090, 0.0121, 0.0205, 0.0155, 0.0148]]]).permute(1, 2, 0)
    
    n_parent_vertices = 13
    n_child1_vertices = 5
    n_child2_vertices = 4
    n_branch = 3
    batch = 1
    
    parent_clamped_selection = torch.tensor((0, 1, -2, -1))
    child1_clamped_selection = torch.tensor((2))
    child2_clamped_selection = torch.tensor((2))
    
    # Extract vertices
    parent_vertices = undeformed_BDLO[:, :n_parent_vertices]
    child1_vertices = undeformed_BDLO[:, n_parent_vertices:n_parent_vertices + n_child1_vertices - 1]
    child2_vertices = undeformed_BDLO[:, n_parent_vertices + n_child1_vertices - 1:]
    
    # Initialize mass, MOI
    b_DLO_mass, parent_MOI, children_MOI, _, _, _ = DEFT_initialization(
        parent_vertices, child1_vertices, child2_vertices, n_branch, n_parent_vertices,
        (n_child1_vertices, n_child2_vertices), [4, 8], 1.0, 10.0, (0.5, 0.5), (1, 1), 0.1
    )
    
    # Construct BDLO
    b_DLOs_vertices, _ = construct_b_DLOs(
        batch, [4, 8], n_parent_vertices, (n_child1_vertices, n_child2_vertices), n_branch,
        parent_vertices, parent_vertices, child1_vertices, child1_vertices, child2_vertices, child2_vertices
    )
    
    # Transform coordinates
    b_DLOs_transform = torch.zeros_like(b_DLOs_vertices)
    b_DLOs_transform[:, :, :, 0] = -b_DLOs_vertices[:, :, :, 2]
    b_DLOs_transform[:, :, :, 1] = -b_DLOs_vertices[:, :, :, 0]
    b_DLOs_transform[:, :, :, 2] = b_DLOs_vertices[:, :, :, 1]
    
    b_undeformed_vert = b_DLOs_transform[0].view(n_branch, -1, 3)
    
    index_selection1, index_selection2, parent_MOI_index1, parent_MOI_index2 = index_init([4, 8], n_branch)
    clamped_index, parent_theta_clamp, child1_theta_clamp, child2_theta_clamp = clamp_index(
        batch, parent_clamped_selection, child1_clamped_selection, child2_clamped_selection,
        n_branch, n_parent_vertices, True, False, False
    )
    
    # Physical parameters (from DEFT_train.py)
    bend_stiffness_parent = nn.Parameter(4e-3 * torch.ones((1, 1, n_parent_vertices - 1), device=device))
    bend_stiffness_child1 = nn.Parameter(4e-3 * torch.ones((1, 1, n_parent_vertices - 1), device=device))
    bend_stiffness_child2 = nn.Parameter(4e-3 * torch.ones((1, 1, n_parent_vertices - 1), device=device))
    twist_stiffness = nn.Parameter(1e-4 * torch.ones((1, n_branch, n_parent_vertices - 1), device=device))
    damping = nn.Parameter(torch.tensor((2.5, 2.5, 2.5), device=device))
    learning_weight = nn.Parameter(torch.tensor(0.02, device=device))
    
    # Create DEFT simulation
    deft_sim = DEFT_sim(
        batch=batch, n_branch=n_branch, n_vert=n_parent_vertices, cs_n_vert=(n_child1_vertices, n_child2_vertices),
        b_init_n_vert=b_undeformed_vert, n_edge=n_parent_vertices - 1, b_undeformed_vert=b_undeformed_vert,
        b_DLO_mass=b_DLO_mass, parent_DLO_MOI=parent_MOI, children_DLO_MOI=children_MOI, device=device,
        clamped_index=clamped_index, rigid_body_coupling_index=[4, 8], parent_MOI_index1=parent_MOI_index1,
        parent_MOI_index2=parent_MOI_index2, parent_clamped_selection=parent_clamped_selection,
        child1_clamped_selection=child1_clamped_selection, child2_clamped_selection=child2_clamped_selection,
        clamp_parent=True, clamp_child1=False, clamp_child2=False, index_selection1=index_selection1,
        index_selection2=index_selection2, bend_stiffness_parent=bend_stiffness_parent,
        bend_stiffness_child1=bend_stiffness_child1, bend_stiffness_child2=bend_stiffness_child2,
        twist_stiffness=twist_stiffness, damping=damping, learning_weight=learning_weight
    )
    
    # Load checkpoint if provided
    if load_checkpoint:
        checkpoint = torch.load(load_checkpoint, map_location=device)
        deft_sim.load_state_dict(checkpoint, strict=False)
        print(f"Loaded checkpoint from {load_checkpoint}")
    
    return deft_sim, b_undeformed_vert, parent_theta_clamp, child1_theta_clamp, child2_theta_clamp


def create_custom_trajectory(initial_state, time_horizon, parent_clamped_selection):
    """Create custom control trajectory for clamped vertices only"""
    # Extract only clamped vertices: [batch, time, n_clamped, 3]
    batch = initial_state.shape[0]
    n_clamped = len(parent_clamped_selection)
    custom_traj = torch.zeros(batch, time_horizon, n_clamped, 3, dtype=torch.float64)
    
    # Initialize with initial positions of clamped vertices
    initial_clamped = initial_state[0, 0, 0, parent_clamped_selection]
    
    # Simple linear motion: first two stay fixed, last two move slowly in +x direction
    for t in range(time_horizon):
        custom_traj[0, t] = initial_clamped.clone()
        
        # Move last two vertices slowly in x direction
        progress = t / time_horizon
        custom_traj[0, t, -2, 0] += 0.3 * progress
        custom_traj[0, t, -1, 0] += 0.3 * progress

        custom_traj[0, t, 0, 0] -= 0.3 * progress
        custom_traj[0, t, 1, 0] -= 0.3 * progress
    return custom_traj

def compute_gradient_wrt_control(deft_sim, b_DLOs_vertices_traj, previous_b_DLOs_vertices_traj, 
                                  target_vertices, custom_control, parent_theta_clamp, 
                                  child1_theta_clamp, child2_theta_clamp):
    """Compute gradient of final position cost w.r.t. reduced 6 DOF displacements"""
    batch = 1
    parent_clamped_selection = torch.tensor((0, 1, 11, 12))
    
    # Get initial and final positions from custom_control
    initial_clamped = custom_control[0, 0]  # [n_clamped, 3]
    final_clamped = custom_control[0, -1]   # [n_clamped, 3]
    
    # Compute full 12 DOF displacements
    full_displacements = (final_clamped - initial_clamped).clone()
    
    # Reduce to 6 DOF: average pairs (0,1) and (11,12)
    reduced_displacements = torch.stack([
        (full_displacements[0] + full_displacements[1]) / 2,  # avg of vertices 0,1
        (full_displacements[2] + full_displacements[3]) / 2   # avg of vertices 11,12
    ]).requires_grad_(True)
    
    # Expand back to 12 DOF
    final_displacements = torch.stack([
        reduced_displacements[0], reduced_displacements[0],  # vertices 0,1
        reduced_displacements[1], reduced_displacements[1]   # vertices 11,12
    ])
    
    # Interpolate trajectory from initial to final
    clamped_full = torch.zeros(batch, SIM_TIME_HORIZON, 3, 13, 3, dtype=torch.float64)
    for t in range(SIM_TIME_HORIZON):
        alpha = t / (SIM_TIME_HORIZON - 1)
        interpolated = initial_clamped + alpha * final_displacements
        for i, idx in enumerate(parent_clamped_selection):
            clamped_full[0, t, 0, idx] = interpolated[i]
    
    predicted_vertices, _ = deft_sim.iterative_predict(
        time_horizon=SIM_TIME_HORIZON,
        b_DLOs_vertices_traj=b_DLOs_vertices_traj,
        previous_b_DLOs_vertices_traj=previous_b_DLOs_vertices_traj,
        clamped_positions=clamped_full,
        dt=0.01,
        parent_theta_clamp=parent_theta_clamp,
        child1_theta_clamp=child1_theta_clamp,
        child2_theta_clamp=child2_theta_clamp,
        inference_1_batch=False
    )
    
    final_pred = predicted_vertices[0, -1]
    final_target = target_vertices[0, -1]
    cost = torch.sum((final_pred - final_target) ** 2)
    
    # Gradient w.r.t. reduced 6 DOF
    grad = torch.autograd.grad(cost, reduced_displacements)[0]
    
    return grad.detach(), cost.item()

def finite_difference_gradient(deft_sim, b_DLOs_vertices_traj, previous_b_DLOs_vertices_traj,
                               target_vertices, custom_control, parent_theta_clamp,
                               child1_theta_clamp, child2_theta_clamp, eps=1e-6):
    """Compute gradient using finite differences for verification (reduced 6 DOF)"""
    parent_clamped_selection = torch.tensor((0, 1, -2, -1))
    
    # Get initial and final positions
    initial_clamped = custom_control[0, 0]
    final_clamped = custom_control[0, -1]
    full_displacements = final_clamped - initial_clamped
    
    # Reduce to 6 DOF
    reduced_displacements = torch.stack([
        (full_displacements[0] + full_displacements[1]) / 2,
        (full_displacements[2] + full_displacements[3]) / 2
    ])
    
    def compute_cost(reduced_disp):
        # Expand to 12 DOF
        expanded = torch.stack([
            reduced_disp[0], reduced_disp[0],
            reduced_disp[1], reduced_disp[1]
        ])
        
        with torch.no_grad():
            clamped_full = torch.zeros(1, SIM_TIME_HORIZON, 3, 13, 3, dtype=torch.float64)
            for t in range(SIM_TIME_HORIZON):
                alpha = t / (SIM_TIME_HORIZON - 1)
                interpolated = initial_clamped + alpha * expanded
                for i, idx in enumerate(parent_clamped_selection):
                    clamped_full[0, t, 0, idx] = interpolated[i]
            
            pred, _ = deft_sim.iterative_predict(
                time_horizon=SIM_TIME_HORIZON, b_DLOs_vertices_traj=b_DLOs_vertices_traj,
                previous_b_DLOs_vertices_traj=previous_b_DLOs_vertices_traj,
                clamped_positions=clamped_full, dt=0.01,
                parent_theta_clamp=parent_theta_clamp,
                child1_theta_clamp=child1_theta_clamp,
                child2_theta_clamp=child2_theta_clamp,
                inference_1_batch=False
            )
            return torch.sum((pred[0, -1] - target_vertices[0, -1]) ** 2).item()
    
    fd_grad = torch.zeros_like(reduced_displacements)
    test_indices = [(0, 0), (0, 2), (1, 1)]  # Test 3 components of reduced 6 DOF
    
    for idx in test_indices:
        disp_plus = reduced_displacements.clone()
        disp_plus[idx] += eps
        cost_plus = compute_cost(disp_plus)
        
        disp_minus = reduced_displacements.clone()
        disp_minus[idx] -= eps
        cost_minus = compute_cost(disp_minus)
        
        fd_grad[idx] = (cost_plus - cost_minus) / (2 * eps)
    
    return fd_grad, test_indices, eps

class DEFTOptimizationProblem:
    """IPOPT problem wrapper for DEFT trajectory optimization"""
    def __init__(self, deft_sim, b_DLOs_vertices_traj, previous_b_DLOs_vertices_traj,
                 target_vertices, parent_theta_clamp, child1_theta_clamp, child2_theta_clamp):
        self.deft_sim = deft_sim
        # Disable gradients for model parameters
        self.deft_sim.eval()
        for param in self.deft_sim.parameters():
            param.requires_grad = False
        
        self.b_DLOs_vertices_traj = b_DLOs_vertices_traj.detach()
        self.previous_b_DLOs_vertices_traj = previous_b_DLOs_vertices_traj.detach()
        self.target_vertices = target_vertices.detach()
        self.parent_theta_clamp = parent_theta_clamp
        self.child1_theta_clamp = child1_theta_clamp
        self.child2_theta_clamp = child2_theta_clamp
        self.parent_clamped_selection = torch.tensor((0, 1, 11, 12))
        self.initial_clamped = b_DLOs_vertices_traj[0, 0, 0, self.parent_clamped_selection].detach().clone()
        self.n = 6  # 2 independent displacements * 3 coords (vertices 0&1 move together, 11&12 move together)
        self.iteration = 0
        self.converged = False
        self.best_solution = None  # Store solution when converged
        
    def _expand_displacements(self, x):
        """Expand 6 DOF to 12 DOF by coupling vertex pairs"""
        # x is [2, 3]: displacement for (0,1) and (11,12)
        disp_pair1 = x[0]  # displacement for vertices 0 and 1
        disp_pair2 = x[1]  # displacement for vertices 11 and 12
        # Return [4, 3]: same displacement for each pair
        return torch.stack([disp_pair1, disp_pair1, disp_pair2, disp_pair2])
        
    def objective(self, x):
        """Compute objective: final configuration error"""
        # Expand 6 DOF to 12 DOF
        final_displacements = self._expand_displacements(torch.from_numpy(x.reshape(2, 3)))
        
        # Build trajectory
        clamped_full = torch.zeros(1, SIM_TIME_HORIZON, 3, 13, 3, dtype=torch.float64)
        for t in range(SIM_TIME_HORIZON):
            alpha = t / (SIM_TIME_HORIZON - 1)
            interpolated = self.initial_clamped + alpha * final_displacements
            for i, idx in enumerate(self.parent_clamped_selection):
                clamped_full[0, t, 0, idx] = interpolated[i]
        
        with torch.no_grad():
            predicted_vertices, _ = self.deft_sim.iterative_predict(
                time_horizon=SIM_TIME_HORIZON,
                b_DLOs_vertices_traj=self.b_DLOs_vertices_traj,
                previous_b_DLOs_vertices_traj=self.previous_b_DLOs_vertices_traj,
                clamped_positions=clamped_full,
                dt=0.01,
                parent_theta_clamp=self.parent_theta_clamp,
                child1_theta_clamp=self.child1_theta_clamp,
                child2_theta_clamp=self.child2_theta_clamp,
                inference_1_batch=False
            )
        
        cost = torch.sum((predicted_vertices[0, -1] - self.target_vertices[0, -1]) ** 2)
        
        # Compute endpoint distances
        pred_ends = predicted_vertices[0, -1, 0, self.parent_clamped_selection]
        target_ends = self.target_vertices[0, -1, 0, self.parent_clamped_selection]
        distances = torch.norm(pred_ends - target_ends, dim=1)
        
        print(f"  [Iter {self.iteration}] Endpoint distances: {distances.numpy()}")
        
        # Early stopping: all endpoints within 0.02
        if torch.all(distances < 0.06) and not self.converged:
            print(f"  *** Converged! All endpoints within 0.025 ***")
            self.converged = True
            self.best_solution = x.copy()  # Save the converged solution
        
        self.iteration += 1
        return cost.item()
    
    def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu,
                    d_norm, regularization_size, alpha_du, alpha_pr, ls_trials):
        """Intermediate callback for early stopping"""
        if self.converged:
            return False  # Stop optimization
        return True  # Continue optimization
    
    def gradient(self, x):
        """Compute gradient using PyTorch autograd"""
        # Create 6 DOF variables
        reduced_displacements = torch.from_numpy(x.reshape(2, 3)).double().requires_grad_(True)
        
        # Expand to 12 DOF
        final_displacements = self._expand_displacements(reduced_displacements)

        clamped_full = torch.zeros(1, SIM_TIME_HORIZON, 3, 13, 3, dtype=torch.float64)
        for t in range(SIM_TIME_HORIZON):
            alpha = t / (SIM_TIME_HORIZON - 1)
            interpolated = self.initial_clamped + alpha * final_displacements
            for i, idx in enumerate(self.parent_clamped_selection):
                clamped_full[0, t, 0, idx] = interpolated[i]

        predicted_vertices, _ = self.deft_sim.iterative_predict(
            time_horizon=SIM_TIME_HORIZON,
            b_DLOs_vertices_traj=self.b_DLOs_vertices_traj,
            previous_b_DLOs_vertices_traj=self.previous_b_DLOs_vertices_traj,
            clamped_positions=clamped_full,
            dt=0.01,
            parent_theta_clamp=self.parent_theta_clamp,
            child1_theta_clamp=self.child1_theta_clamp,
            child2_theta_clamp=self.child2_theta_clamp,
            inference_1_batch=False
        )

        cost = torch.sum((predicted_vertices[0, -1] - self.target_vertices[0, -1]) ** 2)
        
        # Gradient w.r.t. reduced 6 DOF
        grad = torch.autograd.grad(cost, reduced_displacements)[0]
        return grad.detach().numpy().flatten()

def trajectory_optimization_ipopt(deft_sim, b_DLOs_vertices_traj, previous_b_DLOs_vertices_traj,
                                  target_vertices, parent_theta_clamp, child1_theta_clamp, child2_theta_clamp):
    """Optimize trajectory using IPOPT"""
    if not IPOPT_AVAILABLE:
        raise ImportError("cyipopt not available. Install with: pip install cyipopt")
    
    # Create problem
    problem_obj = DEFTOptimizationProblem(
        deft_sim, b_DLOs_vertices_traj, previous_b_DLOs_vertices_traj,
        target_vertices, parent_theta_clamp, child1_theta_clamp, child2_theta_clamp
    )
    
    n = 6  # 2 independent displacements * 3 coords
    lb = [-1e20] * n
    ub = [1e20] * n
    
    # Initial guess: zero displacements (stay at initial positions)
    x0 = np.zeros(n)
    
    # Create IPOPT problem
    problem = cyipopt.Problem(
        n=n,
        m=0,  # No constraints
        problem_obj=problem_obj,
        lb=lb,
        ub=ub
    )
    
    # Set options
    problem.add_option("print_level", 5)
    problem.add_option("tol", 1e-3)
    problem.add_option("max_iter", 100)
    
    print("\nStarting IPOPT optimization...")
    solution, info = problem.solve(x0)
    
    # Use saved solution if early stopping occurred
    if problem_obj.best_solution is not None:
        print("Using early-stopped solution (distances < 0.02)")
        solution = problem_obj.best_solution
    
    print(f"\nIPOPT Status: {info['status_msg']}")
    print(f"Final objective: {info['obj_val']:.6f}")
    
    # Generate optimized trajectory for visualization
    final_displacements = problem_obj._expand_displacements(torch.from_numpy(solution.reshape(2, 3)))
    clamped_full = torch.zeros(1, SIM_TIME_HORIZON, 3, 13, 3, dtype=torch.float64)
    for t in range(SIM_TIME_HORIZON):
        alpha = t / (SIM_TIME_HORIZON - 1)
        interpolated = problem_obj.initial_clamped + alpha * final_displacements
        for i, idx in enumerate(problem_obj.parent_clamped_selection):
            clamped_full[0, t, 0, idx] = interpolated[i]
    
    with torch.no_grad():
        optimized_traj, _ = problem_obj.deft_sim.iterative_predict(
            time_horizon=SIM_TIME_HORIZON,
            b_DLOs_vertices_traj=b_DLOs_vertices_traj,
            previous_b_DLOs_vertices_traj=previous_b_DLOs_vertices_traj,
            clamped_positions=clamped_full,
            dt=0.01,
            parent_theta_clamp=parent_theta_clamp,
            child1_theta_clamp=child1_theta_clamp,
            child2_theta_clamp=child2_theta_clamp,
            inference_1_batch=False
        )
    
    return solution.reshape(2, 3), info, optimized_traj

def move_bdlo_with_data(checkpoint_path=None):
    """Load real data from dataset and run prediction"""
    torch.set_default_dtype(torch.float64)
    device = "cpu"
    
    # Setup DEFT sim with checkpoint
    deft_sim, _, parent_theta_clamp, child1_theta_clamp, child2_theta_clamp = setup_deft_sim(load_checkpoint=checkpoint_path)
    
    # Load evaluation dataset (from DEFT_train.py)
    eval_dataset = Eval_DEFTData(
        BDLO_type=1,
        n_parent_vertices=13,
        n_children_vertices=(5, 4),
        n_branch=3,
        rigid_body_coupling_index=[4, 8],
        eval_set_number=1,
        total_time=500,
        eval_time_horizon=498,
        device=device
    )
    
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)
    
    # Get first sample only
    previous_b_DLOs_vertices_traj, b_DLOs_vertices_traj, target_b_DLOs_vertices_traj = next(iter(eval_loader))
    
    # Create custom control trajectory instead of using target
    parent_clamped_selection = torch.tensor((0, 1, -2, -1))
    custom_control = create_custom_trajectory(b_DLOs_vertices_traj, SIM_TIME_HORIZON, parent_clamped_selection)
    
    print(f"Processing 1 sample: {b_DLOs_vertices_traj.shape}")
    
    with torch.no_grad():
        batch = 1
        clamped_full = torch.zeros(batch, SIM_TIME_HORIZON, 3, 13, 3, dtype=torch.float64)
        clamped_full[:, :, 0, parent_clamped_selection] = custom_control
        
        predicted_vertices, predicted_velocities = deft_sim.iterative_predict(
            time_horizon=SIM_TIME_HORIZON,
            b_DLOs_vertices_traj=b_DLOs_vertices_traj,
            previous_b_DLOs_vertices_traj=previous_b_DLOs_vertices_traj,
            clamped_positions=clamped_full,
            dt=0.01,
            parent_theta_clamp=parent_theta_clamp,
            child1_theta_clamp=child1_theta_clamp,
            child2_theta_clamp=child2_theta_clamp,
            inference_1_batch=False
        )
    
    print(f"Prediction completed: {predicted_vertices.shape}")
    return predicted_vertices, predicted_velocities, target_b_DLOs_vertices_traj

def animate_prediction(predicted_vertices, target_vertices, skip_frames=5, title_prefix="Prediction"):
    """Create animation showing optimization process moving toward static target"""
    pred = predicted_vertices[0]  # [time, branch, vert, 3]
    target = target_vertices[0, -1]  # [branch, vert, 3] - static final configuration
    
    print(f"\n[Animation] Target: target_vertices[0, -1]")
    print(f"  Shape: {target.shape}")
    print(f"  Branch 0 endpoints: {target[0, [0,1,11,12]]}")
    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    colors = ['red', 'green', 'blue']
    
    def update(frame):
        ax.clear()
        
        # Plot static target (faded)
        for i in range(3):
            verts = target[i]
            mask = torch.any(verts != 0, dim=-1)
            valid = verts[mask]
            if len(valid) > 0:
                ax.plot(valid[:, 0], valid[:, 1], valid[:, 2], 'o-', color=colors[i], 
                       linewidth=2, alpha=0.3, label=f'Target {i}')
        
        # Overlay current prediction (solid)
        for i in range(3):
            verts = pred[frame, i]
            mask = torch.any(verts != 0, dim=-1)
            valid = verts[mask]
            if len(valid) > 0:
                ax.plot(valid[:, 0], valid[:, 1], valid[:, 2], 'o-', color=colors[i], 
                       linewidth=2, markersize=6, label=f'{title_prefix} {i}')
        
        ax.set_title(f'{title_prefix} (solid) → Target (faded) - Frame {frame}')
        ax.set_xlim(-0.8, 0.4)
        ax.set_ylim(-0.3, 0.3)
        ax.set_zlim(-0.1, 0.3)
        ax.legend()
    
    frames = range(0, pred.shape[0], skip_frames)
    anim = FuncAnimation(fig, update, frames=frames, interval=50)
    plt.tight_layout()
    return anim

if __name__ == "__main__":
    checkpoint_path = "/home/yizhouch/DEFT_2025/save_model/DEFT_ends_1_3520_1.pth"
    
    print("Loading and predicting with trained model...")
    pred_verts, pred_vels, target = move_bdlo_with_data(checkpoint_path=checkpoint_path)
    print(f"Done!")
    
    # Compute gradients
    print("\nComputing gradients w.r.t. control inputs...")
    torch.set_default_dtype(torch.float64)
    deft_sim, _, parent_theta_clamp, child1_theta_clamp, child2_theta_clamp = setup_deft_sim(load_checkpoint=checkpoint_path)
    
    eval_dataset = Eval_DEFTData(
        BDLO_type=1, n_parent_vertices=13, n_children_vertices=(5, 4),
        n_branch=3, rigid_body_coupling_index=[4, 8],
        eval_set_number=1, total_time=500, eval_time_horizon=498, device="cpu"
    )
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)
    previous_b_DLOs_vertices_traj, b_DLOs_vertices_traj, target_b_DLOs_vertices_traj = next(iter(eval_loader))
    
    parent_clamped_selection = torch.tensor((0, 1, -2, -1))
    custom_control = create_custom_trajectory(b_DLOs_vertices_traj, SIM_TIME_HORIZON, parent_clamped_selection)
    
    grad, cost = compute_gradient_wrt_control(
        deft_sim, b_DLOs_vertices_traj, previous_b_DLOs_vertices_traj,
        target_b_DLOs_vertices_traj, custom_control,
        parent_theta_clamp, child1_theta_clamp, child2_theta_clamp
    )
    print(f"Cost: {cost:.6f}")
    print(f"Gradient shape: {grad.shape}  # [2, 3] = 6 DOF (2 independent displacements)")
    print(f"Gradient norm: {torch.norm(grad).item():.6f}")
    
    # Verify with finite differences
    fd_grad, test_indices, eps = finite_difference_gradient(
        deft_sim, b_DLOs_vertices_traj, previous_b_DLOs_vertices_traj,
        target_b_DLOs_vertices_traj, custom_control,
        parent_theta_clamp, child1_theta_clamp, child2_theta_clamp
    )
    
    print(f"\nGradient comparison (eps={eps}):")
    for idx in test_indices:
        ag = grad[idx].item()
        fd = fd_grad[idx].item()
        print(f"  Pair {idx[0]}, coord {idx[1]}: Autograd={ag:.6f}, FiniteDiff={fd:.6f}, Diff={abs(ag-fd):.6f}")
    
    # Run IPOPT optimization
    if IPOPT_AVAILABLE:
        print("\n" + "="*60)
        print("IPOPT TRAJECTORY OPTIMIZATION")
        print("="*60)
        import time
        start_time = time.time()
        optimized_displacements, info, optimized_traj = trajectory_optimization_ipopt(
            deft_sim, b_DLOs_vertices_traj, previous_b_DLOs_vertices_traj,
            target_b_DLOs_vertices_traj, parent_theta_clamp, child1_theta_clamp, child2_theta_clamp
        )
        elapsed_time = time.time() - start_time
        
        print(f"\nOptimized displacements shape: {optimized_displacements.shape}")
        print(f"\n*** OPTIMIZATION TIME: {elapsed_time:.3f} seconds ***")
        
        print("\nCreating animation of optimized trajectory...")
        anim = animate_prediction(optimized_traj, target_b_DLOs_vertices_traj, title_prefix="Optimized")
        anim.save('optimized_trajectory.mp4', writer='ffmpeg', fps=20)
        print("Animation saved to: optimized_trajectory.mp4")
        plt.show()
    else:
        print("\nIPOPT not available. Skipping IPOPT optimization.")
        print("\nCreating animation...")
        anim = animate_prediction(pred_verts, target)
        plt.show()

    