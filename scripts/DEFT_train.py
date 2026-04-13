# Importing necessary libraries and modules
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
# Importing custom utility functions and classes
# DEFT_initialization: Initializes mass, MOI, rod orientation, etc. for the BDLO
# construct_b_DLOs: Constructs undeformed states for the BDLO
# clamp_index: Builds the necessary clamp indices for boundary condition enforcement
# index_init: Initializes certain indexing variables for the model
# save_pickle: Utility function to save data (e.g., losses) to a pickle file
# Train_DEFTData / Eval_DEFTData: Custom dataset classes to load training/evaluation data
# DEFT_sim: Simulation model class
import sys
import os
import glob
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deft.utils.util import DEFT_initialization, construct_b_DLOs, clamp_index, index_init, save_pickle, Train_DEFTData, Eval_DEFTData
from deft.core.DEFT_sim import DEFT_sim
from tqdm import tqdm
import os
import argparse
import matplotlib.pyplot as plt

# DEBUG: enable autograd anomaly detection so any in-place mutation that breaks
# backward shows the forward stack trace where the offending tensor was created.
# torch.autograd.set_detect_anomaly(True)


def train(train_batch, BDLO_type, total_time, train_time_horizon, undeform_vis, inference_vis, inference_1_batch,
          residual_learning, clamp_type, load_model, training_mode,
          use_orientation_constraints, use_attachment_constraints):
    # The total_time parameter is the maximum timesteps of the loaded data
    # The train_time_horizon is how many timesteps to unroll the simulation during training
    # The function trains or partially fine-tunes a DEFT model for a specific branched BDLO type

    eval_time_horizon = total_time - 2  # Number of timesteps for evaluation

    # Explanation of notation in the code:
    # - undeformed_BDLO: A tensor containing the initial (undeformed) vertex positions of the branched BDLO
    # - n_parent_vertices / n_child1_vertices / n_child2_vertices: The number of vertices for the main branch, child1, and child2, respectively

    # Prepare BDLO-specific data depending on BDLO_type
    if BDLO_type == 1:
        # Set the undeformed shape of BDLO1 as a tensor of shape [1, 20, 3], then permute to [n_parent_vertices+..., 3]
        undeformed_BDLO = torch.tensor([[[-0.6790, -0.6355, -0.5595, -0.4539, -0.3688, -0.2776, -0.1857,
                                          -0.0991, 0.0102, 0.0808, 0.1357, 0.2081, 0.2404, -0.4279,
                                          -0.4880, -0.5394, -0.5559, 0.0698, 0.0991, 0.1125]],
                                        [[0.0035, -0.0066, -0.0285, -0.0349, -0.0704, -0.0663, -0.0744,
                                          -0.0957, -0.0702, -0.0592, -0.0452, -0.0236, -0.0134, -0.0813,
                                          -0.1233, -0.1875, -0.2178, -0.1044, -0.1858, -0.2165]],
                                        [[0.0108, 0.0104, 0.0083, 0.0104, 0.0083, 0.0145, 0.0133,
                                          0.0198, 0.0155, 0.0231, 0.0199, 0.0154, 0.0169, 0.0160,
                                          0.0153, 0.0090, 0.0121, 0.0205, 0.0155, 0.0148]]]).permute(1, 2, 0)

        # Number of vertices along the parent branch and the two child branches
        n_parent_vertices = 13
        n_child1_vertices = 5
        n_child2_vertices = 4

        # Depending on clamp_type, we set the train/eval dataset sizes and the selection of clamped vertices
        if clamp_type == "ends":
            train_set_number = 77
            eval_set_number = 24
            parent_clamped_selection = torch.tensor((0, 1, -2, -1))
            child1_clamped_selection = torch.tensor((2))
            child2_clamped_selection = torch.tensor((2))
        else:
            train_set_number = 71
            eval_set_number = 18
            parent_clamped_selection = torch.tensor((2, -2, -1))
            child1_clamped_selection = torch.tensor((2))
            child2_clamped_selection = torch.tensor((2))

        # cs_n_vert holds the number of child1 and child2 vertices
        cs_n_vert = (n_child1_vertices, n_child2_vertices)
        # n_vert is the parent branch vertex count
        n_vert = n_parent_vertices
        # Number of edges in the parent branch is n_vert - 1
        n_edge = n_vert - 1

        # Sanity check: parent branch should have more vertices than any child branch
        if n_parent_vertices <= max(cs_n_vert):
            raise Exception("warning: number of parent's vertices is larger than children's!")

        # Define the stiffness parameters as nn.Parameters for optimization or subsequent usage
        bend_stiffness_parent = nn.Parameter(4e-3 * torch.ones((1, 1, n_edge), device=device))
        bend_stiffness_child1 = nn.Parameter(4e-3 * torch.ones((1, 1, n_edge), device=device))
        bend_stiffness_child2 = nn.Parameter(4e-3 * torch.ones((1, 1, n_edge), device=device))
        twist_stiffness = nn.Parameter(1e-4 * torch.ones((1, n_branch, n_edge), device=device))

        # Damping parameters for each branch
        damping = nn.Parameter(torch.tensor((2.5, 2.5, 2.5), device=device))

        # If we use residual learning, learning_weight is used to scale the residual from the GNN
        if residual_learning:
            learning_weight = nn.Parameter(torch.tensor(0.1, device=device))
        else:
            learning_weight = nn.Parameter(torch.tensor(0.00, device=device))

        # Indices that define which vertices couple the child branches to the parent
        rigid_body_coupling_index = [4, 8]

        # Mass and moment-of-inertia scaling factors
        parent_mass_scale = 1.
        parent_moment_scale = 10.
        moment_ratio = 0.1
        children_moment_scale = (0.5, 0.5)
        children_mass_scale = (1, 1)
        bdlo5 = False

    if BDLO_type == 2:
        # Similar initialization for BDLO2
        undeformed_BDLO = torch.tensor([
            [[0.0150, 0.0157, 0.0125, 0.0109, 0.0164, 0.0131, 0.0104,
              0.0081, 0.0083, 0.0079, 0.0093, 0.0108, 0.0150, 0.0109,
              0.0116, 0.0110, 0.0111, 0.0084, 0.0103, 0.0097]],
            [[0.1521, 0.1426, 0.1021, 0.0928, 0.0882, 0.0711, 0.0678,
              0.0894, 0.1109, 0.1374, 0.1708, 0.1855, 0.0339, -0.0410,
              -0.1058, -0.1266, 0.0465, -0.0155, -0.0733, -0.0954]],
            [[-0.1706, -0.1409, -0.0875, -0.0018, 0.0702, 0.1685, 0.2583,
              0.3363, 0.3894, 0.4529, 0.5106, 0.5355, 0.0704, 0.0615,
              0.0093, -0.0080, 0.3405, 0.3217, 0.2929, 0.2834]]
        ]).permute(1, 2, 0)

        n_parent_vertices = 12
        n_child1_vertices = 5
        n_child2_vertices = 5
        train_set_number = 110
        eval_set_number = 37

        cs_n_vert = (n_child1_vertices, n_child2_vertices)
        n_vert = n_parent_vertices
        n_edge = n_vert - 1
        if n_parent_vertices <= max(cs_n_vert):
            raise Exception("warning: number of parent's vertices is larger than children's!")

        # Stiffness parameters
        bend_stiffness_parent = nn.Parameter(2e-3 * torch.ones((1, 1, n_edge), device=device))
        bend_stiffness_child1 = nn.Parameter(1.5e-3 * torch.ones((1, 1, n_edge), device=device))
        bend_stiffness_child2 = nn.Parameter(1.5e-3 * torch.ones((1, 1, n_edge), device=device))
        twist_stiffness = nn.Parameter(1e-4 * torch.ones((1, n_branch, n_edge), device=device))

        # Damping parameters
        damping = nn.Parameter(torch.tensor((2.5, 2., 2.), device=device))

        if residual_learning:
            learning_weight = nn.Parameter(torch.tensor(0.1, device=device))
        else:
            learning_weight = nn.Parameter(torch.tensor(0.00, device=device))

        # Rigid body coupling index: the parent-children connection points
        rigid_body_coupling_index = [4, 7]

        # Mass, MOI scaling, etc.
        parent_mass_scale = 1.
        parent_moment_scale = 10.
        moment_ratio = 0.1
        children_moment_scale = (0.5, 0.5)
        children_mass_scale = (1, 1)

        # Which vertices are clamped in the dataset
        parent_clamped_selection = torch.tensor((0, 1, -2, -1))
        child1_clamped_selection = torch.tensor((2))
        child2_clamped_selection = torch.tensor((2))
        bdlo5 = False

    if BDLO_type == 3:
        # Initialization for BDLO3
        undeformed_BDLO = torch.tensor([
            [[0.0099, 0.0114, 0.0109, 0.0084, 0.0130, 0.0143, 0.0119,
              0.0133, 0.0135, 0.0136, 0.0136, 0.0151, 0.0124, 0.0093,
              0.0120, 0.0132, 0.0121]],
            [[-0.0444, -0.0684, -0.1235, -0.1722, -0.1973, -0.2265, -0.2232,
              -0.1956, -0.1675, -0.1150, -0.0544, -0.0249, -0.2632, -0.3340,
              -0.2580, -0.3370, -0.3594]],
            [[-0.5656, -0.5434, -0.4977, -0.4399, -0.3552, -0.2506, -0.1563,
              -0.0530, 0.0092, 0.0709, 0.1222, 0.1390, -0.3781, -0.4082,
              -0.0169, -0.0347, -0.0420]]
        ]).permute(1, 2, 0)

        n_parent_vertices = 12
        n_child1_vertices = 3
        n_child2_vertices = 4

        if clamp_type == "ends":
            train_set_number = 103
            eval_set_number = 26
            parent_clamped_selection = torch.tensor((0, 1, -2, -1))
            child1_clamped_selection = torch.tensor((2))
            child2_clamped_selection = torch.tensor((2))
        else:
            train_set_number = 39
            eval_set_number = 12
            parent_clamped_selection = torch.tensor((3, -2, -1))
            child1_clamped_selection = torch.tensor((2))
            child2_clamped_selection = torch.tensor((2))

        cs_n_vert = (n_child1_vertices, n_child2_vertices)
        n_vert = n_parent_vertices
        n_edge = n_vert - 1
        if n_parent_vertices <= max(cs_n_vert):
            raise Exception("warning: number of parent's vertices is larger than children's!")

        # Stiffness parameters
        bend_stiffness_parent = nn.Parameter(2.5e-3 * torch.ones((1, 1, n_edge), device=device, dtype=torch.float64))
        bend_stiffness_child1 = nn.Parameter(2.5e-3 * torch.ones((1, 1, n_edge), device=device, dtype=torch.float64))
        bend_stiffness_child2 = nn.Parameter(2.5e-3 * torch.ones((1, 1, n_edge), device=device, dtype=torch.float64))
        twist_stiffness = nn.Parameter(1e-4 * torch.ones((1, n_branch, n_edge), device=device, dtype=torch.float64))
        damping = nn.Parameter(torch.tensor((2., 2., 2.), device=device, dtype=torch.float64))

        if residual_learning:
            learning_weight = nn.Parameter(torch.tensor(0.1, device=device, dtype=torch.float64))
        else:
            learning_weight = nn.Parameter(torch.tensor(0.00, device=device, dtype=torch.float64))

        rigid_body_coupling_index = [4, 7]
        parent_mass_scale = 1.
        parent_moment_scale = 10.
        moment_ratio = 0.1
        children_moment_scale = (0.5, 0.5)
        children_mass_scale = (1, 1)
        bdlo5 = False

    if BDLO_type == 4:
        # Initialization for BDLO4
        undeformed_BDLO = torch.tensor([
            [[0.0108, 0.0122, 0.0112, 0.0116, 0.0116, 0.0169, 0.0122,
              0.0198, 0.0173, 0.0140, 0.0152, 0.0156, 0.0120, 0.0107,
              0.0163, 0.0154]],
            [[-0.1680, -0.1938, -0.2439, -0.2991, -0.3230, -0.3345, -0.3376,
              -0.3248, -0.3100, -0.2727, -0.2182, -0.1878, -0.3922, -0.4643,
              -0.3866, -0.4500]],
            [[-0.5774, -0.5491, -0.4909, -0.4085, -0.3219, -0.2371, -0.1568,
              -0.0645, 0.0231, 0.0828, 0.1411, 0.1664, -0.3430, -0.3652,
              0.0434, 0.0658]]
        ]).permute(1, 2, 0)

        n_parent_vertices = 12
        n_child1_vertices = 3
        n_child2_vertices = 3
        train_set_number = 74
        eval_set_number = 25

        cs_n_vert = (n_child1_vertices, n_child2_vertices)
        n_vert = n_parent_vertices
        n_edge = n_vert - 1
        if n_parent_vertices <= max(cs_n_vert):
            raise Exception("warning: number of parent's vertices is larger than children's!")

        # Stiffness parameters
        bend_stiffness_parent = nn.Parameter(3e-3 * torch.ones((1, 1, n_edge), device=device, dtype=torch.float64))
        bend_stiffness_child1 = nn.Parameter(4e-3 * torch.ones((1, 1, n_edge), device=device, dtype=torch.float64))
        bend_stiffness_child2 = nn.Parameter(4e-3 * torch.ones((1, 1, n_edge), device=device, dtype=torch.float64))
        twist_stiffness = nn.Parameter(1e-4 * torch.ones((1, n_branch, n_edge), device=device, dtype=torch.float64))

        # Damping for each of the 3 branches
        damping = nn.Parameter(torch.tensor((3., 4, 4.), device=device, dtype=torch.float64))

        if residual_learning:
            learning_weight = nn.Parameter(torch.tensor(0.1, device=device, dtype=torch.float64))
        else:
            learning_weight = nn.Parameter(torch.tensor(0.00, device=device, dtype=torch.float64))

        # Connection indices for the child branches
        rigid_body_coupling_index = [4, 8]

        parent_mass_scale = 1.
        parent_moment_scale = 10.
        moment_ratio = 0.1
        children_moment_scale = (0.5, 0.5)
        children_mass_scale = (1, 1)

        parent_clamped_selection = torch.tensor((0, 1, -2, -1))
        child1_clamped_selection = torch.tensor((2))
        child2_clamped_selection = torch.tensor((2))
        bdlo5 = False

    if BDLO_type == 5:
        # BDLO5: child1 attaches to parent vertex 5, child2 attaches to child1 local vertex 1
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
        train_set_number = 75
        eval_set_number = 26

        cs_n_vert = (n_child1_vertices, n_child2_vertices)
        n_vert = n_parent_vertices
        n_edge = n_vert - 1
        if n_parent_vertices <= max(cs_n_vert):
            raise Exception("warning: number of parent's vertices is larger than children's!")

        bend_stiffness_parent = nn.Parameter(2e-3 * torch.ones((1, 1, n_edge), device=device, dtype=torch.float64))
        bend_stiffness_child1 = nn.Parameter(2e-3 * torch.ones((1, 1, n_edge), device=device, dtype=torch.float64))
        bend_stiffness_child2 = nn.Parameter(3e-3 * torch.ones((1, 1, n_edge), device=device, dtype=torch.float64))
        twist_stiffness = nn.Parameter(1e-4 * torch.ones((1, n_branch, n_edge), device=device, dtype=torch.float64))

        damping = nn.Parameter(torch.tensor((3., 3., 3.), device=device, dtype=torch.float64))

        if residual_learning:
            learning_weight = nn.Parameter(torch.tensor(0.1, device=device, dtype=torch.float64))
        else:
            learning_weight = nn.Parameter(torch.tensor(0.00, device=device, dtype=torch.float64))

        rigid_body_coupling_index = [5, 1]

        parent_mass_scale = 1.
        parent_moment_scale = 10.
        moment_ratio = 0.1
        children_moment_scale = (0.5, 0.5)
        children_mass_scale = (1, 1)

        parent_clamped_selection = torch.tensor((0, 1, -2, -1))
        child1_clamped_selection = torch.tensor((2))
        child2_clamped_selection = torch.tensor((2))
        bdlo5 = True

    if BDLO_type == 6:
        # BDLO6 — Stage 1: undeform vis only.
        #
        # Topology (4 branches):
        #   parent : 12 vertices
        #   child1 :  4 vertices, attaches at parent[2]   (3rd vertex)
        #   child2 :  2 vertices, attaches at parent[7]   (8th vertex)
        #   child3 :  3 vertices, attaches at parent[7]   (8th vertex)
        # Total stored vertices per frame: 12 + 4 + 2 + 3 = 21.
        #
        # We follow the BDLO1–5 convention: each assembled child[0] is forced
        # to equal its parent attachment vertex (the dataset's raw c[0] is
        # discarded). Both the undeformed pose and the trajectory loaders go
        # through the same `_bdlo6_split_and_pad` helper so they always match.
        from deft.utils.util import load_bdlo6_undeformed

        # Assembled per-branch vertex counts (after prepending parent[attach]
        # to each child, BDLO1–5-style). The dataset's raw stored counts are
        # (4, 2, 3) for c1/c2/c3; assembled counts are (5, 3, 4).
        n_parent_vertices = 12
        n_child1_vertices = 5
        n_child2_vertices = 3
        n_child3_vertices = 4
        n_branch_local    = 4   # parent + 3 children (don't override outer n_branch=3 used elsewhere)

        # `load_bdlo6_undeformed` applies the same coord-transform pipeline
        # (stage 1: BDLO5 swap + Rx(+90°), stage 2: Rx(-90°), stage 3: negate X)
        # as the dataset loaders, then assembles the 4-branch padded layout.
        b_undeformed_vert_bdlo6 = load_bdlo6_undeformed('../dataset/BDLO6_undeformed.pkl')

        if undeform_vis:
            branch_colors = ['red', 'green', 'blue', 'magenta']
            branch_names  = ['parent', 'c1', 'c2', 'c3']
            branch_nv     = [n_parent_vertices, n_child1_vertices, n_child2_vertices, n_child3_vertices]
            attach_idx    = [None, 2, 7, 7]   # c1→p[2], c2→p[7], c3→p[7]

            fig = plt.figure(figsize=(12, 9))
            ax = fig.add_subplot(111, projection='3d')
            for b_i in range(n_branch_local):
                pts = b_undeformed_vert_bdlo6[b_i, :branch_nv[b_i]].numpy()
                ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], '-o',
                        color=branch_colors[b_i],
                        label=f'{branch_names[b_i]} ({branch_nv[b_i]}v)',
                        markersize=4, linewidth=1.5)
            # Dashed lines from each child[0] to its declared parent attachment
            for b_i in range(1, n_branch_local):
                cp = b_undeformed_vert_bdlo6[b_i, 0].numpy()
                pp = b_undeformed_vert_bdlo6[0, attach_idx[b_i]].numpy()
                ax.plot([pp[0], cp[0]], [pp[1], cp[1]], [pp[2], cp[2]],
                        '--', color=branch_colors[b_i], linewidth=1, alpha=0.6)

            # Equal-scale axes so the rod isn't visually distorted.
            all_pts = np.concatenate([
                b_undeformed_vert_bdlo6[b_i, :branch_nv[b_i]].numpy()
                for b_i in range(n_branch_local)
            ], axis=0)
            margin = 0.05
            mins = all_pts.min(0) - margin
            maxs = all_pts.max(0) + margin
            max_range = float((maxs - mins).max())
            mids = (mins + maxs) / 2
            ax.set_xlim(mids[0] - max_range / 2, mids[0] + max_range / 2)
            ax.set_ylim(mids[1] - max_range / 2, mids[1] + max_range / 2)
            ax.set_zlim(mids[2] - max_range / 2, mids[2] + max_range / 2)
            ax.set_box_aspect((1, 1, 1))   # equal aspect ratio in matplotlib 3D

            ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
            ax.set_title('BDLO6 Undeformed reference pose')
            ax.legend(loc='upper right', fontsize=9)
            plt.show()

        # ----------------------------------------------------------------
        # Stage 4: BDLO6 self-contained train+eval path.
        # All BDLO6 code lives in this block and we early-return at the
        # end so the BDLO1–5 training loop below is never reached.
        # Only `--training_mode physics` is supported (the GNN_tree was
        # generalized to 4 branches but we don't have a BDLO6 residual
        # checkpoint and `learning_weight=0` keeps the residual term off).
        # ----------------------------------------------------------------
        from deft.utils.util import (
            DEFT_initialization_BDLO6,
            Eval_DEFTData_BDLO6,
            Train_DEFTData_BDLO6,
            BDLO6_RIGID_BODY_COUPLING_INDEX,
        )

        if training_mode not in ("physics", "residual"):
            raise NotImplementedError(
                f"BDLO6 supports --training_mode physics or residual (got '{training_mode}').")

        # Local hyperparameters mirroring BDLO5 defaults
        n_vert     = n_parent_vertices
        n_edge     = n_vert - 1
        cs_n_vert  = (n_child1_vertices, n_child2_vertices, n_child3_vertices)
        rigid_body_coupling_index = list(BDLO6_RIGID_BODY_COUPLING_INDEX)

        # Trainable physics params for the TRAIN sim
        bend_stiffness_parent = nn.Parameter(3e-3 * torch.ones((1, 1, n_edge), device=device, dtype=torch.float64))
        bend_stiffness_child1 = nn.Parameter(3e-3 * torch.ones((1, 1, n_edge), device=device, dtype=torch.float64))
        bend_stiffness_child2 = nn.Parameter(3e-3 * torch.ones((1, 1, n_edge), device=device, dtype=torch.float64))
        bend_stiffness_child3 = nn.Parameter(3e-3 * torch.ones((1, 1, n_edge), device=device, dtype=torch.float64))
        twist_stiffness       = nn.Parameter(1e-4 * torch.ones((1, n_branch_local, n_edge), device=device, dtype=torch.float64))
        damping               = nn.Parameter(torch.tensor((3., 3., 3., 3.), device=device, dtype=torch.float64))
        if residual_learning:
            learning_weight = nn.Parameter(torch.tensor(0.1, device=device, dtype=torch.float64))
        else:
            learning_weight = nn.Parameter(torch.tensor(0.0, device=device, dtype=torch.float64))

        # Separate eval params (fresh copies) so the eval sim doesn't hold references
        # to the training sim's autograd graph between train/eval calls.
        eval_bend_stiffness_parent = nn.Parameter(2e-3 * torch.ones((1, 1, n_edge), device=device, dtype=torch.float64))
        eval_bend_stiffness_child1 = nn.Parameter(2e-3 * torch.ones((1, 1, n_edge), device=device, dtype=torch.float64))
        eval_bend_stiffness_child2 = nn.Parameter(2e-3 * torch.ones((1, 1, n_edge), device=device, dtype=torch.float64))
        eval_bend_stiffness_child3 = nn.Parameter(2e-3 * torch.ones((1, 1, n_edge), device=device, dtype=torch.float64))
        eval_twist_stiffness       = nn.Parameter(1e-4 * torch.ones((1, n_branch_local, n_edge), device=device, dtype=torch.float64))
        eval_damping               = nn.Parameter(torch.tensor((3., 3., 3., 3.), device=device, dtype=torch.float64))
        if residual_learning:
            eval_learning_weight = nn.Parameter(torch.tensor(0.1, device=device, dtype=torch.float64))
        else:
            eval_learning_weight = nn.Parameter(torch.tensor(0.0, device=device, dtype=torch.float64))

        parent_clamped_selection = torch.tensor((0, 1, -2, -1))
        child1_clamped_selection = torch.tensor((2,))
        child2_clamped_selection = torch.tensor((2,))

        parent_mass_scale     = 1.
        parent_moment_scale   = 10.
        moment_ratio          = 0.1
        children_moment_scale = (0.5, 0.5, 0.5)
        children_mass_scale   = (1, 1, 1)

        b_DLO_mass, parent_MOI, children_MOI, _, _, _ = DEFT_initialization_BDLO6(
            b_undeformed_vert_bdlo6,
            parent_mass_scale, parent_moment_scale,
            children_moment_scale, children_mass_scale,
            moment_ratio,
        )

        # ---- Load datasets ----
        eval_time_horizon_local = total_time - 2
        print("Loading BDLO6 eval dataset...")
        eval_ds = Eval_DEFTData_BDLO6(
            total_time=total_time,
            eval_time_horizon=eval_time_horizon_local,
            device=device,
        )
        eval_batch_local = len(eval_ds)
        print(f"  using {eval_batch_local} eval samples")

        print("Loading BDLO6 train dataset...")
        train_ds = Train_DEFTData_BDLO6(
            total_time=total_time,
            training_time_horizon=train_time_horizon,
            device=device,
        )
        print(f"  loaded {len(train_ds)} train windows")
        train_loader = DataLoader(train_ds, batch_size=train_batch, shuffle=True, drop_last=True)

        # ---- Build sims ----
        index_selection1, index_selection2, parent_MOI_index1, parent_MOI_index2 = index_init(
            rigid_body_coupling_index, n_branch_local)

        clamped_index_train, parent_theta_clamp_local, child1_theta_clamp_local, child2_theta_clamp_local = clamp_index(
            train_batch,
            parent_clamped_selection, child1_clamped_selection, child2_clamped_selection,
            n_branch_local, n_parent_vertices,
            True, False, False,
        )
        clamped_index_eval, _, _, _ = clamp_index(
            eval_batch_local,
            parent_clamped_selection, child1_clamped_selection, child2_clamped_selection,
            n_branch_local, n_parent_vertices,
            True, False, False,
        )

        b_undeformed_vert_for_sim = b_undeformed_vert_bdlo6.view(n_branch_local, -1, 3)

        def _build_sim(batch_size, clamped_idx, _bend_p, _bend_c1, _bend_c2, _bend_c3, _twist, _damp, _lw):
            return DEFT_sim(
                batch=batch_size,
                n_branch=n_branch_local,
                n_vert=n_vert,
                cs_n_vert=cs_n_vert,
                b_init_n_vert=b_undeformed_vert_for_sim,
                n_edge=n_edge,
                b_undeformed_vert=b_undeformed_vert_for_sim,
                b_DLO_mass=b_DLO_mass,
                parent_DLO_MOI=parent_MOI,
                children_DLO_MOI=children_MOI,
                device=device,
                clamped_index=clamped_idx,
                rigid_body_coupling_index=rigid_body_coupling_index,
                parent_MOI_index1=parent_MOI_index1,
                parent_MOI_index2=parent_MOI_index2,
                parent_clamped_selection=parent_clamped_selection,
                child1_clamped_selection=child1_clamped_selection,
                child2_clamped_selection=child2_clamped_selection,
                clamp_parent=True, clamp_child1=False, clamp_child2=False,
                index_selection1=index_selection1,
                index_selection2=index_selection2,
                bend_stiffness_parent=_bend_p,
                bend_stiffness_child1=_bend_c1,
                bend_stiffness_child2=_bend_c2,
                bend_stiffness_child3=_bend_c3,
                twist_stiffness=_twist,
                damping=_damp,
                learning_weight=_lw,
                use_orientation_constraints=True,
                use_attachment_constraints=True,
                bdlo5=False,
                bdlo6=True,
            )

        sim_train = _build_sim(train_batch, clamped_index_train,
                               bend_stiffness_parent, bend_stiffness_child1,
                               bend_stiffness_child2, bend_stiffness_child3,
                               twist_stiffness, damping, learning_weight)
        sim_eval  = _build_sim(eval_batch_local, clamped_index_eval,
                               eval_bend_stiffness_parent, eval_bend_stiffness_child1,
                               eval_bend_stiffness_child2, eval_bend_stiffness_child3,
                               eval_twist_stiffness, eval_damping, eval_learning_weight)

        # ---- Load pretrained checkpoint if requested ----
        if load_model:
            pretrained_path = "../save_model/BDLO6/DEFT_ends_6_pretrained_wo_residual.pth"
            if os.path.exists(pretrained_path):
                pretrained_dict = torch.load(pretrained_path)
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'adjacency_batch' not in k}
                sim_train.load_state_dict(pretrained_dict, strict=False)
                print(f"Loaded BDLO6 pretrained model from {pretrained_path}")
            else:
                print(f"WARNING: pretrained checkpoint not found: {pretrained_path}")

        # ---- Optimizer ----
        lr_scale = 10
        physics_params = [
            {"params": sim_train.p_DLO_diagonal,                   "lr": 1e-6 * lr_scale},
            {"params": sim_train.c_DLO_diagonal,                   "lr": 1e-6 * lr_scale},
            {"params": sim_train.integration_ratio,                "lr": 1e-6 * lr_scale},
            {"params": sim_train.velocity_ratio,                   "lr": 1e-6 * lr_scale},
            {"params": sim_train.undeformed_vert,                  "lr": 1e-6 * lr_scale},
            {"params": sim_train.mass_diagonal,                    "lr": 1e-6 * lr_scale},
            {"params": sim_train.damping,                          "lr": 1e-6 * lr_scale},
            {"params": sim_train.gravity,                          "lr": 1e-6 * lr_scale},
            {"params": sim_train.DEFT_func.twist_stiffness,        "lr": 1e-6 * lr_scale},
            {"params": sim_train.DEFT_func.bend_stiffness_parent,  "lr": 1e-8 * lr_scale},
            {"params": sim_train.DEFT_func.bend_stiffness_child1,  "lr": 1e-8 * lr_scale},
            {"params": sim_train.DEFT_func.bend_stiffness_child2,  "lr": 1e-8 * lr_scale},
            {"params": sim_train.DEFT_func.bend_stiffness_child3,  "lr": 1e-8 * lr_scale},
        ]
        residual_params = [
            {"params": sim_train.GNN_tree.parameters(), "lr": 1e-5 * lr_scale},
            {"params": [sim_train.learning_weight],     "lr": 1e-4 * lr_scale},
        ]

        if training_mode == "physics":
            # Freeze GNN, train physics only
            for p in sim_train.GNN_tree.parameters():
                p.requires_grad = False
            sim_train.learning_weight.requires_grad = False
            optimizer = optim.Adam(physics_params, eps=1e-8)
            print("BDLO6 training mode: physics only (GNN frozen)")
        elif training_mode == "residual":
            # Freeze physics, train GNN only
            sim_train.p_DLO_diagonal.requires_grad = False
            sim_train.c_DLO_diagonal.requires_grad = False
            sim_train.integration_ratio.requires_grad = False
            sim_train.velocity_ratio.requires_grad = False
            sim_train.undeformed_vert.requires_grad = False
            sim_train.mass_diagonal.requires_grad = False
            sim_train.damping.requires_grad = False
            sim_train.gravity.requires_grad = False
            sim_train.DEFT_func.twist_stiffness.requires_grad = False
            sim_train.DEFT_func.bend_stiffness_parent.requires_grad = False
            sim_train.DEFT_func.bend_stiffness_child1.requires_grad = False
            sim_train.DEFT_func.bend_stiffness_child2.requires_grad = False
            sim_train.DEFT_func.bend_stiffness_child3.requires_grad = False
            optimizer = optim.Adam(residual_params, eps=1e-8)
            print("BDLO6 training mode: residual only (physics frozen)")

        loss_func = torch.nn.MSELoss()
        dt = 0.01
        eval_loader = DataLoader(eval_ds, batch_size=eval_batch_local, shuffle=False, drop_last=True)

        # ---- Bookkeeping for losses / checkpoints ----
        save_steps = 0
        evaluate_period = 20
        training_iteration = 0
        training_case = 1
        train_epoch_count = 100
        train_losses_log = []
        eval_losses_log  = []
        eval_iters_log   = []

        os.makedirs("../save_model", exist_ok=True)
        os.makedirs("../training_record", exist_ok=True)

        def run_eval(vis_this_eval):
            """Sync sim_eval state from sim_train, run full eval pass, return mean RMSE."""
            state = {k: v for k, v in sim_train.state_dict().items() if 'adjacency_batch' not in k}
            sim_eval.load_state_dict(state, strict=False)
            with torch.no_grad():
                rmses = []
                for prev, curr, target in eval_loader:
                    traj_loss_eval, _ = sim_eval.iterative_sim(
                        eval_time_horizon_local,
                        curr, prev, target,
                        loss_func, dt,
                        parent_theta_clamp_local, child1_theta_clamp_local, child2_theta_clamp_local,
                        False,                          # inference_1_batch
                        vis_type="DEFT_6",
                        vis=vis_this_eval,
                    )
                    rmses.append(float(np.sqrt(traj_loss_eval.cpu().numpy() / total_time)))
            return float(np.mean(rmses))

        # ---- Main train loop ----
        for epoch in range(train_epoch_count):
            bar = tqdm(train_loader, desc=f"BDLO6 epoch {epoch}")
            for data in bar:
                # Periodic eval (and at step 0, before any training)
                if save_steps % evaluate_period == 0:
                    ckpt_path = "../save_model/DEFT_%s_6_%s_%s.pth" % (
                        clamp_type, str(training_iteration), training_case)
                    torch.save(sim_train.state_dict(), ckpt_path)
                    vis_this_eval = inference_vis if training_iteration == 0 else False
                    eval_rmse = run_eval(vis_this_eval=vis_this_eval)
                    print(f"  [step {training_iteration}] eval RMSE: {eval_rmse:.6f}  (ckpt: {ckpt_path})")
                    eval_losses_log.append(eval_rmse)
                    eval_iters_log.append(training_iteration)
                    save_pickle(eval_losses_log,
                                "../training_record/eval_%s_loss_DEFT_%s_6.pkl" % (clamp_type, training_case))
                    save_pickle(eval_iters_log,
                                "../training_record/eval_%s_epoches_DEFT_%s_6.pkl" % (clamp_type, training_case))

                save_steps += 1
                training_iteration += 1

                prev_b, curr_b, target_b, _mu0 = data
                traj_loss, total_loss = sim_train.iterative_sim(
                    train_time_horizon,
                    curr_b, prev_b, target_b,
                    loss_func, dt,
                    parent_theta_clamp_local, child1_theta_clamp_local, child2_theta_clamp_local,
                    inference_1_batch,
                    vis_type="DEFT_6",
                    vis=False,
                )

                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    print(f"NaN/Inf at iter {training_iteration}, skipping batch")
                    optimizer.zero_grad()
                    continue

                train_losses_log.append(float(traj_loss.cpu().detach().numpy()) / train_time_horizon)
                bar.set_postfix(loss=train_losses_log[-1])

                total_loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(sim_train.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

                save_pickle(train_losses_log,
                            "../training_record/train_%s_loss_DEFT_%s_6.pkl" % (clamp_type, training_case))

        print("BDLO6 training done.")
        return

    # Decide how many batches we use in evaluation (1 batch if inference_1_batch is True, else the entire eval_set_number)
    if inference_1_batch:
        eval_batch = 1
    else:
        eval_batch = eval_set_number

    # Number of vertices in the child branches
    n_children_vertices = (n_child1_vertices, n_child2_vertices)

    # Extract parent and child vertices from the undeformed BDLO
    parent_vertices_undeform = undeformed_BDLO[:, :n_parent_vertices]
    child1_vertices_undeform = undeformed_BDLO[:, n_parent_vertices: n_parent_vertices + n_children_vertices[0] - 1]
    child2_vertices_undeform = undeformed_BDLO[:, n_parent_vertices + n_children_vertices[0] - 1:]

    # DEFT_initialization returns scaled mass, MOI, rod orientations, nominal length, etc.
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

    # Construct the branched BDLO from data for the training batch
    # b_DLOs_vertices_undeform_untransform is the original set of undeformed states across the batch
    # The second return is an optional placeholder for transformations or expansions
    b_DLOs_vertices_undeform_untransform, _ = construct_b_DLOs(
        train_batch,
        rigid_body_coupling_index,
        n_parent_vertices,
        cs_n_vert,
        n_branch,
        parent_vertices_undeform,
        parent_vertices_undeform,
        child1_vertices_undeform,
        child1_vertices_undeform,
        child2_vertices_undeform,
        child2_vertices_undeform,
        bdlo5=bdlo5
    )

    # Transform the axis from local coordinate to global by re-indexing the coordinate axes
    if BDLO_type == 5:
        # BDLO5: apply (-z,-x,y), then rotate -X 90° (x,z,-y), then negate x
        step1 = torch.zeros_like(b_DLOs_vertices_undeform_untransform)
        step1[:, :, :, 0] = -b_DLOs_vertices_undeform_untransform[:, :, :, 2]
        step1[:, :, :, 1] = -b_DLOs_vertices_undeform_untransform[:, :, :, 0]
        step1[:, :, :, 2] = b_DLOs_vertices_undeform_untransform[:, :, :, 1]
        b_DLOs_vertices_undeform_transform = torch.zeros_like(step1)
        b_DLOs_vertices_undeform_transform[:, :, :, 0] = -step1[:, :, :, 0]
        b_DLOs_vertices_undeform_transform[:, :, :, 1] = step1[:, :, :, 2]
        b_DLOs_vertices_undeform_transform[:, :, :, 2] = -step1[:, :, :, 1]
    else:
        b_DLOs_vertices_undeform_transform = torch.zeros_like(b_DLOs_vertices_undeform_untransform)
        b_DLOs_vertices_undeform_transform[:, :, :, 0] = -b_DLOs_vertices_undeform_untransform[:, :, :, 2]
        b_DLOs_vertices_undeform_transform[:, :, :, 1] = -b_DLOs_vertices_undeform_untransform[:, :, :, 0]
        b_DLOs_vertices_undeform_transform[:, :, :, 2] = b_DLOs_vertices_undeform_untransform[:, :, :, 1]

    # The first sample in the batch of undeformed vertices (reshape to [n_branch, n_vert, 3])
    b_undeformed_vert = b_DLOs_vertices_undeform_transform[0].view(n_branch, -1, 3)

    # Initialize index selection for parent MOI indices, etc.
    index_selection1, index_selection2, parent_MOI_index1, parent_MOI_index2 = index_init(
        rigid_body_coupling_index,
        n_branch
    )

    # Decide which vertices get clamped (i.e., fixed in space/rotation) for the training and evaluation sets
    clamped_index, parent_theta_clamp, child1_theta_clamp, child2_theta_clamp = clamp_index(
        train_batch,
        parent_clamped_selection,
        child1_clamped_selection,
        child2_clamped_selection,
        n_branch,
        n_parent_vertices,
        clamp_parent,
        clamp_child1,
        clamp_child2
    )
    eval_clamped_index, eval_parent_theta_clamp, eval_child1_theta_clamp, eval_child2_theta_clamp = clamp_index(
        eval_batch,
        parent_clamped_selection,
        child1_clamped_selection,
        child2_clamped_selection,
        n_branch,
        n_parent_vertices,
        clamp_parent,
        clamp_child1,
        clamp_child2
    )

    # Timestep for simulation
    dt = 0.01

    # Instantiate DEFT_sim objects for training and evaluation
    DEFT_sim_train = DEFT_sim(
        batch=train_batch,
        n_branch=n_branch,
        n_vert=n_vert,
        cs_n_vert=cs_n_vert,
        b_init_n_vert=b_undeformed_vert,
        n_edge=n_vert - 1,
        b_undeformed_vert=b_undeformed_vert,
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
        clamp_parent=clamp_parent,
        clamp_child1=clamp_child1,
        clamp_child2=clamp_child2,
        index_selection1=index_selection1,
        index_selection2=index_selection2,
        bend_stiffness_parent=bend_stiffness_parent,
        bend_stiffness_child1=bend_stiffness_child1,
        bend_stiffness_child2=bend_stiffness_child2,
        twist_stiffness=twist_stiffness,
        damping=damping,
        learning_weight=learning_weight,
        use_orientation_constraints=use_orientation_constraints,
        use_attachment_constraints=use_attachment_constraints,
        bdlo5=bdlo5
    )
    DEFT_sim_eval = DEFT_sim(
        batch=eval_batch,
        n_branch=n_branch,
        n_vert=n_vert,
        cs_n_vert=cs_n_vert,
        b_init_n_vert=b_undeformed_vert,
        n_edge=n_vert - 1,
        b_undeformed_vert=b_undeformed_vert,
        b_DLO_mass=b_DLO_mass,
        parent_DLO_MOI=parent_MOI,
        children_DLO_MOI=children_MOI,
        device=device,
        clamped_index=eval_clamped_index,
        rigid_body_coupling_index=rigid_body_coupling_index,
        parent_MOI_index1=parent_MOI_index1,
        parent_MOI_index2=parent_MOI_index2,
        parent_clamped_selection=parent_clamped_selection,
        child1_clamped_selection=child1_clamped_selection,
        child2_clamped_selection=child2_clamped_selection,
        clamp_parent=clamp_parent,
        clamp_child1=clamp_child1,
        clamp_child2=clamp_child2,
        index_selection1=index_selection1,
        index_selection2=index_selection2,
        bend_stiffness_parent=bend_stiffness_parent,
        bend_stiffness_child1=bend_stiffness_child1,
        bend_stiffness_child2=bend_stiffness_child2,
        twist_stiffness=twist_stiffness,
        damping=damping,
        learning_weight=learning_weight,
        use_orientation_constraints=use_orientation_constraints,
        use_attachment_constraints=use_attachment_constraints,
        bdlo5=bdlo5
    )

    # Load pretrained models for initialization depending on BDLO_type and clamp_type
    # Always loads full model (physics + GNN) when available
    if load_model:
        pretrained_path = None
        if BDLO_type == 1 and clamp_type == "ends":
            pretrained_path = "../save_model/BDLO1/DEFT_ends_1_pretrained_full_model.pth"
        if BDLO_type == 1 and clamp_type == "middle":
            pretrained_path = "../save_model/BDLO1/DEFT_middle_1_pretrained_full_model.pth"
        if BDLO_type == 2:
            pretrained_path = "../save_model/BDLO2/DEFT_ends_2_pretrained_full_model.pth"
        if BDLO_type == 3 and clamp_type == "ends":
            pretrained_path = "../save_model/BDLO3/DEFT_ends_3_pretrained_full_model.pth"
        if BDLO_type == 3 and clamp_type == "middle":
            pretrained_path = "../save_model/BDLO3/DEFT_middle_3_pretrained_full_model.pth"
        if BDLO_type == 4:
            pretrained_path = "../save_model/BDLO4/DEFT_ends_4_pretrained_full_model.pth"
        if BDLO_type == 5:
            pretrained_path = "../save_model/BDLO5/DEFT_ends_5_pretrained_full_model.pth"

        if pretrained_path is not None:
            pretrained_dict = torch.load(pretrained_path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'adjacency_batch' not in k}
            DEFT_sim_train.load_state_dict(pretrained_dict, strict=False)
            print(f"Loaded full model from {pretrained_path}")
    
    # If we want to visualize the undeformed states
    if undeform_vis:
        # Visualize the first batch's undeformed state
        first_batch_vertices = b_DLOs_vertices_undeform_transform[0]  # shape: [3, n_vert, 3]
        colors = ['red', 'green', 'blue']

        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')

        # For each branch, plot all vertex positions (here, just one set since it's "undeformed_vis")
        for branch_idx in range(3):
            branch_positions = first_batch_vertices[branch_idx, :, :]

            for vertex_idx in range(branch_positions.shape[0]):
                vertex_positions = branch_positions[vertex_idx, :]
                x = vertex_positions[0].unsqueeze(dim=0).numpy()
                y = vertex_positions[1].unsqueeze(dim=0).numpy()
                z = vertex_positions[2].unsqueeze(dim=0).numpy()
                ax.scatter(x, y, z, color=colors[branch_idx], alpha=1.)

        ax.set_xlim(-0.5, 1.0)
        ax.set_ylim(-0.5, 1.0)
        ax.set_zlim(-0.25, 1.25)
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')
        ax.set_title('Trajectories of Vertices Over Time (First Batch)')

        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], color=colors[i], lw=2, label=f'Branch {i}') for i in range(3)]
        ax.legend(handles=legend_elements)
        plt.show()

    # Scale factor for learning rate (reduced to prevent NaN)
    lr_scale = 10

    # Define loss function
    loss_func = torch.nn.MSELoss()

    # Define physics and GNN parameter groups
    physics_params = [
        {"params": DEFT_sim_train.p_DLO_diagonal, "lr": 1e-6 * lr_scale},
        {"params": DEFT_sim_train.c_DLO_diagonal, "lr": 1e-6 * lr_scale},
        {"params": DEFT_sim_train.integration_ratio, "lr": 1e-6 * lr_scale},
        {"params": DEFT_sim_train.velocity_ratio, "lr": 1e-6 * lr_scale},
        {"params": DEFT_sim_train.undeformed_vert, "lr": 1e-6 * lr_scale},
        {"params": DEFT_sim_train.mass_diagonal, "lr": 1e-6 * lr_scale},
        {"params": DEFT_sim_train.damping, "lr": 1e-6 * lr_scale},
        {"params": DEFT_sim_train.gravity, "lr": 1e-6 * lr_scale},
        {"params": DEFT_sim_train.DEFT_func.twist_stiffness, "lr": 1e-6 * lr_scale},
        {"params": DEFT_sim_train.DEFT_func.bend_stiffness_parent, "lr": 1e-8 * lr_scale},
        {"params": DEFT_sim_train.DEFT_func.bend_stiffness_child1, "lr": 1e-8 * lr_scale},
        {"params": DEFT_sim_train.DEFT_func.bend_stiffness_child2, "lr": 1e-8 * lr_scale},
    ]
    residual_params = [
        {"params": DEFT_sim_train.learning_weight, "lr": 1e-4 * lr_scale},
        {"params": DEFT_sim_train.GNN_tree.parameters(), "lr": 1e-5 * lr_scale},
    ]

    # Select which parameters to optimize based on training_mode
    # "physics": train only material/physics parameters, freeze GNN
    # "residual": train only GNN + learning_weight, freeze physics
    # "full": train both physics and residual together
    if training_mode == "physics":
        # Freeze GNN parameters
        for param in DEFT_sim_train.GNN_tree.parameters():
            param.requires_grad = False
        DEFT_sim_train.learning_weight.requires_grad = False
        parameters_to_update = physics_params
        print("Training mode: physics only (GNN frozen)")
    elif training_mode == "residual":
        # Freeze physics parameters
        DEFT_sim_train.p_DLO_diagonal.requires_grad = False
        DEFT_sim_train.c_DLO_diagonal.requires_grad = False
        DEFT_sim_train.integration_ratio.requires_grad = False
        DEFT_sim_train.velocity_ratio.requires_grad = False
        DEFT_sim_train.undeformed_vert.requires_grad = False
        DEFT_sim_train.mass_diagonal.requires_grad = False
        DEFT_sim_train.damping.requires_grad = False
        DEFT_sim_train.gravity.requires_grad = False
        DEFT_sim_train.DEFT_func.twist_stiffness.requires_grad = False
        DEFT_sim_train.DEFT_func.bend_stiffness_parent.requires_grad = False
        DEFT_sim_train.DEFT_func.bend_stiffness_child1.requires_grad = False
        DEFT_sim_train.DEFT_func.bend_stiffness_child2.requires_grad = False
        parameters_to_update = residual_params
        print("Training mode: residual only (physics frozen)")
    elif training_mode == "full":
        parameters_to_update = physics_params + residual_params
        print("Training mode: full (physics + residual)")

    # Check for NaN in initial parameters
    for name, param in DEFT_sim_train.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            print(f"Warning: NaN/Inf found in parameter {name}, reinitializing...")
            param.data = torch.randn_like(param.data) * 0.01
    
    # Define the optimizer (here, Adam for better stability) with the chosen parameters
    optimizer = optim.Adam(parameters_to_update, eps=1e-8)

    # We'll store evaluation results after certain intervals
    eval_epochs = []
    eval_losses = []

    # We'll store training results as well
    training_epochs = []
    training_losses = []

    # Loading training / evaluation data from custom datasets
    if clamp_type == "ends":
        eval_dataset = Eval_DEFTData(
            BDLO_type,
            n_parent_vertices,
            n_children_vertices,
            n_branch,
            rigid_body_coupling_index,
            eval_set_number,
            total_time,
            eval_time_horizon,
            device,
            bdlo5=bdlo5
        )
        eval_data_len = len(eval_dataset)
        train_dataset = Train_DEFTData(
            BDLO_type,
            n_parent_vertices,
            n_children_vertices,
            n_branch,
            rigid_body_coupling_index,
            train_set_number,
            total_time,
            train_time_horizon,
            device,
            bdlo5=bdlo5
        )
        train_data_loader = DataLoader(train_dataset, batch_size=train_batch, shuffle=True, drop_last=True)

    if clamp_type == "middle":
        # We load data from the dataset variant with middle clamps
        eval_dataset = Eval_DEFTData(
            str(BDLO_type) + "_mid_clamp",
            n_parent_vertices,
            n_children_vertices,
            n_branch,
            rigid_body_coupling_index,
            eval_set_number,
            total_time,
            eval_time_horizon,
            device,
            bdlo5=bdlo5
        )
        eval_data_len = len(eval_dataset)
        train_dataset = Train_DEFTData(
            str(BDLO_type) + "_mid_clamp",
            n_parent_vertices,
            n_children_vertices,
            n_branch,
            rigid_body_coupling_index,
            train_set_number,
            total_time,
            train_time_horizon,
            device,
            bdlo5=bdlo5
        )
        train_data_loader = DataLoader(train_dataset, batch_size=train_batch, shuffle=True, drop_last=True)

    # Define number of epochs to train
    train_epoch = 100
    save_steps = 0
    evaluate_period = 20
    model = "DEFT"
    training_case = 1

    # The main training loop
    if model == "DEFT":
        training_iteration = 0
        vis_type = "DEFT_%s" % BDLO_type
        for epoch in range(train_epoch):
            bar = tqdm(train_data_loader)
            for data in bar:
                # Evaluate the model on the eval set periodically
                if save_steps % evaluate_period == 0:
                    part_eval = eval_set_number
                    # Random split for partial evaluation if desired
                    eval_set, test_set = torch.utils.data.random_split(eval_dataset,
                                                                       [part_eval, eval_data_len - part_eval])
                    eval_data_loader = DataLoader(eval_set, batch_size=eval_batch, shuffle=True, drop_last=True)

                    # Save the current model
                    torch.save(
                        DEFT_sim_train.state_dict(),
                        os.path.join("../save_model/", "DEFT_%s_%s_%s_%s.pth" % (
                        clamp_type, BDLO_type, str(training_iteration), training_case))
                    )
                    # Load the saved model into the evaluation simulation object
                    saved = torch.load("../save_model/DEFT_%s_%s_%s_%s.pth" % (
                        clamp_type, BDLO_type, str(training_iteration), training_case))
                    saved = {k: v for k, v in saved.items() if 'adjacency_batch' not in k}
                    DEFT_sim_eval.load_state_dict(saved, strict=False)

                    eval_bar = tqdm(eval_data_loader)
                    with torch.no_grad():
                        for eval_data in eval_bar:
                            # The evaluation data has previous, current, and target states
                            previous_b_DLOs_vertices_traj, b_DLOs_vertices_traj, target_b_DLOs_vertices_traj = eval_data
                            vis_type = "DEFT_%s" % BDLO_type
                            # Visualize on first iteration if inference_vis is True
                            if training_iteration == 0:
                                vis = inference_vis
                            else:
                                vis = False
                            # Perform iterative simulation over eval_time_horizon
                            traj_loss_eval, _ = DEFT_sim_eval.iterative_sim(
                                eval_time_horizon,
                                b_DLOs_vertices_traj,
                                previous_b_DLOs_vertices_traj,
                                target_b_DLOs_vertices_traj,
                                loss_func,
                                dt,
                                parent_theta_clamp,
                                child1_theta_clamp,
                                child2_theta_clamp,
                                inference_1_batch,
                                vis_type=vis_type,
                                vis=vis
                            )
                            # Print and record the average loss
                            avg_loss = np.sqrt(traj_loss_eval.cpu().detach().numpy() / total_time)
                            print(f"Eval RMSE: {avg_loss:.6f}")
                            eval_losses.append(traj_loss_eval.cpu().detach().numpy() / total_time)
                            eval_epochs.append(training_iteration)

                            # Save the evaluation losses to pickle
                            save_pickle(eval_losses, "../training_record/eval_%s_loss_DEFT_%s_%s.pkl" % (
                            clamp_type, training_case, BDLO_type))
                            save_pickle(eval_epochs, "../training_record/eval_%s_epoches_DEFT_%s_%s.pkl" % (
                            clamp_type, training_case, BDLO_type))

                # Increment steps and iteration
                save_steps += 1
                training_iteration += 1

                # Get the input data from the loader
                vis = False
                previous_b_DLOs_vertices_traj, b_DLOs_vertices_traj, target_b_DLOs_vertices_traj, m_u0_traj = data

                # Forward pass through the DEFT model for train_time_horizon timesteps
                traj_loss, total_loss = DEFT_sim_train.iterative_sim(
                    train_time_horizon,
                    b_DLOs_vertices_traj,
                    previous_b_DLOs_vertices_traj,
                    target_b_DLOs_vertices_traj,
                    loss_func,
                    dt,
                    parent_theta_clamp,
                    child1_theta_clamp,
                    child2_theta_clamp,
                    inference_1_batch,
                    vis_type=vis_type,
                    vis=vis
                )

                # Check for NaN in loss
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    print(f"NaN/Inf detected at iteration {training_iteration}, skipping batch")
                    optimizer.zero_grad()
                    continue

                # Record and print training loss
                training_losses.append(traj_loss.cpu().detach().numpy() / train_time_horizon)
                training_epochs.append(training_iteration)

                # Backprop through the total loss
                total_loss.backward(retain_graph=True)
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(DEFT_sim_train.parameters(), max_norm=1.0)
                
                optimizer.step()
                optimizer.zero_grad()

                # Save training losses to pickle
                save_pickle(training_losses,
                            "../training_record/train_%s_loss_DEFT_%s_%s.pkl" % (clamp_type, training_case, BDLO_type))
                save_pickle(training_epochs,
                            "../training_record/train_%s_step_DEFT_%s_%s.pkl" % (clamp_type, training_case, BDLO_type))


if __name__ == "__main__":
    # Setting up a command-line interface for hyperparameters and options

    # Make sure to use double precision for stability in the DEFT simulations
    torch.set_default_dtype(torch.float64)
    # Ensure all tensor operations use float64
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(1)
    np.random.seed(1)

    # Limit the number of threads used by PyTorch and underlying libraries for reproducibility
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    # Initialize argument parser
    parser = argparse.ArgumentParser()

    # BDLO_type controls which BDLO dataset (and initial parameter sets) to use
    parser.add_argument("--BDLO_type", type=int, default=1)

    # clamp_type indicates how the BDLO is clamped (ends or middle)
    parser.add_argument("--clamp_type", type=str, default="ends")

    # total_time is the maximum number of timesteps we have in the dataset (e.g. 500)
    parser.add_argument("--total_time", type=int, default=500)

    # train_time_horizon is how many timesteps we simulate in each training iteration
    parser.add_argument("--train_time_horizon", type=int, default=50)

    # Whether to visualize the initial undeformed vertices
    parser.add_argument("--undeform_vis", type=bool, default=False)

    # Whether we do inference only for 1 batch (for speed) or for all eval sets
    parser.add_argument("--inference_1_batch", type=lambda x: x.lower() == 'true', default=False)

    # Whether to enable residual learning: if True, GNN-based updates are used
    parser.add_argument("--residual_learning", type=bool, default=False)

    # Training batch size
    parser.add_argument("--train_batch", type=int, default=32)

    # Whether to visualize inference results (for debugging)
    parser.add_argument("--inference_vis", type=bool, default=False)

    # load trained model
    parser.add_argument("--load_model", type=bool, default=False)

    # training_mode: "physics" (material params only), "residual" (GNN only, physics frozen), "full" (both)
    parser.add_argument("--training_mode", type=str, default="physics", choices=["physics", "residual", "full"])

    # Constraint flags for ablation baselines
    parser.add_argument("--use_orientation_constraints", type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument("--use_attachment_constraints", type=lambda x: x.lower() == 'true', default=True)

    # Flags for which branches are clamped
    clamp_parent = True
    clamp_child1 = False
    clamp_child2 = False

    # Number of branches for the BDLO (1 parent branch + 2 children branches)
    n_branch = 3

    # For simplicity, everything is done on CPU in this version
    device = "cpu"

    args = parser.parse_args()

    # Call the training function with the user-specified arguments
    train(
        train_batch=args.train_batch,
        BDLO_type=args.BDLO_type,
        total_time=args.total_time,
        train_time_horizon=args.train_time_horizon,
        undeform_vis=args.undeform_vis,
        inference_vis=args.inference_vis,
        inference_1_batch=args.inference_1_batch,
        residual_learning=args.residual_learning,
        clamp_type=args.clamp_type,
        load_model=args.load_model,
        training_mode=args.training_mode,
        use_orientation_constraints=args.use_orientation_constraints,
        use_attachment_constraints=args.use_attachment_constraints
    )
