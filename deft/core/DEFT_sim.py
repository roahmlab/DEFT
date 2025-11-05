import torch
import torch.nn as nn

# Import DEFT functions
from .DEFT_func import DEFT_func

# Importing multiple utilities:
# - rotation_matrix: Helper for creating rotation matrices from angles
# - computeW, computeLengths, computeEdges: Utility functions for BDLO geometry
# - visualize_tensors_3d_in_same_plot_no_zeros: For debugging/visualizing results
from ..utils.util import rotation_matrix, computeW, computeLengths, computeEdges, visualize_tensors_3d_in_same_plot_no_zeros

# Import constraints solver(s)
from ..solvers.constraints_solver import constraints_enforcement
import pytorch3d.transforms.rotation_conversions

# A numba-accelerated version of constraints enforcement
from ..solvers.constraints_enforcement_numba import constraints_enforcement_numba
constraints_numba = constraints_enforcement_numba()

# Importing a graph neural network for residual learning
import numpy as np
from ..models.GNN_tree import BatchedGNNModel

import time

class DEFT_sim(nn.Module):
    """
    This class contains the DEFT simulation logic for BDLOs
    with the possibility of residual learning using a GNN.
    """
    def __init__(
        self,
        batch,
        n_branch,
        n_vert,
        cs_n_vert,
        b_init_n_vert,
        n_edge,
        b_undeformed_vert,
        b_DLO_mass,
        parent_DLO_MOI,
        children_DLO_MOI,
        device,
        clamped_index,
        rigid_body_coupling_index,
        parent_MOI_index1,
        parent_MOI_index2,
        parent_clamped_selection,
        child1_clamped_selection,
        child2_clamped_selection,
        clamp_parent,
        clamp_child1,
        clamp_child2,
        index_selection1,
        index_selection2,
        bend_stiffness_parent,
        bend_stiffness_child1,
        bend_stiffness_child2,
        twist_stiffness,
        damping,
        learning_weight
    ):
        super().__init__()
        """
        Parameters:
        ----------
        batch: int
            Number of BDLO instances or samples in a single training batch.
        n_branch: int
            Number of branches for the BDLO (1 parent branch + 2 children branches).
        n_vert: int
            Number of vertices in the parent branch.
        cs_n_vert: tuple
            Number of vertices for each child branch.
        b_init_n_vert: tensor
            The undeformed positions of all vertices for the BDLO, used as initialization.
        n_edge: int
            Number of edges per branch (n_vert - 1).
        b_undeformed_vert: tensor
            Reference undeformed shape for the BDLO (parent + child).
        b_DLO_mass: tensor
            Mass distribution (diagonal form) for each branch's vertices.
        parent_DLO_MOI: tensor
            Moment of inertia for the parent branch (diagonal).
        children_DLO_MOI: tensor
            Moment of inertia for the child branches (diagonal).
        device: str
            Where to run the simulation ('cpu' only for now).
        clamped_index: tensor
            Boolean mask indicating which vertices are clamped (for the inextensibility constraint).
        rigid_body_coupling_index: list
            Indices specifying which vertices link child branches to the parent branch.
        parent_MOI_index1 / parent_MOI_index2: lists
            Indices used for coupling the parent's MOI with child branches.
        parent_clamped_selection / child1_clamped_selection / child2_clamped_selection: tensor
            Indices of vertices that are clamped in the parent or children branches.
        clamp_parent / clamp_child1 / clamp_child2: bool
            Whether the parent or child branches are physically clamped in the experiment.
        index_selection1 / index_selection2: lists
            Index sets used for referencing sub-portions of the BDLO (internal usage).
        bend_stiffness_parent / bend_stiffness_child1 / bend_stiffness_child2 / twist_stiffness: nn.Parameters
            Deformational stiffness parameters for bending and twisting in each branch.
        damping: nn.Parameter
            Damping coefficients for dynamic updates.
        learning_weight: nn.Parameter
            Scaling factor for the residual correction from the GNN.

        The class holds all states needed to run DEFT simulation + potential GNN-based residual corrections.
        """

        # Store constructor inputs
        self.clamped_index = clamped_index
        self.n_vert = n_vert
        self.n_edge = n_edge
        self.device = device
        self.n_branch = n_branch
        self.batch = batch
        self.clamp_parent = clamp_parent
        self.clamp_child1 = clamp_child1
        self.clamp_child2 = clamp_child2

        # For parallelization across a batch and multiple branches:
        # We'll figure out child vs parent branches (indices) in a vectorized way
        selected_children_index = [i for i in range(1, batch * n_branch) if i % n_branch != 0]
        self.rigid_body_coupling_index = rigid_body_coupling_index

        # "index_selection_1" / "index_selection_2": used for referencing the parent's edges
        index_selection_1 = torch.tensor(rigid_body_coupling_index) - 1
        index_selection_2 = torch.tensor(rigid_body_coupling_index)
        # fused_rigid_body_coupling_index merges them in some pattern for convenience
        self.fused_rigid_body_coupling_index = torch.stack((index_selection_1, index_selection_2), dim=1).reshape(-1)

        # For identifying the branches in a batch:
        # - child1 is 1 mod n_branch
        # - child2 is 2 mod n_branch
        # - parent is 0 mod n_branch
        selected_child1_index = list(range(1, batch * n_branch, n_branch))
        selected_child2_index = list(range(2, batch * n_branch, n_branch))
        selected_parent_index = list(range(0, batch * n_branch, n_branch))
        self.selected_parent_index = torch.tensor(selected_parent_index)
        self.selected_child1_index = torch.tensor(selected_child1_index)
        self.selected_child2_index = torch.tensor(selected_child2_index)

        # Expand clamped vertex selection for the parent across the batch
        batch_indices = self.selected_parent_index.unsqueeze(1).expand(-1, parent_clamped_selection.size(0))
        parent_indices = parent_clamped_selection.unsqueeze(0).expand(self.selected_parent_index.size(0), -1)

        # Child1/Child2 clamped indices
        batch_child1_indices = self.selected_child1_index
        child1_indices = child1_clamped_selection
        batch_child2_indices = self.selected_child2_index
        child2_indices = child2_clamped_selection

        # Store index selections for reference
        self.index_selection1 = index_selection1
        self.index_selection2 = index_selection2

        # Flatten them for easier indexing
        self.batch_indices_flat = batch_indices.reshape(-1)
        self.parent_indices_flat = parent_indices.reshape(-1)

        self.batch_child1_indices_flat = batch_child1_indices.reshape(-1)
        self.child1_indices_flat = child1_indices.reshape(-1)

        self.batch_child2_indices_flat = batch_child2_indices.reshape(-1)
        self.child1_indices_flat = child2_indices.reshape(-1)  # reusing variable name but it's for child2

        # Store clamp selections
        self.parent_clamped_selection = parent_clamped_selection
        self.child1_clamped_selection = child1_clamped_selection
        self.child2_clamped_selection = child2_clamped_selection

        # Store inertia (MOI) for parent/child in parameter form
        self.p_DLO_diagonal = nn.Parameter(parent_DLO_MOI)
        self.c_DLO_diagonal = nn.Parameter(children_DLO_MOI)

        # Construct MOI matrices for children/parent rods
        self.children_MOI_matrix = torch.zeros(n_branch-1, 3, 3)
        self.children_MOI_matrix[:, 0, 0] = self.c_DLO_diagonal[:, 0]
        self.children_MOI_matrix[:, 1, 1] = self.c_DLO_diagonal[:, 1]
        self.children_MOI_matrix[:, 2, 2] = self.c_DLO_diagonal[:, 2]

        self.parent_MOI_matrix = torch.zeros((n_branch-1)*2, 3, 3)
        self.parent_MOI_matrix[:, 0, 0] = self.p_DLO_diagonal[:, 0]
        self.parent_MOI_matrix[:, 1, 1] = self.p_DLO_diagonal[:, 1]
        self.parent_MOI_matrix[:, 2, 2] = self.p_DLO_diagonal[:, 2]

        # Integration parameters (how we integrate the dynamic system)
        self.integration_ratio = nn.Parameter(torch.tensor(1., device=device))
        self.velocity_ratio = nn.Parameter(torch.tensor(0., device=device))

        # zero_mask: tracks vertices that may not exist (e.g., in shorter child branches)
        self.zero_mask = torch.all(b_undeformed_vert[:, 1:] == 0, dim=-1)

        # Compute reference lengths of edges and Voronoi region
        self.m_restEdgeL, self.m_restRegionL = computeLengths(
            computeEdges(b_undeformed_vert.clone(), self.zero_mask)
        )

        # Create a mask to handle situations where child branches end sooner
        m_restRegionL_mask = torch.ones_like(self.m_restRegionL)
        for i in range(len(cs_n_vert)):
            m_restRegionL_mask[i+1, cs_n_vert[i]-1:] = 0.

        # Instantiate the DEFT_func object, which does the heavy-lifting for bending, twisting, etc.
        self.DEFT_func = DEFT_func(
            batch,
            n_branch,
            n_vert,
            n_edge,
            m_restRegionL_mask,
            self.zero_mask,
            self.m_restEdgeL,
            bend_stiffness_parent,
            bend_stiffness_child1,
            bend_stiffness_child2,
            twist_stiffness,
            device=device
        )

        # Apply the masks so that unused edges are 0
        self.m_restRegionL = self.m_restRegionL * m_restRegionL_mask
        self.m_restEdgeL = self.m_restEdgeL * m_restRegionL_mask

        # Repeat across the batch dimension
        self.batched_m_restEdgeL = self.m_restEdgeL.repeat(self.batch, 1, 1).view(-1, n_edge)
        self.batched_m_restRegionL = self.m_restRegionL.repeat(self.batch, 1, 1).view(-1, n_edge)

        # Set the BDLO's initial undeformed positions as a trainable parameter
        self.undeformed_vert = nn.Parameter(b_init_n_vert)

        # Mass diagonal holds mass for each vertex in each branch
        self.mass_diagonal = nn.Parameter(b_DLO_mass)

        # Construct the mass matrix: shape [batch * n_branch, n_vert, 3, 3]
        # We replicate the diagonal mass for each vertex, then tile for the batch
        self.mass_matrix = (
            torch.eye(3)
            .unsqueeze(dim=0)
            .unsqueeze(dim=0)
            .repeat(n_branch, n_vert, 1, 1)
            * (self.mass_diagonal.unsqueeze(dim=-1).unsqueeze(dim=-1))
        ).unsqueeze(dim=0).repeat(batch, 1, 1, 1, 1).view(-1, n_vert, 3, 3)

        # Damping for each branch
        self.damping = damping

        # zero_mask_num is used for ignoring vertices that don't exist in child branches
        self.zero_mask_num = 1 - self.zero_mask.repeat(batch, 1).to(torch.uint8)

        # We compute momentum scaling factors for rotation constraints
        self.parent_MOI_index1 = parent_MOI_index1
        self.parent_MOI_index2 = parent_MOI_index2

        rod_MOI1, rod_MOI2 = self.parent_MOI_matrix[parent_MOI_index1].repeat(batch, 1, 1), self.children_MOI_matrix.repeat(batch, 1, 1)
        momentum_scale1 = -rod_MOI2 @ torch.linalg.pinv(rod_MOI1 + rod_MOI2)
        momentum_scale2 = rod_MOI1 @ torch.linalg.pinv(rod_MOI1 + rod_MOI2)
        self.momentum_scale_previous = torch.cat((momentum_scale1, momentum_scale2), dim=1).view(-1, 3, 3)

        rod_MOI1, rod_MOI2 = self.parent_MOI_matrix[parent_MOI_index2].repeat(batch, 1, 1), self.children_MOI_matrix.repeat(batch, 1, 1)
        momentum_scale1 = -rod_MOI2 @ torch.linalg.pinv(rod_MOI1 + rod_MOI2)
        momentum_scale2 = rod_MOI1 @ torch.linalg.pinv(rod_MOI1 + rod_MOI2)
        self.momentum_scale_next = torch.cat((momentum_scale1, momentum_scale2), dim=1).view(-1, 3, 3)

        # inext_scale is used for inextensibility enforcement: a high penalty for edges that must remain fixed length
        inext_scale = clamped_index * 1e20
        self.inext_scale = (inext_scale + 1.).repeat(batch, 1)
        self.inext_scale = torch.cat((self.inext_scale[:, :-1], self.inext_scale[:, 1:]), dim=1).view(-1, n_edge)

        # mass_scale is used for local mass-based updates in inextensibility enforcement
        mass_scale1 = self.mass_matrix[:, 1:] @ torch.linalg.pinv(self.mass_matrix[:, 1:] + self.mass_matrix[:, :-1])
        mass_scale2 = self.mass_matrix[:, :-1] @ torch.linalg.pinv(self.mass_matrix[:, 1:] + self.mass_matrix[:, :-1])
        self.mass_scale = torch.cat((mass_scale1, -mass_scale2), dim=1).view(-1, self.n_edge, 3, 3)

        # Next, we compute coupling mass scale for the branching points
        parent_mass = self.mass_matrix[selected_parent_index][:, rigid_body_coupling_index].view(-1, 3, 3)
        children_mass = self.mass_matrix[selected_children_index, 0]
        self.selected_children_index = selected_children_index
        mass_scale1 = children_mass @ torch.linalg.inv(parent_mass + children_mass)
        mass_scale2 = parent_mass @ torch.linalg.inv(parent_mass + children_mass)
        self.coupling_mass_scale = torch.cat((mass_scale1.unsqueeze(dim=1), -mass_scale2.unsqueeze(dim=1)), dim=1)

        # Axis angle representation for rod orientation (parent + children). Typically 3 angles per rod.
        self.rod_axis_angle = nn.Parameter(torch.zeros(3*n_branch, 3).to(device))
        self.rod_orientation = pytorch3d.transforms.rotation_conversions.axis_angle_to_quaternion(
            self.rod_axis_angle.clone()
        ).unsqueeze(dim=0).repeat(batch, 1, 1)

        # Gravity is another parameter we can tune if desired
        self.gravity = nn.Parameter(torch.tensor((0, 0, -4.81), device=device))

        # Base timestep
        self.dt = 1e-2

        # Constraint enforcement object
        self.constraints_enforcement = constraints_enforcement(n_branch)

        # Precompute masks for vectorizing internal force calculations
        self.w_masks = torch.zeros(1, n_vert, n_edge, 1).to(device)
        self.m_masks = torch.zeros(1, n_vert, n_edge, 1, 1).to(device)
        self.plusGKB_masks = torch.zeros(1, n_vert, n_edge, 1, 1).to(device)
        self.eqGKB_masks = torch.zeros(1, n_vert, n_edge, 1, 1).to(device)
        self.minusGKB_masks = torch.zeros(1, n_vert, n_edge, 1, 1).to(device)
        self.plusGH_masks_1 = torch.zeros(1, n_vert, n_edge, 1).to(device)
        self.eqGH_masks_1 = torch.zeros(1, n_vert, n_edge, 1).to(device)
        self.minusGH_masks_1 = torch.zeros(1, n_vert, n_edge, 1).to(device)
        self.plusGH_masks_2 = torch.zeros(1, n_vert, n_edge, 1).to(device)
        self.eqGH_masks_2 = torch.zeros(1, n_vert, n_edge, 1).to(device)
        self.minusGH_masks_2 = torch.zeros(1, n_vert, n_edge, 1).to(device)
        self.plusGH_masks_n = torch.zeros(1, n_vert, n_edge, 1).to(device)
        self.eqGH_masks_n = torch.zeros(1, n_vert, n_edge, 1).to(device)
        self.minusGH_masks_n = torch.zeros(1, n_vert, n_edge, 1).to(device)

        # We fill up these masks in a loop based on adjacency in the chain
        n = n_edge - 1
        for i in range(n_vert):
            for k in range(max(i - 1, 1), n_edge):
                self.w_masks[:, i, k, :] = 1
                if k < i + 2:
                    self.m_masks[:, i, k] = 1
                    if k == i - 1:
                        self.plusGKB_masks[:, i, k] = 1
                    elif k == i:
                        self.eqGKB_masks[:, i, k] = 1
                    elif k == i + 1:
                        self.minusGKB_masks[:, i, k] = 1

                if k - 1 >= (i - 1) and i > 1 and (i - 1) < n_edge:
                    self.plusGH_masks_1[:, i, k] = 1
                if k - 1 >= i and i < n_edge:
                    self.eqGH_masks_1[:, i, k] = 1
                if k - 1 >= (i + 1) and (i + 1) < n_edge:
                    self.minusGH_masks_1[:, i, k] = 1
                if k >= (i - 1) and i > 1 and (i - 1) < n_edge:
                    self.plusGH_masks_2[:, i, k] = 1
                if k >= i and i < n_edge:
                    self.eqGH_masks_2[:, i, k] = 1
                if k >= (i + 1) and (i + 1) < n_edge:
                    self.minusGH_masks_2[:, i, k] = 1

            if n >= (i - 1) and i > 1 and (i - 1) < n_edge:
                self.plusGH_masks_n[:, i, n] = 1
            if n >= i and i < n_edge:
                self.eqGH_masks_n[:, i, n] = 1
            if n >= (i + 1) and (i + 1) < n_edge:
                self.minusGH_masks_n[:, i, n] = 1

        # JB_n: a helper for bending stiffness (J is the 2D rotation matrix of 90 degrees)
        J = (
            torch.tensor([[0., -1.], [1., 0.]])
            .unsqueeze(0)
            .unsqueeze(0)
        ).repeat(self.batch * self.n_branch, self.n_edge, 1, 1)
        bend_stiffness_clamped = torch.clamp(
            self.DEFT_func.bend_stiffness,
            min=self.DEFT_func.stiff_threshold
        ).repeat(self.batch, 1, 1).view(self.batch * self.n_branch, self.n_edge)

        self.JB_n = (J * bend_stiffness_clamped[:, :, None, None])

        # Setup for residual learning
        # We store previous and next edge stiffness for parent/child edges
        self.nn_previous_bend_stiffness = torch.cat(
            (
                torch.zeros(self.batch, n_branch, 1),
                torch.clamp(self.DEFT_func.bend_stiffness, min=self.DEFT_func.stiff_threshold).repeat(self.batch, 1, 1)
            ),
            dim=-1
        ).view(self.batch, -1, 1)
        self.nn_next_bend_stiffness = torch.cat(
            (
                torch.clamp(self.DEFT_func.bend_stiffness, min=self.DEFT_func.stiff_threshold).repeat(self.batch, 1, 1),
                torch.zeros(self.batch, n_branch, 1)
            ),
            dim=-1
        ).view(self.batch, -1, 1)

        self.nn_previous_twist_stiffness = torch.cat(
            (
                torch.zeros(self.batch, n_branch, 1),
                torch.clamp(self.DEFT_func.twist_stiffness.clone(), min=self.DEFT_func.stiff_threshold).repeat(self.batch, 1, 1)
            ),
            dim=-1
        ).view(self.batch, -1, 1)
        self.nn_next_twist_stiffness = torch.cat(
            (
                torch.clamp(self.DEFT_func.twist_stiffness.clone(), min=self.DEFT_func.stiff_threshold).repeat(self.batch, 1, 1),
                torch.zeros(self.batch, n_branch, 1)
            ),
            dim=-1
        ).view(self.batch, -1, 1)

        # Instantiate the GNN for residual corrections
        in_features = 16
        hidden_features = 64
        out_features = 3

        self.learning_weight = learning_weight
        self.GNN_tree = BatchedGNNModel(
            batch,
            in_features,
            hidden_features,
            out_features,
            n_vert,
            cs_n_vert,
            rigid_body_coupling_index,
            clamp_parent,
            clamp_child1,
            clamp_child2,
            parent_clamped_selection,
            child1_clamped_selection,
            child2_clamped_selection,
            selected_child1_index,
            selected_child2_index,
            selected_parent_index,
            selected_children_index
        )


    def Rod_Init(self, batch, init_direction, m_restEdgeL, clamped_index, inference_1_batch):
        """
        Initialize rod geometry by enforcing inextensibility constraints once,
        then computing bishop frames and curvature for each edge.

        Returns:
        --------
        m_restWprev, m_restWnext: Tensors
            Material curvatures in the bishop frame for each edge.
        """
        # If we are in single-batch inference mode, we use a numba-based approach
        if inference_1_batch:
            undeformed_vert = constraints_numba.Inextensibility_Constraint_Enforcement(
                batch,
                (self.undeformed_vert.clone()).repeat(batch, 1, 1).detach().cpu().numpy().astype(np.float64),
                m_restEdgeL.detach().cpu().numpy().astype(np.float64),
                self.inext_scale.detach().cpu().numpy().astype(np.float64),
                self.mass_scale.detach().cpu().numpy().astype(np.float64),
                self.zero_mask_num
            )
            undeformed_vert = torch.from_numpy(undeformed_vert)
        else:
            undeformed_vert = self.constraints_enforcement.Inextensibility_Constraint_Enforcement(
                batch,
                (self.undeformed_vert.clone()).repeat(batch, 1, 1),
                m_restEdgeL,
                self.mass_matrix,
                clamped_index,
                self.inext_scale,
                self.mass_scale,
                self.zero_mask_num
            )

        # Compute edges for the (adjusted) undeformed shape
        m_edges = computeEdges(undeformed_vert, self.zero_mask.repeat(batch, 1))

        # Use bishop frames for the rod
        m_u0 = self.DEFT_func.compute_u0(m_edges[:, 0], init_direction.view(-1, 2, 3)[:, 0])
        m_m1, m_m2, m_kb = self.DEFT_func.computeBishopFrame(m_u0, m_edges, m_restEdgeL)
        m_restWprev, m_restWnext = self.DEFT_func.computeMaterialCurvature(m_kb, m_m1, m_m2)

        return m_restWprev, m_restWnext

    def Internal_Force_Vectorize(
        self,
        m_edges,
        clamped_index,
        m_restEdgeL,
        m_restRegionL,
        m_kb,
        m_restWprev,
        m_restWnext,
        theta_full,
        m_m1,
        m_m2
    ):
        """
        Compute internal forces arising from bending, twisting, and curvature
        using the vectorized DEFT approach.

        Returns:
        --------
        o_forces: tensor [batch * n_branch, n_vert, 3]
            The resultant internal forces for each vertex.
        """
        batch = m_kb.size()[0]
        m_theta = theta_full

        # Compute gradient of the curvature binormal wrt each vertex
        minusGKB, plusGKB, eqGKB = self.DEFT_func.computeGradientKB(m_kb, m_edges, m_restEdgeL)

        # Compute gradient of the holonomy terms
        minusGH, plusGH, eqGH = self.DEFT_func.computeGradientHolonomyTerms(m_kb, m_restEdgeL)

        # J is the 2D rotation matrix for 90 degrees
        J = rotation_matrix(torch.pi / 2. * torch.ones(batch)).to(self.device)

        # dEdtheta: derivative of energy wrt twist angle
        dEdtheta = self.DEFT_func.computedEdtheta(
            m_m1,
            m_m2,
            m_kb,
            m_theta,
            self.JB_n,
            m_restWprev,
            m_restWnext,
            m_restRegionL
        )

        # b_w1, b_w2 are the material curvature vectors for edges
        b_w1 = (
            self.w_masks
            * computeW(m_kb, torch.cat((torch.zeros(batch, 1, 3).to(self.device), m_m1[:, :-1]), dim=1),
                       torch.cat((torch.zeros(batch, 1, 3).to(self.device), m_m2[:, :-1]), dim=1))
              .unsqueeze(dim=1)
              .repeat(1, self.n_vert, 1, 1)
        )
        b_w2 = (
            self.w_masks
            * computeW(m_kb, torch.cat((torch.zeros(batch, 1, 3).to(self.device), m_m1[:, 1:]), dim=1),
                       torch.cat((torch.zeros(batch, 1, 3).to(self.device), m_m2[:, 1:]), dim=1))
              .unsqueeze(dim=1)
              .repeat(1, self.n_vert, 1, 1)
        )

        # O_GW1, O_GW2 are the gradient of W wrt the curvature binormal for adjacent edges
        # We do partial expansions using masks to accumulate relevant terms

        # Construct the bishop frames b_m1, b_m2 for edges
        b_m1 = torch.cat(
            (
                torch.zeros(batch, self.n_vert, 1, 2, 3).to(self.device),
                torch.cat(
                    (
                        m_m2.unsqueeze(dim=1).unsqueeze(dim=-2).repeat(1, self.n_vert, 1, 1, 1),
                        -m_m1.unsqueeze(dim=1).unsqueeze(dim=-2).repeat(1, self.n_vert, 1, 1, 1)
                    ),
                    -2
                )[:, :, :-1]
            ),
            dim=2
        ) * self.m_masks

        # O_GWplus1, O_GWeq1, O_GWminus1 accumulate the partial derivatives
        O_GWplus1 = torch.bmm(
            b_m1.view(-1, 2, 3),
            (plusGKB.unsqueeze(dim=1).repeat(1, self.n_vert, 1, 1, 1)).view(-1, 3, 3)
        ).view(batch, self.n_vert, self.n_edge, 2, 3) * self.plusGKB_masks

        O_GWeq1 = torch.bmm(
            b_m1.view(-1, 2, 3),
            (eqGKB.unsqueeze(dim=1).repeat(1, self.n_vert, 1, 1, 1)).view(-1, 3, 3)
        ).view(batch, self.n_vert, self.n_edge, 2, 3) * self.eqGKB_masks

        O_GWminus1 = torch.bmm(
            b_m1.view(-1, 2, 3),
            (minusGKB.unsqueeze(dim=1).repeat(1, self.n_vert, 1, 1, 1)).view(-1, 3, 3)
        ).view(batch, self.n_vert, self.n_edge, 2, 3) * self.minusGKB_masks

        O_GW1 = O_GWplus1 + O_GWeq1 + O_GWminus1

        # Similarly for b_m2
        b_m2 = torch.cat(
            (
                m_m2.unsqueeze(dim=1).unsqueeze(dim=-2).repeat(1, self.n_vert, 1, 1, 1),
                -m_m1.unsqueeze(dim=1).unsqueeze(dim=-2).repeat(1, self.n_vert, 1, 1, 1)
            ),
            -2
        ) * self.m_masks

        O_GWplus2 = torch.bmm(
            b_m2.view(-1, 2, 3),
            (plusGKB.unsqueeze(dim=1).repeat(1, self.n_vert, 1, 1, 1)).view(-1, 3, 3)
        ).view(batch, self.n_vert, self.n_edge, 2, 3) * self.plusGKB_masks

        O_GWeq2 = torch.bmm(
            b_m2.view(-1, 2, 3),
            (eqGKB.unsqueeze(dim=1).repeat(1, self.n_vert, 1, 1, 1)).view(-1, 3, 3)
        ).view(batch, self.n_vert, self.n_edge, 2, 3) * self.eqGKB_masks

        O_GWminus2 = torch.bmm(
            b_m2.view(-1, 2, 3),
            (minusGKB.unsqueeze(dim=1).repeat(1, self.n_vert, 1, 1, 1)).view(-1, 3, 3)
        ).view(batch, self.n_vert, self.n_edge, 2, 3) * self.minusGKB_masks

        O_GW2 = O_GWplus2 + O_GWeq2 + O_GWminus2

        # b_plusGH / b_eqGH / b_minusGH handle the holonomy gradients
        b_plusGH = torch.cat((torch.zeros(batch, 1, 3).to(self.device), plusGH), dim=1).unsqueeze(-2).repeat(1, 1, self.n_edge, 1)
        b_eqGH   = torch.cat((eqGH, torch.zeros(batch, 1, 3).to(self.device)), dim=1).unsqueeze(-2).repeat(1, 1, self.n_edge, 1)
        b_minusGH= torch.cat((minusGH[:, 1:], torch.zeros(batch, 2, 3).to(self.device)), dim=1).unsqueeze(-2).repeat(1, 1, self.n_edge, 1)

        b_GH1 = b_plusGH * self.plusGH_masks_1 + b_eqGH * self.eqGH_masks_1 + b_minusGH * self.minusGH_masks_1
        b_GH2 = b_plusGH * self.plusGH_masks_2 + b_eqGH * self.eqGH_masks_2 + b_minusGH * self.minusGH_masks_2
        b_GHn = b_plusGH * self.plusGH_masks_n + b_eqGH * self.eqGH_masks_n + b_minusGH * self.minusGH_masks_n

        # Subtract J * W from O_GW for twisting
        O_GW1 = O_GW1 - torch.bmm(
            (J.unsqueeze(dim=1).unsqueeze(dim=1).repeat(1, self.n_vert, self.n_edge, 1, 1)).view(-1, 2, 2),
            torch.einsum('bijc,bijd->bijcd', b_w1, b_GH1).view(-1, 2, 3)
        ).view(batch, self.n_vert, self.n_edge, 2, 3)

        O_GW2 = O_GW2 - torch.bmm(
            (J.unsqueeze(dim=1).unsqueeze(dim=1).repeat(1, self.n_vert, self.n_edge, 1, 1)).view(-1, 2, 2),
            torch.einsum('bijc,bijd->bijcd', b_w2, b_GH2).view(-1, 2, 3)
        ).view(batch, self.n_vert, self.n_edge, 2, 3)

        # Combine the two edges to get final forces
        b_m_restRegionL = (
            m_restRegionL.unsqueeze(dim=1)
            .unsqueeze(dim=-1)
            .repeat(1, self.n_vert, 1, 3)
            * self.w_masks
        )

        b_bend_stiffness1 = (
            torch.cat(
                (
                    torch.zeros(1, self.n_branch, 1).to(self.device),
                    self.DEFT_func.bend_stiffness[:, :, :-1]
                ),
                dim=2
            ).repeat(self.batch, 1, 1)
        ).view(-1, 1, self.n_edge, 1).repeat(1, self.n_vert, 1, 1)

        b_m_restWprev = m_restWprev.unsqueeze(dim=1).repeat(1, self.n_vert, 1, 1) * self.w_masks
        term1 = torch.bmm(
            torch.transpose(O_GW1.view(-1, 2, 3), 2, 1),
            (
                torch.clamp(b_bend_stiffness1, self.DEFT_func.stiff_threshold)
                * (b_w1 - b_m_restWprev)
            ).view(-1, 2, 1)
        ).view(batch, self.n_vert, self.n_edge, 3)

        b_bend_stiffness2 = (
            torch.cat(
                (
                    torch.zeros(1, self.n_branch, 1).to(self.device),
                    self.DEFT_func.bend_stiffness[:, :, 1:]
                ),
                dim=2
            ).repeat(self.batch, 1, 1)
        ).view(-1, 1, self.n_edge, 1).repeat(1, self.n_vert, 1, 1)

        b_m_restWnext = m_restWnext.unsqueeze(dim=1).repeat(1, self.n_vert, 1, 1) * self.w_masks
        term2 = torch.bmm(
            torch.transpose(O_GW2.view(-1, 2, 3), 2, 1),
            (
                torch.clamp(b_bend_stiffness2, self.DEFT_func.stiff_threshold)
                * (b_w2 - b_m_restWnext)
            ).view(-1, 2, 1)
        ).view(batch, self.n_vert, self.n_edge, 3)

        o_forces = torch.div(
            -(term1 + term2),
            b_m_restRegionL.where(b_m_restRegionL != 0, torch.tensor(1.).to(self.device))
        )
        o_forces[b_m_restRegionL == 0] = 0.

        # Add twist contribution from the last edge
        o_forces = torch.sum(o_forces, -2)
        o_forces += b_GHn[:, :, -1] * dEdtheta[:, -1].unsqueeze(dim=1).unsqueeze(dim=1)

        # Multiply by (1 - clamped_index) to zero out forces where vertices are fully clamped
        o_forces = o_forces * (
            1 - clamped_index.to(self.device)
        ).repeat(self.batch, 1).unsqueeze(dim=-1)

        return o_forces

    def External_Force(self, mass_matrix):
        """
        Compute external force contributions (like gravity, damping, etc.).
        Here we only account for gravity (and possibly velocity compensation, if used).
        """
        # Gravity
        forces = mass_matrix @ self.gravity.clone()
        # Mask out non-existent vertices
        forces[:, 1:] *= (1 - self.zero_mask.to(torch.uint8)).repeat(self.batch, 1).unsqueeze(-1)

        return forces

    def Numerical_Integration(
        self,
        mass_matrix,
        Total_force,
        b_DLOs_velocity,
        b_DLOs_vertices,
        damping,
        integration_ratio,
        dt
    ):
        """
        Update positions and velocities in a forward-Euler style integration.
        The velocity is updated with (Total_force / mass) * dt, then positions are integrated.
        """
        # velocity update
        b_DLOs_velocity = b_DLOs_velocity.clone() + (
            (
                Total_force.unsqueeze(dim=-2)
                - b_DLOs_velocity.unsqueeze(dim=-2) * damping.repeat(self.batch).clone().view(-1, 1, 1, 1)
                  * self.mass_diagonal.repeat(self.batch, 1).unsqueeze(dim=-1).unsqueeze(dim=-1)
            )
            @ torch.linalg.pinv(mass_matrix)
            * dt
        ).reshape(-1, b_DLOs_velocity.size()[1], 3)

        # position update
        b_DLOs_vertices = b_DLOs_vertices.clone() + b_DLOs_velocity * dt * integration_ratio.clone()

        return b_DLOs_vertices, b_DLOs_velocity

    def branch_forward(
        self,
        current_vert,
        init_direction,
        clamped_index,
        m_u0,
        theta_full,
        children_control_theta,
        selected_parent_index,
        selected_children_index,
        parent_theta_clamp,
        optimization_mask,
        inference_1_batch
    ):
        """
        Forward pass for computing internal + external forces for each branch,
        then returns total force and updated twist angles.
        """
        # If in single-batch inference, use stored material curvatures
        if inference_1_batch:
            m_restWprev, m_restWnext = self.m_restWprev, self.m_restWnext
        else:
            # Otherwise, re-initialize rod geometry
            m_restWprev, m_restWnext = self.Rod_Init(
                self.batch,
                init_direction,
                self.batched_m_restEdgeL,
                torch.zeros_like(clamped_index),
                inference_1_batch
            )

        # Compute current edges
        current_edges = computeEdges(current_vert, self.zero_mask.repeat(self.batch, 1))

        # Parent twist from theta_full, restricted by clamp
        parent_control_theta = theta_full[selected_parent_index][:, parent_theta_clamp]

        # Update the DEFT model state (curvature, bishop frames, etc.)
        theta_full, material_m1, material_m2, m_kb = self.DEFT_func.updateCurrentState(
            current_vert,
            m_u0,
            self.batched_m_restEdgeL,
            m_restWprev,
            m_restWnext,
            self.batched_m_restRegionL,
            self.zero_mask.repeat(self.batch, 1),
            parent_control_theta,
            children_control_theta,
            theta_full,
            selected_parent_index,
            selected_children_index,
            optimization_mask,
            parent_theta_clamp,
            inference_1_batch
        )

        # Compute internal forces
        Internal_force = self.Internal_Force_Vectorize(
            current_edges,
            clamped_index,
            self.batched_m_restEdgeL,
            self.batched_m_restRegionL,
            m_kb,
            m_restWprev,
            m_restWnext,
            theta_full,
            material_m1,
            material_m2
        )

        # Build mask for clamped vertices
        batch_clamped_index = 1 - clamped_index.unsqueeze(0).repeat(self.batch, 1, 1).view(-1, self.n_vert, 1)

        # Combine with external forces
        External_force = self.External_Force(self.mass_matrix)
        return (External_force[:, :self.n_vert] + Internal_force) * batch_clamped_index, theta_full

    def iterative_sim(
        self,
        time_horizon,
        b_DLOs_vertices_traj,
        previous_b_DLOs_vertices_traj,
        target_b_DLOs_vertices_traj,
        loss_func,
        dt,
        parent_theta_clamp,
        child1_theta_clamp,
        child2_theta_clamp,
        inference_1_batch,
        vis_type,
        vis=False,
    ):
        """
        Perform iterative simulation for 'time_horizon' steps, updating positions and velocities at each step.
        Also, apply GNN-based residual corrections and enforce constraints.

        Parameters:
        -----------
        time_horizon: int
            Number of timesteps to simulate forward.
        b_DLOs_vertices_traj: tensor
            The current BDLO trajectory data (positions).
        previous_b_DLOs_vertices_traj: tensor
            BDLO vertices from the previous timeframe (needed for velocity or continuity).
        target_b_DLOs_vertices_traj: tensor
            Ground truth reference positions for computing loss.
        loss_func: callable
            A PyTorch loss function, e.g. MSELoss.
        dt: float
            Timestep size.
        parent_theta_clamp / child1_theta_clamp / child2_theta_clamp:
            Indices for controlling which twist angles are clamped in parent/child branches.
        inference_1_batch: bool
            If True, uses the numba-based single-batch approach for constraints.
        vis_type: str
            Descriptor string used for naming plots or debugging visuals.
        vis: bool
            Whether or not to visualize each timestep.

        Returns:
        --------
        traj_loss_eval: float
            Accumulated position loss over all timesteps.
        total_loss: float
            Accumulated total loss (position + velocity) over all timesteps.
        """
        # Number of constraint solution iterations per timestep
        constraint_loop = 20

        # Prepare input to GNN
        inputs = torch.zeros_like(target_b_DLOs_vertices_traj)

        # If branches are clamped, copy the ground-truth clamp positions
        parent_fix_point = None
        child1_fix_point = None
        child2_fix_point = None

        if self.clamp_parent:
            parent_fix_point = target_b_DLOs_vertices_traj[:, :, 0, self.parent_clamped_selection]
            inputs[:, :, 0, self.parent_clamped_selection] = parent_fix_point

        if self.clamp_child1:
            child1_fix_point = target_b_DLOs_vertices_traj[:, :, 1, self.child1_clamped_selection]
            inputs[:, :, 1, self.child1_clamped_selection] = child1_fix_point

        if self.clamp_child2:
            child2_fix_point = target_b_DLOs_vertices_traj[:, :, 2, self.child2_clamped_selection]
            inputs[:, :, 2, self.child2_clamped_selection] = child2_fix_point

        # Initialize accumulators for losses
        traj_loss_eval = 0.0
        total_loss = 0.0

        # Initialize orientation/twist states
        parent_rod_orientation = None
        children_rod_orientation = None
        theta_full = None
        optimization_mask = None

        # For parent-child constraints iteration
        previous_parent_vertices_iteration_edge1 = None
        previous_parent_vertices_iteration_edge2 = None
        previous_children_vertices_iteration_edge = None

        # For storing the updated states after each iteration
        b_DLOs_vertices_old = None
        b_DLOs_velocity = None
        m_u0 = None

        # Precompute zero mask repeated for the batch
        zero_mask_batched = self.zero_mask.repeat(self.batch, 1)

        # Initialize bishop frames for the first time
        self.m_restWprev, self.m_restWnext = self.Rod_Init(
            self.batch,
            torch.tensor([[0.0, 0.6, 0.8], [0.0, 0.0, 1.0]])  # example initial directions
                .unsqueeze(dim=0)
                .repeat(self.batch, self.n_branch, 1, 1),
            self.batched_m_restEdgeL,
            torch.zeros_like(self.clamped_index),
            inference_1_batch
        )

        # Main loop over timesteps
        for ith in range(time_horizon):
            # 1) Retrieve current/previous BDLO states
            if ith == 0:
                b_DLOs_vertices = b_DLOs_vertices_traj[:, ith].reshape(-1, self.n_vert, 3)
                prev_b_DLOs_vertices = previous_b_DLOs_vertices_traj[:, ith].reshape(-1, self.n_vert, 3)
            else:
                prev_b_DLOs_vertices = b_DLOs_vertices_old.clone()

            # 2) Initialize or parallel transport the material-frame directions m_u0
            if ith == 0:
                rest_edges = computeEdges(b_DLOs_vertices, zero_mask_batched)
                init_direction = torch.tensor([[0.0, 0.6, 0.8], [0.0, 0.0, 1.0]]) \
                    .unsqueeze(dim=0) \
                    .repeat(self.n_branch, 1, 1)
                m_u0 = self.DEFT_func.compute_u0(
                    rest_edges[:, 0],
                    init_direction.repeat(self.batch, 1, 1)[:, 0]
                )

                # Initialize rod orientation for parent + children
                parent_rod_axis_angle = torch.zeros(1, 3)
                parent_rod_orientation = pytorch3d.transforms.rotation_conversions.axis_angle_to_quaternion(
                    parent_rod_axis_angle
                ).unsqueeze(dim=0).repeat(self.batch, self.n_vert - 1, 1)

                child_rod_axis_angle = torch.zeros(1, 3)
                children_rod_orientation = pytorch3d.transforms.rotation_conversions.axis_angle_to_quaternion(
                    child_rod_axis_angle
                ).unsqueeze(dim=0).repeat(self.batch, len(self.rigid_body_coupling_index), 1)

                # Initialize twist angles along the branches
                rigid_body_orientation_axis_angle = pytorch3d.transforms.rotation_conversions \
                    .quaternion_to_axis_angle(parent_rod_orientation[:, self.fused_rigid_body_coupling_index]) \
                    .view(-1, 2, 3)
                angles = torch.norm(rigid_body_orientation_axis_angle, dim=2) + 1e-20
                axes = rigid_body_orientation_axis_angle / angles.unsqueeze(2)
                children_axis = torch.nn.functional.normalize(
                    b_DLOs_vertices[self.selected_children_index, 1] - b_DLOs_vertices[self.selected_children_index, 0],
                    dim=1
                ).unsqueeze(dim=1).repeat(1, 2, 1)
                children_rotation_angles = ((children_axis * axes).sum(-1) * angles).sum(-1)

                # Build theta_full and an optimization_mask
                theta_full = torch.zeros(self.batch * self.n_branch, self.n_vert - 1)
                theta_full[self.selected_children_index, 0] = children_rotation_angles

                optimization_mask = 1 - torch.zeros_like(theta_full).unsqueeze(1)

                # Apply clamp for the parent branch angles
                if self.clamp_parent:
                    for p_idx in parent_theta_clamp:
                        optimization_mask[self.selected_parent_index, :, int(p_idx)] = 0

                # The first child edge is zero => no update
                optimization_mask[self.selected_children_index, :, 0] = 0

                # Child1 clamp
                if self.clamp_child1:
                    optimization_mask[self.selected_child1_index, :, child1_theta_clamp] = 0

                # Child2 clamp
                if self.clamp_child2:
                    optimization_mask[self.selected_child2_index, :, child2_theta_clamp] = 0

            else:
                # Parallel transport bishop frames
                previous_edge = computeEdges(prev_b_DLOs_vertices, zero_mask_batched)
                current_edge = computeEdges(b_DLOs_vertices, zero_mask_batched)
                m_u0 = self.DEFT_func.parallelTransportFrame(
                    previous_edge[:, 0],
                    current_edge[:, 0],
                    m_u0
                )

                # Update children rotation angles
                rigid_body_orientation_axis_angle = pytorch3d.transforms.rotation_conversions \
                    .quaternion_to_axis_angle(
                        parent_rod_orientation[:, self.fused_rigid_body_coupling_index]
                    ).view(-1, 2, 3)
                angles = torch.norm(rigid_body_orientation_axis_angle, dim=2) + 1e-20
                axes = rigid_body_orientation_axis_angle / angles.unsqueeze(2)
                children_axis = torch.nn.functional.normalize(
                    b_DLOs_vertices[self.selected_children_index, 1] - b_DLOs_vertices[self.selected_children_index, 0],
                    dim=1
                ).unsqueeze(dim=1).repeat(1, 2, 1)
                children_rotation_angles = ((children_axis * axes).sum(-1) * angles).sum(-1)
                theta_full[self.selected_children_index, 0] = children_rotation_angles

            # 3) DEFT forward pass to get total forces + updated twist angles
            Total_force, theta_full = self.branch_forward(
                b_DLOs_vertices,
                torch.tensor([[0.0, 0.6, 0.8], [0.0, 0.0, 1.0]])
                    .unsqueeze(dim=0)
                    .repeat(self.batch, self.n_branch, 1, 1),
                self.clamped_index,
                m_u0,
                theta_full,
                -children_rotation_angles if ith > 0 else -children_rotation_angles,
                self.selected_parent_index,
                self.selected_children_index,
                parent_theta_clamp,
                optimization_mask,
                inference_1_batch
            )

            # Initialize velocity (from difference) for the first frame
            if ith == 0:
                b_DLOs_velocity = (b_DLOs_vertices - prev_b_DLOs_vertices) / dt

            # Perform numerical integration
            prev_b_DLOs_vertices_copy = b_DLOs_vertices.clone()
            b_DLOs_vertices, b_DLOs_velocity = self.Numerical_Integration(
                self.mass_matrix,
                Total_force,
                b_DLOs_velocity,
                b_DLOs_vertices,
                self.damping,
                self.integration_ratio,
                dt
            )

            # 4) GNN-based residual correction
            current_input = inputs[:, ith].reshape(self.batch, -1, 3)
            gnn_input = torch.cat([
                b_DLOs_vertices.view(self.batch, -1, 3),
                prev_b_DLOs_vertices.view(self.batch, -1, 3),
                current_input,
                self.nn_previous_bend_stiffness,
                self.nn_next_bend_stiffness,
                self.nn_previous_twist_stiffness,
                self.nn_next_twist_stiffness,
                self.undeformed_vert.unsqueeze(0).repeat(self.batch, 1, 1, 1).view(self.batch, -1, 3),
            ], dim=-1)

            delta_b_DLOs_vertices = self.GNN_tree.inference(gnn_input, current_input) * self.learning_weight * dt
            delta_b_DLOs_vertices = delta_b_DLOs_vertices.view(-1, self.n_vert, 3)

            # Enforce clamp constraints on the GNN delta
            if self.clamp_parent:
                parent_fix = parent_fix_point[:, ith].reshape(-1, 3)
                b_DLOs_vertices[self.batch_indices_flat, self.parent_indices_flat] = parent_fix
                delta_b_DLOs_vertices[self.batch_indices_flat, self.parent_indices_flat] = 0

            if self.clamp_child1:
                c1_fix = child1_fix_point[:, ith].reshape(-1, 3)
                b_DLOs_vertices[self.batch_child1_indices_flat, self.child1_indices_flat] = c1_fix
                delta_b_DLOs_vertices[self.batch_child1_indices_flat, self.child1_indices_flat] = 0

            if self.clamp_child2:
                c2_fix = child2_fix_point[:, ith].reshape(-1, 3)
                b_DLOs_vertices[self.batch_child2_indices_flat, self.child2_indices_flat] = c2_fix
                delta_b_DLOs_vertices[self.batch_child2_indices_flat, self.child2_indices_flat] = 0

            # Apply the GNN correction
            b_DLOs_vertices = b_DLOs_vertices + delta_b_DLOs_vertices

            # 5) Constraints Enforcement (rotational and inextensibility)
            if ith == 0:
                previous_parent_vertices_iteration_edge1 = b_DLOs_vertices[self.selected_parent_index].clone()
                previous_parent_vertices_iteration_edge2 = b_DLOs_vertices[self.selected_parent_index].clone()
                previous_children_vertices_iteration_edge = b_DLOs_vertices[self.selected_children_index].view(self.batch, -1, self.n_vert, 3).clone()

                if inference_1_batch:
                    previous_parent_vertices_iteration_edge1 = previous_parent_vertices_iteration_edge1.detach().cpu().numpy().astype(np.float64).copy()
                    previous_parent_vertices_iteration_edge2 = previous_parent_vertices_iteration_edge2.detach().cpu().numpy().astype(np.float64).copy()
                    previous_children_vertices_iteration_edge = previous_children_vertices_iteration_edge.detach().cpu().numpy().astype(np.float64).copy()

            if inference_1_batch:
                parent_rod_orientation = parent_rod_orientation.detach().cpu().numpy().astype(np.float64).copy()
                children_rod_orientation = children_rod_orientation.detach().cpu().numpy().astype(np.float64).copy()

                b_DLOs_vertices = b_DLOs_vertices.detach().cpu().numpy().astype(np.float64).copy()
                rotation_constraints_index1 = torch.linspace(0, (self.n_branch-1) * 2 - 2, len(self.rigid_body_coupling_index)).to(torch.int).detach().cpu().numpy().copy()
                rotation_constraints_index2 = torch.linspace(1, ((self.n_branch-1) * 2 - 1), len(self.rigid_body_coupling_index)).to(torch.int).detach().cpu().numpy().copy()
                parent_MOI_matrix_numpy = self.parent_MOI_matrix.detach().cpu().numpy().astype(np.float64).copy()
                children_MOI_matrix_numpy = self.children_MOI_matrix.detach().cpu().numpy().astype(np.float64).copy()
                momentum_scale_previous_numpy = self.momentum_scale_previous.detach().cpu().numpy().astype(np.float64).copy()
                momentum_scale_next_numpy = self.momentum_scale_next.detach().cpu().numpy().astype(np.float64).copy()
                coupling_mass_scale_numpy = self.coupling_mass_scale.detach().cpu().numpy().astype(np.float64).copy()
                batched_m_restEdgeL_numpy = self.batched_m_restEdgeL.detach().cpu().numpy().astype(np.float64).copy()
                inext_scale_numpy = self.inext_scale.detach().cpu().numpy().astype(np.float64).copy()
                mass_scale_numpy = self.mass_scale.detach().cpu().numpy().astype(np.float64).copy()

                # Repeatedly enforce rotation and inextensibility constraints
                for constraint_loop_i in range(constraint_loop):
                    parent_vertices = b_DLOs_vertices[self.selected_parent_index].reshape((self.batch, self.n_vert, 3))
                    children_vertices = b_DLOs_vertices[self.selected_children_index].reshape((self.batch, -1, self.n_vert, 3))

                    # Edge1
                    parent_vertices, parent_rod_orientation, children_vertices, children_rod_orientation = \
                        constraints_numba.Rotation_Constraints_Enforcement_Parent_Children(
                            parent_vertices,
                            parent_rod_orientation,
                            previous_parent_vertices_iteration_edge1,
                            children_vertices,
                            children_rod_orientation,
                            previous_children_vertices_iteration_edge,
                            parent_MOI_matrix_numpy,
                            children_MOI_matrix_numpy,
                            np.array(self.rigid_body_coupling_index) - 1,
                            rotation_constraints_index1,
                            momentum_scale_previous_numpy
                        )

                    previous_parent_vertices_iteration_edge1 = parent_vertices.copy()
                    previous_children_vertices_iteration_edge = children_vertices.copy()

                    # Edge2
                    parent_vertices, parent_rod_orientation, children_vertices, children_rod_orientation = \
                        constraints_numba.Rotation_Constraints_Enforcement_Parent_Children(
                            parent_vertices,
                            parent_rod_orientation,
                            previous_parent_vertices_iteration_edge2,
                            children_vertices,
                            children_rod_orientation,
                            previous_children_vertices_iteration_edge,
                            parent_MOI_matrix_numpy,
                            children_MOI_matrix_numpy,
                            np.array(self.rigid_body_coupling_index),
                            rotation_constraints_index2,
                            momentum_scale_next_numpy
                        )

                    previous_parent_vertices_iteration_edge2 = parent_vertices.copy()
                    previous_children_vertices_iteration_edge = children_vertices.copy()

                    # Coupling constraints among parent and child
                    children_vertices = children_vertices.reshape((-1, self.n_vert, 3))
                    b_DLOs_vertices = constraints_numba.Inextensibility_Constraint_Enforcement_Coupling(
                        parent_vertices,
                        children_vertices,
                        np.array(self.rigid_body_coupling_index),
                        coupling_mass_scale_numpy,
                        self.selected_parent_index,
                        self.selected_children_index
                    )

                    b_DLOs_vertices = constraints_numba.Inextensibility_Constraint_Enforcement(
                        self.batch,
                        b_DLOs_vertices,
                        batched_m_restEdgeL_numpy,
                        inext_scale_numpy,
                        mass_scale_numpy,
                        self.zero_mask_num
                    )

                b_DLOs_vertices = torch.from_numpy(b_DLOs_vertices).to(torch.float64)
                parent_rod_orientation = torch.from_numpy(parent_rod_orientation).to(torch.float64)
                children_rod_orientation = torch.from_numpy(children_rod_orientation).to(torch.float64)
            else:
                for _ in range(constraint_loop):
                    parent_vertices = b_DLOs_vertices[self.selected_parent_index]
                    children_vertices = b_DLOs_vertices[self.selected_children_index].view(self.batch, -1, self.n_vert, 3)

                    # Edge1
                    parent_vertices, parent_rod_orientation, children_vertices, children_rod_orientation = \
                        self.constraints_enforcement.Rotation_Constraints_Enforcement_Parent_Children(
                            parent_vertices,
                            parent_rod_orientation,
                            previous_parent_vertices_iteration_edge1,
                            children_vertices,
                            children_rod_orientation,
                            previous_children_vertices_iteration_edge,
                            self.parent_MOI_matrix,
                            self.children_MOI_matrix,
                            torch.tensor(self.rigid_body_coupling_index) - 1,
                            torch.linspace(0, (children_vertices.size(1) * 2 - 2), len(self.rigid_body_coupling_index)).to(torch.int),
                            self.momentum_scale_previous
                        )

                    previous_parent_vertices_iteration_edge1 = parent_vertices.clone()
                    previous_children_vertices_iteration_edge = children_vertices.clone()

                    # Edge2
                    parent_vertices, parent_rod_orientation, children_vertices, children_rod_orientation = \
                        self.constraints_enforcement.Rotation_Constraints_Enforcement_Parent_Children(
                            parent_vertices,
                            parent_rod_orientation,
                            previous_parent_vertices_iteration_edge2,
                            children_vertices,
                            children_rod_orientation,
                            previous_children_vertices_iteration_edge,
                            self.parent_MOI_matrix,
                            self.children_MOI_matrix,
                            torch.tensor(self.rigid_body_coupling_index),
                            torch.linspace(1, (children_vertices.size(1) * 2 - 1), len(self.rigid_body_coupling_index)).to(torch.int),
                            self.momentum_scale_next
                        )
                    previous_parent_vertices_iteration_edge2 = parent_vertices.clone()
                    previous_children_vertices_iteration_edge = children_vertices.clone()

                    # Coupling constraints (parent <-> children rods)
                    children_vertices = children_vertices.view(-1, self.n_vert, 3)
                    b_DLOs_vertices = self.constraints_enforcement.Inextensibility_Constraint_Enforcement_Coupling(
                        parent_vertices,
                        children_vertices,
                        self.rigid_body_coupling_index,
                        self.coupling_mass_scale,
                        self.selected_parent_index,
                        self.selected_children_index
                    )

                    # Finally, general inextensibility constraints along each branch
                    b_DLOs_vertices = self.constraints_enforcement.Inextensibility_Constraint_Enforcement(
                        self.batch,
                        b_DLOs_vertices,
                        self.batched_m_restEdgeL,
                        self.mass_matrix,
                        self.clamped_index,
                        self.inext_scale,
                        self.mass_scale,
                        self.zero_mask_num
                    )

            # 6) Update velocities based on final positions + compute losses
            b_DLOs_velocity = (b_DLOs_vertices - prev_b_DLOs_vertices_copy) / dt

            gt_vertices = target_b_DLOs_vertices_traj[:, ith].reshape(-1, self.n_vert, 3)
            gt_velocity = (
                (target_b_DLOs_vertices_traj[:, ith] - b_DLOs_vertices_traj[:, ith]).view(-1, self.n_vert, 3) / dt
            )

            # Position and velocity loss
            step_loss_pos = loss_func(gt_vertices, b_DLOs_vertices)
            step_loss_vel = loss_func(b_DLOs_velocity, gt_velocity)
            traj_loss_eval += step_loss_pos
            total_loss += (step_loss_pos + step_loss_vel)

            # Visualization if requested
            if vis:
                vis_batch = self.batch  # how many samples we visualize
                for i_eval_batch in range(vis_batch):
                    parent_vertices_traj_vis = target_b_DLOs_vertices_traj[i_eval_batch][:, 0]
                    child1_vertices_traj_vis = target_b_DLOs_vertices_traj[i_eval_batch][:, 1]
                    child2_vertices_traj_vis = target_b_DLOs_vertices_traj[i_eval_batch][:, 2]

                    child1_vertices_vis = torch.cat(
                        (
                            parent_vertices_traj_vis[ith, self.rigid_body_coupling_index[0]].unsqueeze(0),
                            child1_vertices_traj_vis[ith]
                        ),
                        dim=0
                    )
                    child2_vertices_vis = torch.cat(
                        (
                            parent_vertices_traj_vis[ith, self.rigid_body_coupling_index[1]].unsqueeze(0),
                            child2_vertices_traj_vis[ith]
                        ),
                        dim=0
                    )

                    parent_vertices_pred = b_DLOs_vertices[self.selected_parent_index]
                    children_vertices_pred = b_DLOs_vertices[self.selected_children_index].view(self.batch, -1, 3)

                    visualize_tensors_3d_in_same_plot_no_zeros(
                        self.parent_clamped_selection,
                        parent_vertices_pred[i_eval_batch],
                        children_vertices_pred[i_eval_batch],
                        ith,
                        0,
                        self.clamp_parent,
                        self.clamp_child1,
                        self.clamp_child2,
                        parent_fix_point[:, ith].reshape(-1, 3) if self.clamp_parent else None,
                        child1_fix_point[:, ith].reshape(-1, 3) if self.clamp_child1 else None,
                        child2_fix_point[:, ith].reshape(-1, 3) if self.clamp_child2 else None,
                        parent_vertices_traj_vis[ith],
                        child1_vertices_vis,
                        child2_vertices_vis,
                        i_eval_batch,
                        vis_type
                    )

            # Save updated positions for the next iteration
            b_DLOs_vertices_old = b_DLOs_vertices.clone()

        # Return the accumulated losses
        return traj_loss_eval, total_loss
