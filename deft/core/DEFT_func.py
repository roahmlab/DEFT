import torch
import torch.nn as nn
import torch.nn.functional as F
import theseus as th

# Custom utility functions
from ..utils.util import (
    extractSinandCos,
    computeEdges,
    computeKB,
    quaternion_q,
    computeW,
    skew_symmetric,
    scalar_func,
    quaternion_rotation,
    quaternion_rotation_parallel,
)
from ..solvers.theta_solver_numpy import theta_optimize_numpy
from ..solvers.theta_solver import theta_optimize


class DEFT_func(nn.Module):
    """
    A differentiable rod-modeling class that implements the Discrete Elastic Rods
    theory (DER). It uses a "Bishop frame" approach for computing rod kinematics
    (curvature, twist, etc.) and works with a solver to minimize the elastic
    energy of the rod segments.

    Args:
        batch (int): Batch size for parallel processing.
        n_branch (int): Number of branches in the rod structure (if multiple rods or branching).
        n_vert (int): Number of vertices in the rod discretization.
        n_edge (int): Number of edges (n_vert-1 typically).
        edge_mask (torch.Tensor): A mask indicating which edges are valid/active.
        zero_mask (torch.Tensor): A mask indicating which vertices are valid/active.
        m_restEdgeL (torch.Tensor): The undeformed (rest) lengths of all edges.
        bend_stiffness_parent (torch.Tensor): Per-edge bending stiffness for the parent rod segment.
        bend_stiffness_child1 (torch.Tensor): Per-edge bending stiffness for child rod segment #1.
        bend_stiffness_child2 (torch.Tensor): Per-edge bending stiffness for child rod segment #2.
        twist_stiffness (torch.Tensor): Per-edge twisting stiffness.
        device (torch.device): CPU or GPU device.

    """
    def __init__(
        self,
        batch,
        n_branch,
        n_vert,
        n_edge,
        edge_mask,
        zero_mask,
        m_restEdgeL,
        bend_stiffness_parent,
        bend_stiffness_child1,
        bend_stiffness_child2,
        twist_stiffness,
        device,
    ):
        super().__init__()

        # Initialize basic properties
        self.batch = batch
        self.n_vert = n_vert
        self.n_edge = n_edge
        self.device = device
        self.n_branch = n_branch

        # Combine bending stiffness from parent, child1, and child2
        self.bend_stiffness_parent = bend_stiffness_parent
        self.bend_stiffness_child1 = bend_stiffness_child1
        self.bend_stiffness_child2 = bend_stiffness_child2
        self.bend_stiffness = torch.cat(
            (self.bend_stiffness_parent,
             self.bend_stiffness_child1,
             self.bend_stiffness_child2),
            dim=1
        )
        self.twist_stiffness = twist_stiffness

        # Error/stiffness thresholds for numerical stability
        self.err = torch.tensor(1e-6).to(device)
        self.stiff_threshold = torch.tensor(1e-10).to(device)

        # For batch-wise operations, repeat/reshape stiffness arrays
        # bend_stiffness_last_unsq and bend_stiffness_next_unsq correspond to
        # edges [:-1] and edges [1:] respectively, repeated over the batch.
        self.bend_stiffness_last_unsq = (
            self.bend_stiffness[:, :, :-1]
            .repeat(batch, 1, 1)
            .view(-1, n_edge - 1, 1)
        )
        self.bend_stiffness_next_unsq = (
            self.bend_stiffness[:, :, 1:]
            .repeat(batch, 1, 1)
            .view(-1, n_edge - 1, 1)
        )
        self.twist_stiffness_unsq = (
            self.twist_stiffness[:, :, 1:]
            .repeat(batch, 1, 1)
            .view(-1, self.n_edge - 1)
        )

        # Repeat edge_mask for batch dimension
        # shape: (batch*n_branch, n_edge, 1)
        self.edge_mask = edge_mask.repeat(self.batch, 1, 1).view(-1, self.n_edge, 1)

    def compute_u0(self, e0, init_direct):
        """
        Compute the initial (first edge) Bishop-frame direction 'u0',
        ensuring it is perpendicular to the first edge e0.

        Args:
            e0 (torch.Tensor): The first edge vector, shape (batch, 3).
            init_direct (torch.Tensor): A random 3D vector not parallel to e0.

        Returns:
            u0 (torch.Tensor): The 'u' axis of the Bishop frame for the first edge,
                               normalized and perpendicular to e0.
        """
        batch = e0.size(0)

        # Compute a normal vector N_0 that is perpendicular to e0
        # and init_direct by cross product
        N_0 = torch.cross(e0, init_direct.view(batch, -1), dim=1)

        # Compute u0 by crossing N_0 with e0 and normalizing
        u0 = F.normalize(torch.cross(N_0, e0, dim=1), dim=1)
        return u0

    def computeBishopFrame(self, u0, edges, restEdgeL):
        """
        Compute the Bishop frames (u, v) and curvature binormal kb
        for all edges based on the initial frame u0.

        Bishop Frame is a twist-minimizing reference frame for rods,
        described in the Discrete Elastic Rods (DER) literature.

        Args:
            u0 (torch.Tensor): Initial Bishop frame 'u' for the first edge, shape (batch, 3).
            edges (torch.Tensor): Edge vectors of the rod, shape (batch, n_edge, 3).
            restEdgeL (torch.Tensor): Undeformed edge lengths, shape (batch, n_edge).

        Returns:
            b_u (torch.Tensor): The Bishop frame 'u' for each edge, shape (batch, n_edge, 3).
            b_v (torch.Tensor): The Bishop frame 'v' for each edge, shape (batch, n_edge, 3).
            kb (torch.Tensor): The discrete curvature binormal, shape (batch, n_edge, 3).
        """
        batch = edges.size(0)

        # Compute curvature binormal kb from edges and rest lengths
        # DER eqn (1)
        kb = computeKB(edges, restEdgeL)

        # Initialize b_u with the first edge's Bishop frame 'u0'
        b_u = u0.unsqueeze(dim=1)

        # Magnitude of kb for each edge
        magnitude = (kb * kb).sum(dim=2)

        # Extract sinPhi and cosPhi from curvature magnitudes
        sinPhi, cosPhi = extractSinandCos(magnitude)

        # Convert (cosPhi, sinPhi*kb) to quaternions for rotation
        q = quaternion_q(cosPhi, sinPhi.unsqueeze(dim=2) * F.normalize(kb, dim=2))

        # Recursively compute b_u for each edge using the Bishop update:
        # b_u(i+1) = rotate(b_u(i)) by the quaternion that aligns edges i -> i+1.
        for i in range(1, self.n_edge):
            # If rotation is very small (cosPhi ~ 1), skip the rotation
            # otherwise apply quaternion rotation to get new b_u
            b_u = torch.cat(
                (
                    b_u,
                    torch.where(
                        (1 - cosPhi[:, i].unsqueeze(dim=1)) <= self.err,
                        b_u[:, i - 1],
                        quaternion_rotation(b_u, edges, q, i)[0][:, 0, :],
                    ).unsqueeze(dim=1),
                ),
                dim=1,
            )

        # b_v is orthogonal to the edge and b_u
        b_v = F.normalize(torch.cross(edges, b_u, dim=2), dim=2)

        # Mask out invalid edges (if any) using edge_mask
        b_u = b_u * self.edge_mask
        b_v = b_v * self.edge_mask

        return b_u, b_v, kb

    def computeMaterialCurvature(self, kb, m1, m2):
        """
        Compute the material curvature (kappa1, kappa2) in the material frame
        given the discrete curvature binormal kb and the material frames m1, m2.

        From DER eqn (2): kappa = (kappa . m1, kappa . m2).

        Args:
            kb (torch.Tensor): Discrete curvature binormal, shape (batch, n_edge, 3).
            m1 (torch.Tensor): Material frame axis 1, shape (batch, n_edge, 3).
            m2 (torch.Tensor): Material frame axis 2, shape (batch, n_edge, 3).

        Returns:
            m_W1 (torch.Tensor): Material curvature wrt m1, shape (batch, n_edge, 2).
            m_W2 (torch.Tensor): Material curvature wrt m2, shape (batch, n_edge, 2).
        """
        batch, n_edge = kb.size(0), kb.size(1)

        # Initialize outputs
        m_W1 = torch.zeros(batch, n_edge, 2).to(self.device)
        m_W2 = torch.zeros(batch, n_edge, 2).to(self.device)

        # computeW returns (kappa . m1, kappa . m2) from discrete curvature binormal
        # Indices [1:] because the curvature for the first edge is zero or not used
        m_W1[:, 1:] = computeW(kb[:, 1:], m1[:, :-1], m2[:, :-1])
        m_W2[:, 1:] = computeW(kb[:, 1:], m1[:, 1:], m2[:, 1:])
        return m_W1, m_W2

    def parallelTransportFrame(self, previous_e0, current_e0, b_u0):
        """
        Update the first edge's Bishop frame vector after a change in the first edge,
        ensuring orthonormality. (Section 4.2.2 "Bishop Frame Update" in DER paper.)

        Args:
            previous_e0 (torch.Tensor): The previous (old) first-edge vector, shape (batch, 3).
            current_e0 (torch.Tensor): The new (updated) first-edge vector, shape (batch, 3).
            b_u0 (torch.Tensor): The previous Bishop frame 'u' for the first edge, shape (batch, 3).

        Returns:
            b_u0 (torch.Tensor): The updated, orthonormal Bishop frame 'u' for the current first edge.
        """
        batch = previous_e0.size(0)

        # Axis for rotation is cross(previous_e0, current_e0) normalized
        # The factor in the denominator is from the Rodrigues' rotation formula
        axis = 2.0 * torch.cross(previous_e0, current_e0, dim=1) / (
            previous_e0.norm(dim=1) * current_e0.norm(dim=1)
            + (previous_e0 * current_e0).sum(dim=1)
        ).unsqueeze(dim=1)

        # Magnitude of the rotation axis
        magnitude = (axis * axis).sum(dim=1)
        sinPhi, cosPhi = extractSinandCos(magnitude)

        # If the rotation is negligible, skip it; otherwise rotate b_u0
        b_u0 = torch.where(
            (1 - cosPhi.unsqueeze(dim=1)) <= self.err,
            F.normalize(
                torch.cross(torch.cross(current_e0, b_u0, dim=1), current_e0, dim=1),
                dim=1,
            ),
            quaternion_rotation_parallel(cosPhi, sinPhi, axis, b_u0),
        )
        return b_u0

    def computeMaterialFrame(self, m_theta, b_u, b_v):
        """
        Construct the material frame axes (m1, m2) by rotating the Bishop frame
        about the edge tangent by angle m_theta.

        From DER paper, Section 4.1, the rotation of the Bishop frame (b_u, b_v)
        by angle theta forms the material frame.

        Args:
            m_theta (torch.Tensor): The twist angle about the edge tangent, shape (batch, n_edge).
            b_u (torch.Tensor): The Bishop 'u' axis, shape (batch, n_edge, 3).
            b_v (torch.Tensor): The Bishop 'v' axis, shape (batch, n_edge, 3).

        Returns:
            m1 (torch.Tensor): Material frame axis #1, shape (batch, n_edge, 3).
            m2 (torch.Tensor): Material frame axis #2, shape (batch, n_edge, 3).
        """
        # Reshape for broadcast
        cosQ = torch.cos(m_theta.clone()).unsqueeze(dim=2)
        sinQ = torch.sin(m_theta.clone()).unsqueeze(dim=2)

        # Standard 2D rotation about the tangent:
        # [m1, m2] = [ cosQ * b_u + sinQ * b_v, -sinQ * b_u + cosQ * b_v ]
        m1 = cosQ * b_u + sinQ * b_v
        m2 = -sinQ * b_u + cosQ * b_v
        return m1, m2

    def updateCurrentState(
        self,
        current_vert,
        b_u0,
        restEdgeL,
        m_restW1,
        m_restW2,
        restRegionL,
        zero_mask,
        parent_end_theta,
        children_end_theta,
        theta_full,
        selected_parent_index,
        selected_children_index,
        optimization_mask,
        parent_theta_clamp,
        inference_1_batch
    ):
        """
        Update the rod's current state (material frames, twist angles, etc.)
        by computing the energy-minimizing twist angle distribution.
        Uses a solver that sets dE/dtheta = 0.

        Args:
            current_vert (torch.Tensor): Current vertex positions, shape (batch, n_vert, 3).
            b_u0 (torch.Tensor): The Bishop frame 'u' for the first edge, shape (batch, 3).
            restEdgeL (torch.Tensor): Rest edge lengths, shape (batch, n_edge).
            m_restW1 (torch.Tensor): Rest material curvature (kappa . m1), shape (batch, n_edge, 2).
            m_restW2 (torch.Tensor): Rest material curvature (kappa . m2), shape (batch, n_edge, 2).
            restRegionL (torch.Tensor): Voronoi region lengths, shape (batch, n_edge).
            zero_mask (torch.Tensor): Mask for valid vertices.
            parent_end_theta (torch.Tensor): Specified twist angle for the parent rod's end.
            children_end_theta (torch.Tensor): Specified twist angle for children rods' end.
            theta_full (torch.Tensor): Initial guess or previously computed distribution of twist angles.
            selected_parent_index (list): Indices corresponding to parent rods for end conditions.
            selected_children_index (list): Indices corresponding to child rods for end conditions.
            optimization_mask (torch.Tensor): Mask that indicates which edges to consider in optimization.
            parent_theta_clamp (torch.Tensor): Additional clamp information for parent angles.
            inference_1_batch (bool): If True, uses numpy-based solver for single-batch inference.

        Returns:
            theta_full (torch.Tensor): Updated twist angle distribution (solution of energy minimization).
            m1 (torch.Tensor): Material frame axis #1, shape (batch, n_edge, 3).
            m2 (torch.Tensor): Material frame axis #2, shape (batch, n_edge, 3).
            kb (torch.Tensor): Curvature binormal after the update, shape (batch, n_edge, 3).
        """
        # 1. Compute current edges from current vertex positions
        current_edges = computeEdges(current_vert, zero_mask)

        # 2. Compute the current Bishop frame based on first-edge bishop 'u0'
        b_u, b_v, kb = self.computeBishopFrame(b_u0, current_edges, restEdgeL)

        # Prepare boundary conditions for end twist angles
        # (parent, children, clamp, etc.)
        children_end_theta = torch.cat(
            (
                children_end_theta.unsqueeze(dim=1),
                torch.zeros_like(
                    children_end_theta.unsqueeze(dim=1).repeat(1, len(parent_theta_clamp) - 1)
                ),
            ),
            dim=1,
        )
        end_theta = torch.empty(
            len(selected_parent_index) + len(selected_children_index),
            len(parent_theta_clamp)
        )

        # Fill in parent and child end angles
        end_theta[selected_parent_index] = parent_end_theta
        end_theta[selected_children_index] = children_end_theta

        # 3. Create Theseus Variables for optimization
        cost_weight_variable = th.ScaleCostWeight(1.0)
        kb_variable = th.Variable(kb, name="kb")
        b_u_variable = th.Variable(b_u, name="b_u")
        b_v_variable = th.Variable(b_v, name="b_v")
        m_restW1_variable = th.Variable(m_restW1, name="m_restW1")
        m_restW2_variable = th.Variable(m_restW2, name="m_restW2")
        restRegionL_variable = th.Variable(restRegionL, name="restRegionL")
        target_energy_variable = th.Variable(
            torch.zeros((kb.size()[0], 1)), name="target_energy"
        )
        inner_theta_variable = th.Vector(b_u.size()[1], name="inner_theta_variable")
        end_theta = th.Variable(end_theta, name="end_theta")
        stiff_threshold_variable = th.Variable(self.stiff_threshold, name="stiff_threshold")
        bend_stiffness_last_unsq_variable = th.Variable(
            self.bend_stiffness_last_unsq, name="bend_stiffness_last_unsq"
        )
        bend_stiffness_next_unsq_variable = th.Variable(
            self.bend_stiffness_next_unsq, name="bend_stiffness_next_unsq"
        )
        bend_stiffness_variable = th.Variable(self.bend_stiffness, name="bend_stiffness")
        twist_stiffness_variable = th.Variable(
            self.twist_stiffness, name="twist_stiffness"
        )

        # 4. Call the solver to optimize theta_full
        # Switch between a numpy-based solver or a pure PyTorch-based solver
        # depending on inference_1_batch
        if inference_1_batch:
            theta_full = theta_optimize_numpy(
                self.n_branch,
                cost_weight_variable,
                target_energy_variable,
                kb_variable,
                b_u_variable,
                b_v_variable,
                m_restW1_variable,
                m_restW2_variable,
                restRegionL_variable,
                inner_theta_variable,
                theta_full,
                end_theta,
                stiff_threshold_variable,
                bend_stiffness_last_unsq_variable,
                bend_stiffness_next_unsq_variable,
                bend_stiffness_variable,
                twist_stiffness_variable,
                optimization_mask,
            )
        else:
            theta_full = theta_optimize(
                self.n_branch,
                cost_weight_variable,
                target_energy_variable,
                kb_variable,
                b_u_variable,
                b_v_variable,
                m_restW1_variable,
                m_restW2_variable,
                restRegionL_variable,
                inner_theta_variable,
                theta_full,
                end_theta,
                stiff_threshold_variable,
                bend_stiffness_last_unsq_variable,
                bend_stiffness_next_unsq_variable,
                bend_stiffness_variable,
                twist_stiffness_variable,
                optimization_mask,
            )

        # 5. Compute final material frames based on optimized twist angles
        m1, m2 = self.computeMaterialFrame(theta_full, b_u, b_v)

        return theta_full, m1, m2, kb

    def computeGradientKB(self, kb, edges, restEdgeL):
        """
        Compute the gradient of the discrete curvature binormal (kb) w.r.t.
        vertex positions. Used for computing internal forces and Jacobians.

        From DER, eq. (7) and eq. (8).

        Args:
            kb (torch.Tensor): Curvature binormal, shape (batch, n_edge, 3).
            edges (torch.Tensor): Edge vectors, shape (batch, n_edge, 3).
            restEdgeL (torch.Tensor): Undeformed edge lengths, shape (batch, n_edge).

        Returns:
            o_minusGKB (torch.Tensor): Gradient of kb for the edge i-1 side.
            o_plusGKB (torch.Tensor): Gradient of kb for the edge i+1 side.
            o_eqGKB (torch.Tensor): Gradient of kb for edge i itself.
        """
        batch, n_edge = edges.size(0), edges.size(1)

        # Prepare containers for partial gradients
        o_minusGKB = torch.zeros(batch, n_edge, 3, 3).to(self.device)
        o_plusGKB = torch.zeros(batch, n_edge, 3, 3).to(self.device)

        # Precompute skew-symmetric matrices of edges
        edgeMatrix = skew_symmetric(edges)

        # scalar_factor includes normalized edge lengths in the denominator
        scalar_factor = scalar_func(edges, restEdgeL).view(batch, n_edge - 1, 1, 1)

        # Compute partial derivatives for i >= 1
        o_minusGKB[:, 1:] = (
            2.0 * edgeMatrix[:, :-1]
            + torch.einsum("bki,bkj->bkij", kb[:, 1:], edges[:, :-1])
        )
        o_plusGKB[:, 1:] = (
            2.0 * edgeMatrix[:, 1:]
            - torch.einsum("bki,bkj->bkij", kb[:, 1:], edges[:, 1:])
        )

        # Divide by scalar_factor except where zero
        epsilon = 1e-20
        o_minusGKB[:, 1:] = torch.where(
            scalar_factor != 0,
            o_minusGKB[:, 1:].clone() / (scalar_factor + epsilon),
            torch.zeros_like(o_minusGKB[:, 1:]),
        )
        o_plusGKB[:, 1:] = torch.where(
            scalar_factor != 0,
            o_plusGKB[:, 1:].clone() / (scalar_factor + epsilon),
            torch.zeros_like(o_plusGKB[:, 1:]),
        )

        # o_eqGKB = -(o_minusGKB + o_plusGKB)
        # Because the sum of the partials must be zero across all edges
        o_eqGKB = -(o_minusGKB + o_plusGKB)
        return o_minusGKB, o_plusGKB, o_eqGKB

    def computeGradientHolonomyTerms(self, kb, restEdgeL):
        """
        Compute the gradient of the "holonomy term" used in the discrete geometry
        of the rod (related to twist consistency). See DER eq. (9).

        Args:
            kb (torch.Tensor): Curvature binormal, shape (batch, n_edge, 3).
            restEdgeL (torch.Tensor): Rest lengths, shape (batch, n_edge).

        Returns:
            o_minusGH (torch.Tensor): Partial derivative for i-1 side
            o_plusGH (torch.Tensor): Partial derivative for i+1 side
            o_eqGH (torch.Tensor): Partial derivative for edge i
        """
        batch, n_edge = kb.size(0), kb.size(1)

        o_minusGH = torch.zeros(batch, n_edge, 3).to(self.device)
        o_plusGH = torch.zeros(batch, n_edge, 3).to(self.device)

        # +0.5 * kb_i / restEdgeL_i for i >= 1
        o_minusGH[:, 1:] = 0.5 * kb[:, 1:]
        o_plusGH[:, 1:] = -0.5 * kb[:, 1:]

        epsilon = 1e-20
        # Divide by restEdgeL while avoiding division by zero
        o_minusGH[:, 1:] = torch.where(
            restEdgeL[:, :-1].unsqueeze(dim=2) != 0,
            o_minusGH[:, 1:] / (restEdgeL[:, :-1].unsqueeze(dim=2) + epsilon),
            torch.zeros_like(o_minusGH[:, 1:]),
        )
        o_plusGH[:, 1:] = torch.where(
            restEdgeL[:, 1:].unsqueeze(dim=2) != 0,
            o_plusGH[:, 1:] / (restEdgeL[:, 1:].unsqueeze(dim=2) + epsilon),
            torch.zeros_like(o_plusGH[:, 1:]),
        )

        o_eqGH = -(o_minusGH + o_plusGH)
        return o_minusGH, o_plusGH, o_eqGH

    def computeGradientHolonomy(self, i, j, minusGH, plusGH, eqGH):
        """
        Helper function for distributing the gradient of holonomy terms
        (i.e. partial derivatives wrt vertex positions for the discrete rod).
        See DER eqn. (11) for the distribution logic.

        Args:
            i (int): The current edge index being processed for holonomy gradient.
            j (int): Some index in the chain of edges.
            minusGH (torch.Tensor): Holonomy partial derivative for i-1 side.
            plusGH (torch.Tensor): Holonomy partial derivative for i+1 side.
            eqGH (torch.Tensor): Holonomy partial derivative for edge i.

        Returns:
            o_Gh (torch.Tensor): Aggregated partial derivative of the holonomy.
        """
        batch = minusGH.size(0)
        o_Gh = torch.zeros(batch, 3).to(self.device)

        # For edges i-1, i, i+1
        if j >= (i - 1) and i > 1 and (i - 1) < plusGH.size(1):
            o_Gh += plusGH[:, i - 1]
        if j >= i and i < eqGH.size(1):
            o_Gh += eqGH[:, i]
        if j >= (i + 1) and (i + 1) < minusGH.size(1):
            o_Gh += minusGH[:, i + 1]

        return o_Gh

    def computeGradientCurvature(
        self, i, k, j, m_m1, m_m2, minusGKB, plusGKB, eqGKB, minusGH, plusGH, eqGH, wkj, J
    ):
        """
        Compute gradient of material frame curvature and holonomy
        for the DER's shape operator, see eq. (11).

        Args:
            i, k, j (int): Indices in eq. (11) describing edges and adjacency.
            m_m1, m_m2 (torch.Tensor): Material frames (m1, m2).
            minusGKB, plusGKB, eqGKB (torch.Tensor): Gradients of curvature binormal.
            minusGH, plusGH, eqGH (torch.Tensor): Gradients of holonomy terms.
            wkj (torch.Tensor): Current curvature in the material frame, shape (batch, 2).
            J (torch.Tensor): Some Jacobian for distribution, shape (batch, 2, 2).

        Returns:
            o_GW (torch.Tensor): Gradient of the material curvature terms.
            GH (torch.Tensor): Gradient of the holonomy.
        """
        batch = minusGH.size(0)

        # Initialize the gradient of W in material frame
        o_GW = torch.zeros(batch, 2, 3).to(self.device)

        # If k < i+2, compute partial derivative for curvature:
        # (m1, m2) cross terms with GKB, etc.
        if k < i + 2:
            # Each row in o_GW gets the direction from material frames
            o_GW[:, 0, 0] = m_m2[:, j, 0]
            o_GW[:, 0, 1] = m_m2[:, j, 1]
            o_GW[:, 0, 2] = m_m2[:, j, 2]

            o_GW[:, 1, 0] = -m_m1[:, j, 0]
            o_GW[:, 1, 1] = -m_m1[:, j, 1]
            o_GW[:, 1, 2] = -m_m1[:, j, 2]

            # Multiply by GKB part depending on k
            if k == (i - 1):
                o_GW = torch.bmm(o_GW, plusGKB[:, k])
            elif k == i:
                o_GW = torch.bmm(o_GW, eqGKB[:, k])
            elif k == i + 1:
                o_GW = torch.bmm(o_GW, minusGKB[:, k])

        # Compute gradient of holonomy
        GH = self.computeGradientHolonomy(i, j, minusGH, plusGH, eqGH)

        # Subtract the twist-part from the gradient:
        # ( GH cross wkj ) scaled by J
        o_GW -= torch.bmm(
            J,
            torch.einsum("bi,bj->bij", wkj.view(batch, 2), GH)
        )
        return o_GW, GH

    def computedEdtheta(self, m1, m2, kb, theta, JB, m_restW1, m_restW2, restRegionL):
        """
        Compute the gradient of elastic energy wrt the twist angles theta
        (i.e., dE/dtheta). This is used in optimization steps.

        Args:
            m1, m2 (torch.Tensor): Material frame axes, shape (batch, n_edge, 3).
            kb (torch.Tensor): Curvature binormal, shape (batch, n_edge, 3).
            theta (torch.Tensor): Current twist angles, shape (batch, n_edge).
            JB (torch.Tensor): The 2x2 block of the "bending" stiffness matrix
                               (clamped or thresholded).
            m_restW1 (torch.Tensor): Rest curvature about m1, shape (batch, n_edge, 2).
            m_restW2 (torch.Tensor): Rest curvature about m2, shape (batch, n_edge, 2).
            restRegionL (torch.Tensor): Voronoi region lengths for each edge
                                         (used in twist stiffness distribution).

        Returns:
            dEdtheta (torch.Tensor): Gradient of elastic energy wrt twist angles,
                                     shape (batch, n_edge).
        """
        # Initialize gradient w.r.t. theta to zero
        dEdtheta = torch.zeros_like(theta)

        if self.n_edge > 1:
            # Compute current material curvature
            # (o_W1, o_W2) relative to m1, m2
            o_W1, o_W2 = self.computeMaterialCurvature(kb, m1, m2)

            # Bending part:
            # derivative for edges [:-1]
            temp = (o_W1[:, 1:] - m_restW1[:, 1:]).unsqueeze(-1)
            JB_j = JB[:, :-1]  # JB for edges [:-1]
            JB_wij = torch.matmul(JB_j, temp).squeeze(-1)
            term1 = (o_W1[:, 1:] * JB_wij).sum(dim=-1)
            dEdtheta[:, :-1] += term1

            # derivative for edges [1:]
            temp = (o_W2[:, 1:] - m_restW2[:, 1:]).unsqueeze(-1)
            JB_j = JB[:, 1:]  # JB for edges [1:]
            JB_wij = torch.matmul(JB_j, temp).squeeze(-1)
            term2 = (o_W2[:, 1:] * JB_wij).sum(dim=-1)
            dEdtheta[:, 1:] += term2

            # Twist part:
            # clamp twist_stiffness to a minimum threshold
            twist_stiffness_clamped = torch.clamp(
                self.twist_stiffness_unsq, min=self.stiff_threshold
            )
            # difference in theta among adjacent edges
            term1 = 2.0 * twist_stiffness_clamped * (theta[:, 1:] - theta[:, :-1])

            valid_mask = restRegionL[:, 1:] != 0
            term1_result = torch.zeros_like(term1)
            term1_result[valid_mask] = term1[valid_mask] / restRegionL[:, 1:][valid_mask]
            term1 = term1_result

            # Add to dEdtheta for edges
            dEdtheta[:, 1:] += term1
            dEdtheta[:, :-1] -= term1

        return dEdtheta
