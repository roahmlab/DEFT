import torch
import torch.nn.functional as F
import glob
import os
import pytorch3d
from pathlib import Path
import pickle
import pytorch3d.transforms.rotation_conversions
from sympy import pprint
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set random seed for reproducibility.
random.seed(0)
torch.manual_seed(0)


def computeEdges(vertices, zero_mask):
    """
    Computes edge vectors (v_{i+1} - v_i) for each pair of consecutive vertices.
    If zero_mask is True at position i, the corresponding edge is set to zero.

    Args:
        vertices (torch.Tensor): Shape [batch, n_vert, 3]. The 3D coordinates of each vertex.
        zero_mask (torch.Tensor): Shape [batch, n_vert-1]. Boolean mask indicating which edges should be zeroed.

    Returns:
        edges (torch.Tensor): Shape [batch, n_vert-1, 3]. The computed edges, possibly zeroed at certain indices.
    """
    # Subtract consecutive vertices to get edges,
    # then zero them out where zero_mask is True.
    edges = torch.where(zero_mask.unsqueeze(dim=-1), 0., vertices[:, 1:] - vertices[:, :-1])
    return edges


def computeLengths(edges):
    """
    Computes:
      1. The length of each edge (EdgeL).
      2. The sum of lengths of two adjacent edges (RegionL). Often used in energy or curvature calculations.

    Args:
        edges (torch.Tensor): Shape [batch, n_edge, 3]. The edge vectors.

    Returns:
        EdgeL (torch.Tensor): Shape [batch, n_edge]. Length of each edge.
        RegionL (torch.Tensor): Shape [batch, n_edge]. Sum of lengths of two adjacent edges, with RegionL[:,0] initialized as 0 + first adjacent sum.
    """
    batch = edges.size()[0]
    # Magnitude of each edge
    EdgeL = torch.norm(edges, dim=2)
    # Initialize RegionL with zeros for the first column
    RegionL = torch.zeros(batch, 1, device=edges.device)
    # RegionL for edges i is EdgeL[i] + EdgeL[i-1], concatenated for all i
    RegionL = torch.cat((RegionL, (EdgeL[:, 1:] + EdgeL[:, :-1])), dim=1)
    return EdgeL, RegionL


def sqrt_safe(value):
    """
    Computes a 'safe' square root by clamping the input to prevent negative values.

    Args:
        value (torch.Tensor): The input tensor for which sqrt is to be taken.

    Returns:
        (torch.Tensor): The square root of the clamped input.
    """
    return torch.sqrt(torch.clamp(value, 1e-10))


def extractSinandCos(magnitude):
    """
    Extracts the sine and cosine of an angle from a given 'magnitude' using a specific function:
        sin(phi) = sqrt(magnitude / (4 + magnitude))
        cos(phi) = sqrt(4 / (4 + magnitude))
    This is typically used for rotation angles in discrete rod calculations.

    Args:
        magnitude (torch.Tensor): The input magnitude used to compute sinPhi and cosPhi.

    Returns:
        o_sinPhi (torch.Tensor): sin(phi).
        o_cosPhi (torch.Tensor): cos(phi).
    """
    constant = 4.0
    o_sinPhi = sqrt_safe(magnitude / (constant + magnitude))
    o_cosPhi = sqrt_safe(constant / (constant + magnitude))
    return o_sinPhi, o_cosPhi


def computeKB(edges, m_restEdgeL):
    """
    Computes the discrete curvature binormal k_b for each edge using the cross product approach from DER (Discrete Elastic Rods).
    See DER paper eq. 1 for reference.

    Args:
        edges (torch.Tensor): Shape [batch, n_edge, 3]. Edge vectors for consecutive vertices.
        m_restEdgeL (torch.Tensor): Shape [batch, n_edge]. Rest lengths for each edge.

    Returns:
        o_kb (torch.Tensor): Shape [batch, n_edge, 3]. The computed discrete curvature binormal for each edge.
    """
    o_kb = torch.zeros_like(edges)
    zero_mask = m_restEdgeL[:, 1:] == 0
    epsilon = 1e-20
    # Inverse length factor ensures no division by zero or near-zero.
    inv_length = torch.where(
        zero_mask,
        0.,
        1 / (m_restEdgeL[:, :-1] * m_restEdgeL[:, 1:] + (edges[:, :-1] * edges[:, 1:]).sum(dim=-1) + epsilon)
    )
    # Cross product of adjacent edges, scaled by the factor. Clamped to avoid large values.
    o_kb[:, 1:] = torch.clamp(
        2 * torch.linalg.cross(edges[:, :-1], edges[:, 1:]) * inv_length.unsqueeze(dim=-1),
        min=-500, max=500
    )
    return o_kb


def quaternion_q(theta, kb):
    """
    Forms a partial quaternion by concatenating twist angles theta and curvature binormal kb along dim=2.

    Args:
        theta (torch.Tensor): Shape [batch, n_edge, 1]. Twist angles.
        kb (torch.Tensor): Shape [batch, n_edge, 3]. Curvature binormal.

    Returns:
        (torch.Tensor): Shape [batch, n_edge, 4]. The combined quaternion-like [theta, kb].
    """
    return torch.cat((theta.unsqueeze(dim=2), kb), dim=2)


def quaternion_p(theta, kb):
    """
    Forms a partial quaternion by concatenating twist angles theta and curvature binormal kb along dim=1.
    (Slightly different shape arrangement than quaternion_q.)

    Args:
        theta (torch.Tensor): Shape [batch, 1].
        kb (torch.Tensor): Shape [batch, 3].

    Returns:
        (torch.Tensor): Shape [batch, 4]. [theta, kb].
    """
    return torch.cat((theta, kb), dim=1)


def computeW(kb, m1, m2):
    """
    Computes the local 2D curvature components by projecting the discrete curvature binormal onto the material frame.

    Args:
        kb (torch.Tensor): [batch, n_edge, 3]. The curvature binormal.
        m1, m2 (torch.Tensor): [batch, n_edge, 3]. The two perpendicular axes of the material frame.

    Returns:
        o_wij (torch.Tensor): [batch, n_edge, 2]. The curvature in local (m1, m2) coordinates.
    """
    o_wij = torch.cat((
        (kb * m2).sum(dim=2).unsqueeze(dim=2),
        -(kb * m1).sum(dim=2).unsqueeze(dim=2)
    ), dim=2)
    return o_wij


def skew_symmetric(edges):
    """
    Creates a batch of skew-symmetric matrices for each 3D vector in edges.
    If v = (x, y, z), the corresponding skew-symmetric matrix is:
        [ 0   -z   y ]
        [ z    0  -x ]
        [-y    x   0 ]

    Args:
        edges (torch.Tensor): Shape [batch, n_edge, 3]. Each row is a vector.

    Returns:
        matrix (torch.Tensor): Shape [batch, n_edge, 3, 3]. The skew-symmetric matrices.
    """
    batch = edges.size()[0]
    n_edges = edges.size()[1]
    matrix = torch.zeros(batch, n_edges, 3, 3, dtype=edges.dtype, device=edges.device)
    matrix[:, :, 0, 1] = -edges[:, :, 2]
    matrix[:, :, 0, 2] = edges[:, :, 1]
    matrix[:, :, 1, 0] = edges[:, :, 2]
    matrix[:, :, 1, 2] = -edges[:, :, 0]
    matrix[:, :, 2, 0] = -edges[:, :, 1]
    matrix[:, :, 2, 1] = edges[:, :, 0]
    return matrix


def scalar_func(edges, restEdgeL):
    """
    Computes a scalar function often used in calculations of the discrete rod (like <e_{i}, e_{i+1}>).

    Args:
        edges (torch.Tensor): [batch, n_edge, 3].
        restEdgeL (torch.Tensor): [batch, n_edge]. The rest lengths of edges.

    Returns:
        (torch.Tensor): Scalar result for each pair of adjacent edges in the batch.
    """
    return restEdgeL[:, :-1] * restEdgeL[:, 1:] + (edges[:, :-1] * edges[:, 1:]).sum(dim=2)


def rotation_matrix(theta):
    """
    Constructs 2D rotation matrices for each element in theta.

    Args:
        theta (torch.Tensor): [batch]. Angles for each batch element.

    Returns:
        transform_basis (torch.Tensor): [batch, 2, 2]. The rotation matrices.
    """
    batch = theta.size()[0]
    rot_sin = torch.sin(theta)
    rot_cos = torch.cos(theta)
    transform_basis = torch.zeros(batch, 2, 2)
    transform_basis[:, 0, 0] = rot_cos
    transform_basis[:, 0, 1] = -rot_sin
    transform_basis[:, 1, 0] = rot_sin
    transform_basis[:, 1, 1] = rot_cos
    return transform_basis


def quaternion_multiply(q1, q2):
    """
    Multiplies two quaternions q1 and q2. Each has shape (batch_size, 4) -> [w, x, y, z].

    Args:
        q1 (torch.Tensor): [batch, 4].
        q2 (torch.Tensor): [batch, 4].

    Returns:
        (torch.Tensor): [batch, 4]. The product of the two quaternions.
    """
    w1 = q1[:, 0]
    x1 = q1[:, 1]
    y1 = q1[:, 2]
    z1 = q1[:, 3]
    w2 = q2[:, 0]
    x2 = q2[:, 1]
    y2 = q2[:, 2]
    z2 = q2[:, 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack((w, x, y, z), -1)


def quaternion_conjugate(q):
    """
    Computes the conjugate of a quaternion q = [w, x, y, z].
    The conjugate is q* = [w, -x, -y, -z].

    Args:
        q (torch.Tensor): (..., 4). A quaternion or batch of quaternions.

    Returns:
        q_conj (torch.Tensor): (..., 4). The conjugate of the input quaternion.
    """
    q_conj = q.clone()  # copy
    q_conj[..., 1:] *= -1  # negate the vector part
    return q_conj


def quaternion_rotation(o_u, edges, q, i):
    """
    Applies quaternion-based rotation to update the frame vectors (u, v).

    Args:
        o_u (torch.Tensor): [batch, n_edge, 3]. The initial 'u' direction (Bishop frame).
        edges (torch.Tensor): [batch, n_edge, 3]. The edge vectors.
        q (torch.Tensor): [batch, n_edge, 4]. The quaternion for each edge.
        i (int): The index of the current edge.

    Returns:
        (u, v) (torch.Tensor, torch.Tensor): The updated frame vectors (u, v) after rotation.
    """
    batch = o_u.size()[0]
    # Build partial quaternion for p
    p = quaternion_p(torch.zeros(batch, 1).to(o_u.device), o_u[:, i - 1])
    # Rotate p by q[i]
    quat_p = quaternion_multiply(quaternion_multiply(q[:, i], p), quaternion_conjugate(q[:, i]))
    # Normalize to get updated u
    u = F.normalize(quat_p[:, 1:4], dim=1)
    # Compute v as cross of edges and u
    v = F.normalize(torch.cross(edges[:, i], u, dim=-1), dim=1)
    return u.unsqueeze(dim=1), v.unsqueeze(dim=1)


def quaternion_rotation_parallel(cosPhi, sinPhi, axis, io_u):
    """
    Applies a rotation about 'axis' using angle components cosPhi, sinPhi in quaternion form to rotate io_u.

    Args:
        cosPhi (torch.Tensor): Shape [batch].
        sinPhi (torch.Tensor): Shape [batch].
        axis (torch.Tensor): Shape [batch, 3]. The rotation axis.
        io_u (torch.Tensor): Shape [batch, 3]. The vector to rotate.

    Returns:
        io_u (torch.Tensor): Shape [batch, 3]. The rotated vector.
    """
    batch = cosPhi.size()[0]
    # Form quaternion q
    q = quaternion_p(
        cosPhi.view(-1, 1),
        sinPhi.view(-1, 1) * F.normalize(axis, dim=1)
    )
    # Build p
    p = quaternion_p(torch.zeros(batch, 1).to(io_u.device), io_u)
    # Apply quaternion rotation
    quat_p = quaternion_multiply(quaternion_multiply(q, p), quaternion_conjugate(q))
    io_u = F.normalize(quat_p[:, 1:4], dim=1)
    return io_u


def compute_u0(e0, init_direct):
    """
    Initializes the first Bishop frame vector u0 for the first edge.
    Typically, N_0 = e0 x init_direct, then we define m_u0 = normalize(N_0 x e0).

    Args:
        e0 (torch.Tensor): [batch, 3]. The first edge vector.
        init_direct (torch.Tensor): [1, 3]. A reference direction to help define the initial frame.

    Returns:
        m_u0 (torch.Tensor): [batch, 3]. The initialized first 'u' direction.
    """
    batch = e0.size()[0]
    N_0 = torch.cross(e0, init_direct.view(batch, -1))
    m_u0 = F.normalize(torch.cross(N_0, e0), dim=1)
    return m_u0


def parallelTransportFrame(e0, e1, io_u):
    """
    Parallel transports the frame vector io_u from edge e0 to edge e1, accounting for holonomy in discrete rods.
    A quaternion-based rotation is applied if the rotation angle is not near zero.

    Args:
        e0, e1 (torch.Tensor): [batch, 3]. Consecutive edges.
        io_u (torch.Tensor): [batch, 3]. The frame vector 'u' to be updated.

    Returns:
        io_u (torch.Tensor): The updated frame vector after parallel transport.
    """
    batch = e0.size()[0]
    err = torch.tensor(1e-6).to(io_u.device)

    # Define axis for rotation:
    axis = 2 * torch.cross(e0, e1, dim=1) / (e0.norm(dim=1) * e1.norm(dim=1) + (e0 * e1).sum(dim=1)).unsqueeze(dim=1)
    # Magnitude of the axis (squared)
    magnitude = (axis * axis).sum(dim=1)
    sinPhi, cosPhi = extractSinandCos(magnitude)

    # If rotation is almost zero, we apply a simpler re-orthogonalization approach. Otherwise, use quaternion rotation.
    io_u = torch.where(
        torch.ones(batch, 1).to(io_u.device) - cosPhi.unsqueeze(dim=1) <= err * torch.ones(batch, 1).to(io_u.device),
        F.normalize(torch.cross(torch.cross(e1, io_u, dim=1), e1), dim=1),
        quaternion_rotation_parallel(cosPhi, sinPhi, axis, io_u)
    )
    return io_u


def DEFT_initialization(parent_vertices, child1_vertices, child2_vertices, n_branch, p_n_vert, cs_n_vert,
                        rigid_body_coupling_index, parent_mass_scale, parent_moment_scale, children_moment_scale,
                        children_mass_scale, moment_ratio):
    """
    Initializes mass, moment of inertia (MOI), and orientation data for a branched discrete elastic rod system
    with 1 parent rod and (n_branch-1) child rods.

    Args:
        parent_vertices (torch.Tensor): [1, p_n_vert, 3]. Coordinates of the parent rod's vertices.
        child1_vertices, child2_vertices (torch.Tensor): [1, c_n_vert, 3]. Coordinates of child rods.
        n_branch (int): Total number of rods (1 parent + children).
        p_n_vert (int): Number of parent vertices.
        cs_n_vert (List[int]): Number of child vertices for each branch.
        rigid_body_coupling_index (List[int]): Indices in the parent rod where child rods attach.
        parent_mass_scale, parent_moment_scale: Scaling factors for parent rod's mass and MOI.
        children_moment_scale, children_mass_scale: Scaling factors for children rods' MOI and mass.
        moment_ratio (float): Radius scale ratio relative to parent edge length for MOI calculations.

    Returns:
        Various tensors related to mass, MOI, orientations, and nominal length for the entire branched system.
    """
    # Compute nominal length and radius for parent rod
    parent_nominal_length = torch.norm(parent_vertices[:, 1:] - parent_vertices[:, :-1], dim=-1)[0]
    parent_nominal_radius = parent_nominal_length * moment_ratio

    # Compute mass distribution for the parent rod
    p_DLO_mass = torch.zeros(p_n_vert)
    # Each edge length is half attributed to the two vertices it connects
    p_DLO_mass[0:p_n_vert - 1] += parent_nominal_length / 2.
    p_DLO_mass[1:p_n_vert] += parent_nominal_length / 2.
    p_DLO_mass = p_DLO_mass * parent_mass_scale

    # Prepare MOI placeholders for the parent (at each coupling index), and define rod orientation.
    parent_MOI = torch.zeros(len(rigid_body_coupling_index) * 2, 3)
    parent_rod_axis_angle = torch.zeros(1, 3)
    parent_rod_orientation = pytorch3d.transforms.rotation_conversions.axis_angle_to_quaternion(parent_rod_axis_angle) \
        .unsqueeze(dim=0).repeat(1, p_n_vert - 1, 1)

    # Prepare placeholders for children
    children_vertices = torch.zeros(1, len(cs_n_vert), max(cs_n_vert), 3)
    children_mass = torch.zeros(len(cs_n_vert), max(cs_n_vert))
    children_MOI = torch.zeros(len(rigid_body_coupling_index), 3)
    children_nominal_length = torch.zeros(len(cs_n_vert), max(cs_n_vert) - 1)
    children_nominal_radius = torch.zeros(len(cs_n_vert), max(cs_n_vert) - 1)
    child_rod_axis_angle = torch.zeros(1, 3)
    children_rod_orientation = pytorch3d.transforms.rotation_conversions.axis_angle_to_quaternion(child_rod_axis_angle) \
        .unsqueeze(dim=0).repeat(1, len(rigid_body_coupling_index), 1)

    # Loop through rigid body coupling points to compute MOI for parent rods at those couplings,
    # and build child rods.
    for i in range(len(rigid_body_coupling_index)):
        # MOI for parent rod at rigid_body_coupling_index[i]-1
        I_x_parent = 1 / 12 * parent_nominal_length[rigid_body_coupling_index[i] - 1] ** 2 \
                     + 1 / 4 * parent_nominal_radius[rigid_body_coupling_index[i] - 1] ** 2
        I_y_parent = I_x_parent
        I_z_parent = 1 / 2 * parent_nominal_radius[rigid_body_coupling_index[i] - 1] ** 2
        parent_MOI[2 * i, 0] = I_x_parent * parent_moment_scale
        parent_MOI[2 * i, 1] = I_y_parent * parent_moment_scale
        parent_MOI[2 * i, 2] = I_z_parent * parent_moment_scale

        # Another MOI for the second index usage
        I_x_parent = 1 / 12 * parent_nominal_length[rigid_body_coupling_index[0]] ** 2 \
                     + 1 / 4 * parent_nominal_radius[rigid_body_coupling_index[0]] ** 2
        I_y_parent = I_x_parent
        I_z_parent = 1 / 2 * parent_nominal_radius[rigid_body_coupling_index[0]] ** 2
        parent_MOI[2 * i + 1, 0] = I_x_parent * parent_moment_scale
        parent_MOI[2 * i + 1, 1] = I_y_parent * parent_moment_scale
        parent_MOI[2 * i + 1, 2] = I_z_parent * parent_moment_scale

        c_n_vert = cs_n_vert[i]

        # Merge child rod with the parent rod at the coupling index
        if i == 0:
            # Insert the parent's coupling point + child1 vertices
            children_vertices[:, i, :c_n_vert] = torch.cat(
                (parent_vertices[:, rigid_body_coupling_index[i]].unsqueeze(dim=1), child1_vertices), dim=1
            )
        if i == 1:
            children_vertices[:, i, :c_n_vert] = torch.cat(
                (parent_vertices[:, rigid_body_coupling_index[i]].unsqueeze(dim=1), child2_vertices), dim=1
            )

        # Compute length and radius for children
        child_nominal_length = \
        torch.norm(children_vertices[:, i, 1:c_n_vert] - children_vertices[:, i, :c_n_vert - 1], dim=-1)[0]
        child_nominal_radius = child_nominal_length * moment_ratio
        # Mass distribution for the child rod
        child_mass = torch.zeros(c_n_vert)
        child_mass[0:c_n_vert - 1] += child_nominal_length / 2.
        child_mass[1:c_n_vert] += child_nominal_length / 2.
        child_mass = child_mass * children_mass_scale[i]
        children_mass[i, :c_n_vert] = child_mass

        # Save child nominal lengths/radii
        children_nominal_length[i, :c_n_vert - 1] = child_nominal_length
        children_nominal_radius[i, :c_n_vert - 1] = child_nominal_radius
        I_y_child = 1 / 12 * child_nominal_length[0] ** 2 + 1 / 4 * child_nominal_radius[0] ** 2
        I_x_child = I_y_parent
        I_z_child = 1 / 2 * child_nominal_radius[0] ** 2
        children_MOI[i, 0] = I_x_child * children_moment_scale[i]
        children_MOI[i, 1] = I_y_child * children_moment_scale[i]
        children_MOI[i, 2] = I_z_child * children_moment_scale[i]

    # Collect mass data for the branched rods (b_DLO refers to 'branched DLO').
    b_DLO_mass = torch.zeros(n_branch, p_n_vert)
    b_DLO_mass[0] = p_DLO_mass
    b_DLO_mass[1:, :children_mass.size()[1]] = children_mass

    # Collect nominal length data
    b_nominal_length = torch.zeros(n_branch, p_n_vert - 1)
    b_nominal_length[0] = parent_nominal_length
    b_nominal_length[1:, :children_nominal_length.size()[1]] = children_nominal_length

    return b_DLO_mass, parent_MOI, children_MOI, parent_rod_orientation, children_rod_orientation, b_nominal_length


def construct_b_DLOs(batch, rigid_body_coupling_index, p_n_vert, cs_n_vert, n_branch,
                     previous_parent_vertices, parent_vertices,
                     previous_child1_vertices, child1_vertices,
                     previous_child2_vertices, child2_vertices):
    """
    Constructs a batched representation of the full branched DLO's vertices:
    [batch, n_branch, p_n_vert, 3].

    We integrate the parent rod and the child rods by merging the child's start vertex with
    the parent's coupling vertex.

    Args:
        batch (int): Batch size.
        rigid_body_coupling_index (List[int]): The indices on the parent rod where each child rod connects.
        p_n_vert (int): Number of parent rod vertices.
        cs_n_vert (List[int]): Number of child vertices for each branch.
        n_branch (int): Total branches (1 parent + others children).
        previous_parent_vertices, parent_vertices, previous_child1_vertices, child1_vertices, ...
            (torch.Tensor): The 3D coordinates for each rod's vertices, possibly from two timesteps (previous & current).

    Returns:
        b_DLOs_vertices, previous_b_DLOs_vertices (torch.Tensor):
            Shape [batch, n_branch, p_n_vert, 3]. The constructed branched rods for current and previous timesteps.
    """
    b_DLOs_vertices = torch.zeros(batch, n_branch, p_n_vert, 3)
    previous_b_DLOs_vertices = torch.zeros(batch, n_branch, p_n_vert, 3)

    # The first branch is the parent itself
    b_DLOs_vertices[:, 0] = parent_vertices
    previous_b_DLOs_vertices[:, 0] = previous_parent_vertices

    # Build the children rods
    for i in range(n_branch - 1):
        c_n_vert = cs_n_vert[i]
        if i == 0:
            b_DLOs_vertices[:, i + 1, :c_n_vert] = torch.cat(
                (parent_vertices[:, rigid_body_coupling_index[i]].unsqueeze(dim=1), child1_vertices), dim=1
            )
            previous_b_DLOs_vertices[:, i + 1, :c_n_vert] = torch.cat(
                (previous_parent_vertices[:, rigid_body_coupling_index[i]].unsqueeze(dim=1), previous_child1_vertices),
                dim=1
            )
        if i == 1:
            b_DLOs_vertices[:, i + 1, :c_n_vert] = torch.cat(
                (parent_vertices[:, rigid_body_coupling_index[i]].unsqueeze(dim=1), child2_vertices), dim=1
            )
            previous_b_DLOs_vertices[:, i + 1, :c_n_vert] = torch.cat(
                (previous_parent_vertices[:, rigid_body_coupling_index[i]].unsqueeze(dim=1), previous_child2_vertices),
                dim=1
            )
    return b_DLOs_vertices, previous_b_DLOs_vertices


def construct_BDLOs_data(total_length, rigid_body_coupling_index, p_n_vert, cs_n_vert, n_branch,
                         parent_vertices, child1_vertices, child2_vertices):
    """
    Constructs a timeline of branched DLO vertices for a sequence of length total_length.

    Args:
        total_length (int): The total number of timesteps in the data.
        rigid_body_coupling_index (List[int]): Indices on the parent rod for branching.
        p_n_vert (int): Number of vertices in the parent rod.
        cs_n_vert (List[int]): Number of vertices in each child rod.
        n_branch (int): Total branches (1 parent + children).
        parent_vertices, child1_vertices, child2_vertices (torch.Tensor): [total_length, n_*_vertices, 3].

    Returns:
        b_DLOs_vertices (torch.Tensor): [total_length, n_branch, p_n_vert, 3]. The merged rod system over time.
    """
    b_DLOs_vertices = torch.zeros(total_length, n_branch, p_n_vert, 3)
    # First branch is the parent
    b_DLOs_vertices[:, 0] = parent_vertices
    # Build the child rods
    for i in range(n_branch - 1):
        c_n_vert = cs_n_vert[i]
        if i == 0:
            b_DLOs_vertices[:, i + 1, :c_n_vert] = torch.cat(
                (parent_vertices[:, rigid_body_coupling_index[i]].unsqueeze(dim=1), child1_vertices), dim=1
            )
        if i == 1:
            b_DLOs_vertices[:, i + 1, :c_n_vert] = torch.cat(
                (parent_vertices[:, rigid_body_coupling_index[i]].unsqueeze(dim=1), child2_vertices), dim=1
            )
    return b_DLOs_vertices


def label_tensor(tensor):
    """
    Interprets special clamp values within a tensor (0, 1), (-2, -1) as boundary conditions.
    Also handles extremely large clamp marks (1e10).

    Returns a list or tuple of 'labels' extracted. E.g., if (0,1) is present, add 0 to output.
    If (-2,-1) is present, add -1 to output. Otherwise, includes the leftover elements.
    """
    clamp_mark = 1e10
    tensor = tensor.float()

    # If entire tensor is the clamp_mark, return empty
    if torch.all(tensor == clamp_mark):
        return ()

    # If fewer than 2 elements, just return them
    if tensor.numel() < 2:
        other_numbers = tensor if tensor.numel() > 0 else torch.tensor([])
        output = []
        if other_numbers.numel() > 0:
            output.extend(other_numbers.tolist())
        return tuple(output)

    # Create pairs
    pairs = torch.stack((tensor[:-1], tensor[1:]), dim=1)

    # Define patterns
    pair_0_1 = torch.tensor([0, 1])
    pair_neg2_neg1 = torch.tensor([-2, -1])

    pair_mask = torch.zeros(tensor.size(0), dtype=torch.bool)

    is_0_1 = torch.all(pairs == pair_0_1, dim=1)
    indices_0_1 = torch.nonzero(is_0_1, as_tuple=False).flatten()

    is_neg2_neg1 = torch.all(pairs == pair_neg2_neg1, dim=1)
    indices_neg2_neg1 = torch.nonzero(is_neg2_neg1, as_tuple=False).flatten()

    for idx in indices_0_1:
        pair_mask[idx] = True
        pair_mask[idx + 1] = True
    for idx in indices_neg2_neg1:
        pair_mask[idx] = True
        pair_mask[idx + 1] = True

    other_numbers = tensor[~pair_mask]
    output = []

    if indices_0_1.numel() > 0:
        output.append(0)

    if other_numbers.numel() > 0:
        output.extend(other_numbers.tolist())

    if indices_neg2_neg1.numel() > 0:
        output.append(-1)

    return output


def clamp_init(batch, parent_clamped_selection, child1_clamped_selection, child2_clamped_selection,
               n_branch, p_n_vert,
               clamp_parent, clamp_child1, clamp_child2,
               parent_vertices_traj, child1_vertices_traj, child2_vertices_traj):
    """
    Initializes clamp indices for the rods. A clamp means a vertex whose twist angle is fixed.

    Args:
        parent_clamped_selection, child1_clamped_selection, child2_clamped_selection:
            The vertices chosen to clamp (can include special values like -1).
        clamp_parent, clamp_child1, clamp_child2 (bool): Whether each rod is clamped.
        parent_vertices_traj, child*_vertices_traj (torch.Tensor): Vertex trajectories for each rod.

    Returns:
        clamped_index (torch.Tensor): [n_branch, p_n_vert]. Mark =1 where clamp is applied.
        parent_fix_point, child1_fix_point, child2_fix_point (torch.Tensor or None):
            The positions of the clamped vertices repeated along batch dimension.
        parent_theta_clamp, child1_theta_clamp, child2_theta_clamp (Union[torch.Tensor, None]):
            The final clamp 'labels' derived from label_tensor or None if not clamped.
    """
    clamped_index = torch.zeros(n_branch, p_n_vert)

    if clamp_parent:
        clamped_index[0, parent_clamped_selection] = torch.tensor((1.))
        # Repeat the fix point for the entire batch
        parent_fix_point = parent_vertices_traj[:, parent_clamped_selection].unsqueeze(dim=0).repeat(batch, 1, 1, 1)
        parent_theta_clamp = label_tensor(parent_clamped_selection)
    else:
        parent_fix_point = None
        parent_theta_clamp = None

    if clamp_child1:
        clamped_index[1, child1_clamped_selection] = torch.tensor((1.))
        child1_fix_point = child1_vertices_traj[:, child1_clamped_selection - 1].unsqueeze(dim=0).repeat(batch, 1, 1, 1)
        if child1_clamped_selection == -1:
            child1_theta_clamp = torch.tensor((-1))
        else:
            child1_theta_clamp = child1_clamped_selection
    else:
        child1_fix_point = None
        child1_theta_clamp = None

    if clamp_child2:
        clamped_index[2, child2_clamped_selection] = torch.tensor((1.))
        child2_fix_point = child2_vertices_traj[:, child2_clamped_selection - 1].unsqueeze(dim=0).repeat(batch, 1, 1, 1)
        if child2_clamped_selection == -1:
            child2_theta_clamp = torch.tensor((-1))
        else:
            child2_theta_clamp = child2_clamped_selection
    else:
        child2_fix_point = None
        child2_theta_clamp = None

    return (clamped_index, parent_fix_point, child1_fix_point, child2_fix_point,
            parent_theta_clamp, child1_theta_clamp, child2_theta_clamp)


def clamp_index(batch, parent_clamped_selection, child1_clamped_selection, child2_clamped_selection,
                n_branch, p_n_vert, clamp_parent, clamp_child1, clamp_child2):
    """
    Similar to clamp_init but only returns the clamp indices and theta clamp values without fix points.

    Args:
        parent_clamped_selection, child1_clamped_selection, child2_clamped_selection: Indices to clamp.
        n_branch, p_n_vert (int): Number of rods and parent vertices.
        clamp_parent, clamp_child1, clamp_child2 (bool): Flags for each rod.

    Returns:
        clamped_index (torch.Tensor): [n_branch, p_n_vert]. Mark =1 where clamp is applied.
        parent_theta_clamp, child1_theta_clamp, child2_theta_clamp (Union[torch.Tensor, None]):
            The clamp selection or None for each rod.
    """
    clamped_index = torch.zeros(n_branch, p_n_vert)

    if clamp_parent:
        clamped_index[0, parent_clamped_selection] = torch.tensor((1.))
        parent_theta_clamp = label_tensor(parent_clamped_selection)
    else:
        parent_theta_clamp = None

    if clamp_child1:
        clamped_index[1, child1_clamped_selection] = torch.tensor((1.))
        if child1_clamped_selection == -1:
            child1_theta_clamp = torch.tensor((-1))
        else:
            child1_theta_clamp = child1_clamped_selection
    else:
        child1_theta_clamp = None

    if clamp_child2:
        clamped_index[2, child2_clamped_selection] = torch.tensor((1.))
        if child2_clamped_selection == -1:
            child2_theta_clamp = torch.tensor((-1))
        else:
            child2_theta_clamp = child2_clamped_selection
    else:
        child2_theta_clamp = None

    return clamped_index, parent_theta_clamp, child1_theta_clamp, child2_theta_clamp


def index_init(rigid_body_coupling_index, n_branch):
    """
    Generates index arrays for MOI-based indexing in parent rods.

    Args:
        rigid_body_coupling_index (List[int]): Indices to be used for coupling or MOI references.
        n_branch (int): Number of branches (parent + children).

    Returns:
        index_selection1, index_selection2, parent_MOI_index1, parent_MOI_index2 (torch.Tensor):
            Various index selections computed from the coupling indices.
    """
    # Adjust the index for parent rod
    index_selection1 = torch.tensor(rigid_body_coupling_index) - 1
    start = 0
    end = (n_branch - 1) * 2 - 2
    num_points = len(rigid_body_coupling_index)
    parent_MOI_index1 = torch.linspace(start, end, num_points).to(torch.int)

    index_selection2 = torch.tensor(rigid_body_coupling_index)
    start = 1
    end = (n_branch - 1) * 2 - 1
    parent_MOI_index2 = torch.linspace(start, end, num_points).to(torch.int)
    return index_selection1, index_selection2, parent_MOI_index1, parent_MOI_index2


def visualize_tensors_3d_in_same_plot_no_zeros(
        parent_clamped_selection, tensor_1, tensor_2, ith, test_data_index,
        clamp_parent, clamp_child1, clamp_child2,
        parent_fix_point_flat2, child1_fix_point_flat, child2_fix_point_flat,
        additional_tensor_1, additional_tensor_2, additional_tensor_3, i_eval_batch, vis_type
):
    """
    Plots multiple 3D tensor datasets in a single figure with two 3D subplots (two different view angles).
    Zeros are filtered out (ignored), and certain indices from parent_clamped_selection are highlighted.

    Args:
        parent_clamped_selection (torch.Tensor): Indices in tensor_1 (the main rod) that are clamped or special.
        tensor_1 (torch.Tensor): A main set of 3D points (e.g., predicted rod).
        tensor_2 (torch.Tensor): Possibly multiple sets of 3D points to overlay (e.g., other rods).
        additional_tensor_1, additional_tensor_2, additional_tensor_3: Extra sets of points to overlay (e.g. ground truth).
        clamp_parent, clamp_child1, clamp_child2 (bool): Flags that decide if fix points are plotted.
        parent_fix_point_flat2, child1_fix_point_flat, child2_fix_point_flat (torch.Tensor or None):
            The fix/clamp points to highlight if clamps are used.
        i_eval_batch (int): ID for saving directory path.
        vis_type (str): Directory subfolder for saving the results.

    Returns:
        None. Saves a .png file of the 3D plot.
    """

    def filter_non_zero_points(tensor):
        # Filters out rows that are all zeros in the last dimension
        non_zero_mask = torch.any(tensor != 0, dim=-1)
        return tensor[non_zero_mask]

    # Create 2 subplots (two views) in 3D
    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title('View 1')
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title('View 2')

    # Core plotting function
    def plot_tensors(ax):
        N = tensor_1.shape[0]

        # Convert negative indices to positive with modulo, then remove duplicates
        selected_indices = parent_clamped_selection % N
        selected_indices = np.unique(selected_indices)
        total_indices = np.arange(N)
        non_selected_indices = np.setdiff1d(total_indices, selected_indices)

        # Identify negative indices (originally <0)
        negative_indices = parent_clamped_selection[parent_clamped_selection < 0] % N
        negative_indices = np.unique(negative_indices)
        # Positive indices
        positive_indices = np.setdiff1d(selected_indices, negative_indices)

        # Plot additional_tensor_1 (likely ground truth)
        additional_tensor_1_filtered = filter_non_zero_points(additional_tensor_1)
        if additional_tensor_1_filtered.size(0) > 0:
            ax.scatter(
                additional_tensor_1_filtered[:, 0].detach().cpu().numpy(),
                additional_tensor_1_filtered[:, 1].detach().cpu().numpy(),
                additional_tensor_1_filtered[:, 2].detach().cpu().numpy(),
                c='black', marker='o', s=30, label='Ground Truth'
            )
        # Plot additional_tensor_2
        additional_tensor_2_filtered = filter_non_zero_points(additional_tensor_2)
        if additional_tensor_2_filtered.size(0) > 0:
            ax.scatter(
                additional_tensor_2_filtered[:, 0].detach().cpu().numpy(),
                additional_tensor_2_filtered[:, 1].detach().cpu().numpy(),
                additional_tensor_2_filtered[:, 2].detach().cpu().numpy(),
                c='black', marker='o', alpha=1.0, s=30
            )
        # Plot additional_tensor_3
        additional_tensor_3_filtered = filter_non_zero_points(additional_tensor_3)
        if additional_tensor_3_filtered.size(0) > 0:
            ax.scatter(
                additional_tensor_3_filtered[:, 0].detach().cpu().numpy(),
                additional_tensor_3_filtered[:, 1].detach().cpu().numpy(),
                additional_tensor_3_filtered[:, 2].detach().cpu().numpy(),
                c='black', marker='o', alpha=1.0, s=30
            )

        # Plot non-selected points from tensor_1
        tensor_1_non_selected = tensor_1[non_selected_indices]
        tensor_1_non_selected = filter_non_zero_points(tensor_1_non_selected)
        if tensor_1_non_selected.size(0) > 0:
            ax.scatter(
                tensor_1_non_selected[:, 0].detach().cpu().numpy(),
                tensor_1_non_selected[:, 1].detach().cpu().numpy(),
                tensor_1_non_selected[:, 2].detach().cpu().numpy(),
                c='blue', marker='o', alpha=1.0, s=30, label='Prediction'
            )

        # Plot selected points with positive indices
        tensor_1_positive = tensor_1[positive_indices]
        tensor_1_positive = filter_non_zero_points(tensor_1_positive)
        if tensor_1_positive.size(0) > 0:
            ax.scatter(
                tensor_1_positive[:, 0].detach().cpu().numpy(),
                tensor_1_positive[:, 1].detach().cpu().numpy(),
                tensor_1_positive[:, 2].detach().cpu().numpy(),
                c='red', marker='o', alpha=1.0, s=40, label='Clamped Points'
            )

        # Plot selected points with negative indices
        tensor_1_negative = tensor_1[negative_indices]
        tensor_1_negative = filter_non_zero_points(tensor_1_negative)
        if tensor_1_negative.size(0) > 0:
            ax.scatter(
                tensor_1_negative[:, 0].detach().cpu().numpy(),
                tensor_1_negative[:, 1].detach().cpu().numpy(),
                tensor_1_negative[:, 2].detach().cpu().numpy(),
                c='red', marker='o', alpha=1.0, s=40
            )

        # Plot each row in tensor_2
        for i in range(tensor_2.shape[0]):
            filtered_tensor_2 = filter_non_zero_points(tensor_2[i])
            if filtered_tensor_2.size(0) > 0:
                ax.scatter(
                    filtered_tensor_2[:, 0].detach().cpu().numpy(),
                    filtered_tensor_2[:, 1].detach().cpu().numpy(),
                    filtered_tensor_2[:, 2].detach().cpu().numpy(),
                    c='blue', alpha=1.0, s=30, marker='o'
                )

        # If child rods are clamped, plot the fix points
        if clamp_child1 and child1_fix_point_flat is not None:
            points = child1_fix_point_flat[0]
            filtered_points = filter_non_zero_points(points)
            if filtered_points.size(0) > 0:
                ax.scatter(
                    filtered_points[:, 0].detach().cpu().numpy(),
                    filtered_points[:, 1].detach().cpu().numpy(),
                    filtered_points[:, 2].detach().cpu().numpy(),
                    c='red', s=40, marker='o', label='Child1 Fix Points'
                )
        if clamp_child2 and child2_fix_point_flat is not None:
            points = child2_fix_point_flat[0]
            filtered_points = filter_non_zero_points(points)
            if filtered_points.size(0) > 0:
                ax.scatter(
                    filtered_points[:, 0].detach().cpu().numpy(),
                    filtered_points[:, 1].detach().cpu().numpy(),
                    filtered_points[:, 2].detach().cpu().numpy(),
                    c='red', s=40, marker='o', label='Child2 Fix Points'
                )

        # Configure plot
        ax.set_xlim(-0.5, 1.0)
        ax.set_ylim(-0.5, 1.0)
        ax.set_zlim(-0.25, 1.25)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()

    # Plot on both subplots
    plot_tensors(ax1)
    plot_tensors(ax2)

    # Adjust viewpoints
    ax1.view_init(elev=0, azim=90)
    ax2.view_init(elev=30, azim=-45)

    # Save the figure
    dir_path = Path(f"sanity_check/{vis_type}/{i_eval_batch}")
    dir_path.mkdir(parents=True, exist_ok=True)

    plt.savefig(f"sanity_check/{vis_type}/{i_eval_batch}/{ith}.png")
    plt.close(fig)


def save_pickle(data, myfile):
    """
    Saves data to a pickle file.

    Args:
        data: The Python object to save.
        myfile (str): Path where the file will be stored.

    Returns:
        None.
    """
    with open(myfile, "wb") as f:
        pickle.dump(data, f)


class Train_DEFTData(Dataset):
    """
    A PyTorch Dataset for loading and transforming training data for DEFT-based rod simulation.
    Each data item is a triple of (previous_vertices, current_vertices, target_vertices, mu_0).

    This is typically used to train a model that predicts how the rod evolves from t to t+training_time_horizon.
    """

    def __init__(self, BDLO_type, n_parent_vertices, n_children_vertices, n_branch,
                 rigid_body_coupling_index, train_set_number, total_time, training_time_horizon, device):
        super(Train_DEFTData, self).__init__()
        # Root directory containing data
        self.root_dir = "../dataset/BDLO%s/train/" % BDLO_type
        file_list = glob.glob(self.root_dir + "*")
        self.device = device

        # Preallocate data holders
        self.BDLOs_previous_vertices = []
        self.BDLOs_vertices = []
        self.BDLOs_target_vertices = []
        n_child1_vertices, n_child2_vertices = n_children_vertices
        self.mu_0 = []
        self.parent_input = []
        self.child1_input = []
        self.child2_input = []

        # Hard-coded points / midpoint (seem to be a user-specific coordinate transformation)
        point1 = np.array([0.652495, 0.012239, -0.703962])
        point2 = np.array([0.359612, 0.012701, -0.701503])
        point3 = np.array([0.358077, 0.009511, -0.995053])
        midpoint = np.array([
            point2[0] + (point1[0] - point2[0]) / 2.,
            (point1[1] + point2[1] + point3[1]) / 3,
            point2[2] - (point2[2] - point3[2]) / 2.
        ])
        midpoint_mod = torch.tensor(np.array([-midpoint[2], -midpoint[0], midpoint[1]]))

        bar = tqdm(file_list)
        for rope_data in bar:
            # Load data, shape: (3, total_time, n_parent_vertices + something?), then rearrange
            verts = torch.tensor(pd.read_pickle(r'%s' % str(rope_data))) \
                .view(3, total_time, -1).permute(1, 2, 0)

            # Separate into parent, child1, child2
            parent_vertices = verts[:, :n_parent_vertices]
            child1_vertices = verts[:, n_parent_vertices: n_parent_vertices + n_child1_vertices - 1]
            child2_vertices = verts[:, n_parent_vertices + n_child1_vertices - 1:]

            # Construct the branched rod data structure for the entire time sequence
            BDLO_vert_no_trans = construct_BDLOs_data(total_time, rigid_body_coupling_index,
                                                      n_parent_vertices, n_children_vertices,
                                                      n_branch, parent_vertices, child1_vertices, child2_vertices)
            # Transform from (x,y,z)->(-z, -x, y) for user coordinate system
            BDLO_vert = torch.zeros_like(BDLO_vert_no_trans)
            BDLO_vert[:, :, :, 0] = -BDLO_vert_no_trans[:, :, :, 2]
            BDLO_vert[:, :, :, 1] = -BDLO_vert_no_trans[:, :, :, 0]
            BDLO_vert[:, :, :, 2] = BDLO_vert_no_trans[:, :, :, 1]

            mu_0_list = torch.zeros(verts.size()[0] - 1 - 1, n_branch, 3).to(self.device)

            # Build up sequences (previous, current, next) for the training horizon
            for i in range(total_time - 1 - training_time_horizon):
                # The size check ensures each chunk is [training_time_horizon, n_branch, n_parent_vertices, 3]
                if not BDLO_vert[i: i + training_time_horizon].size() == (
                training_time_horizon, n_branch, n_parent_vertices, 3):
                    print("False Size")
                self.BDLOs_previous_vertices.append(BDLO_vert[i: i + training_time_horizon].numpy())
                self.BDLOs_vertices.append(BDLO_vert[i + 1: i + 1 + training_time_horizon].numpy())
                self.BDLOs_target_vertices.append(BDLO_vert[i + 2: i + 2 + training_time_horizon].numpy())
                self.mu_0.append(mu_0_list[i: i + training_time_horizon])

        # Convert to np.array for indexing
        self.previous_vertices = np.array(self.BDLOs_previous_vertices)
        self.vertices = np.array(self.BDLOs_vertices)
        self.target_vertices = np.array(self.BDLOs_target_vertices)

    def __len__(self):
        # Number of training sequences
        return len(self.vertices)

    def __getitem__(self, index):
        # Return a single training sequence: (previous, current, target, mu_0)
        previous_vertices = torch.tensor(self.previous_vertices[index]).to(self.device)
        vertices = torch.tensor(self.vertices[index]).to(self.device)
        target_vertices = torch.tensor(self.target_vertices[index]).to(self.device)
        return (previous_vertices.clone().detach(),
                vertices.clone().detach(),
                target_vertices.clone().detach(),
                self.mu_0[index].clone().detach())


class Eval_DEFTData(Dataset):
    """
    A PyTorch Dataset for loading and transforming evaluation data for DEFT-based rod simulation.
    Each data item is (previous_vertices, current_vertices, target_vertices), typically over a longer time horizon.
    """

    def __init__(self, BDLO_type, n_parent_vertices, n_children_vertices, n_branch,
                 rigid_body_coupling_index, eval_set_number, total_time, eval_time_horizon, device):
        super(Eval_DEFTData, self).__init__()
        # Root directory for evaluation data
        self.root_dir = "../dataset/BDLO%s/eval/" % BDLO_type
        file_list = glob.glob(self.root_dir + "*")
        self.device = device

        self.BDLOs_previous_vertices = []
        self.BDLOs_vertices = []
        self.BDLOs_target_vertices = []

        n_child1_vertices, n_child2_vertices = n_children_vertices

        # Hard-coded transformations
        point1 = np.array([0.652495, 0.012239, -0.703962])
        point2 = np.array([0.359612, 0.012701, -0.701503])
        point3 = np.array([0.358077, 0.009511, -0.995053])
        midpoint = np.array([
            point2[0] + (point1[0] - point2[0]) / 2.,
            (point1[1] + point2[1] + point3[1]) / 3,
            point2[2] - (point2[2] - point3[2]) / 2.
        ])
        midpoint_mod = torch.tensor(np.array([-midpoint[2], -midpoint[0], midpoint[1]]))

        bar = tqdm(file_list)
        for rope_data in bar:
            # Same reading procedure as training set
            verts = torch.tensor(pd.read_pickle(r'%s' % str(rope_data))) \
                .view(3, total_time, -1).permute(1, 2, 0)

            parent_vertices = verts[:, :n_parent_vertices]
            child1_vertices = verts[:, n_parent_vertices: n_parent_vertices + n_child1_vertices - 1]
            child2_vertices = verts[:, n_parent_vertices + n_child1_vertices - 1:]

            BDLO_vert_no_trans = construct_BDLOs_data(total_time, rigid_body_coupling_index,
                                                      n_parent_vertices, n_children_vertices,
                                                      n_branch, parent_vertices, child1_vertices, child2_vertices)
            BDLO_vert = torch.zeros_like(BDLO_vert_no_trans)
            BDLO_vert[:, :, :, 0] = -BDLO_vert_no_trans[:, :, :, 2]
            BDLO_vert[:, :, :, 1] = -BDLO_vert_no_trans[:, :, :, 0]
            BDLO_vert[:, :, :, 2] = BDLO_vert_no_trans[:, :, :, 1]

            # We only take the first [eval_time_horizon] chunk for the previous, current, target.
            if not BDLO_vert[0: 0 + eval_time_horizon].size() == (eval_time_horizon, n_branch, n_parent_vertices, 3):
                print("False Size")
            self.BDLOs_previous_vertices.append(BDLO_vert[0: 0 + eval_time_horizon].numpy())
            self.BDLOs_vertices.append(BDLO_vert[1: 1 + eval_time_horizon].numpy())
            self.BDLOs_target_vertices.append(BDLO_vert[2: 2 + eval_time_horizon].numpy())

        self.previous_vertices = np.array(self.BDLOs_previous_vertices)
        self.vertices = np.array(self.BDLOs_vertices)
        self.target_vertices = np.array(self.BDLOs_target_vertices)

    def __len__(self):
        # Number of evaluation sequences
        return len(self.vertices)

    def __getitem__(self, index):
        # Return (previous, current, target)
        previous_vertices = torch.tensor(self.previous_vertices[index]).to(self.device)
        vertices = torch.tensor(self.vertices[index]).to(self.device)
        target_vertices = torch.tensor(self.target_vertices[index]).to(self.device)
        return (previous_vertices.clone().detach(),
                vertices.clone().detach(),
                target_vertices.clone().detach())
