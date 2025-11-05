import torch.nn as nn
import numpy as np
from numba import njit

@njit
def _numba_inextensibility_constraint_enforcement(
    current_vertices,   # (B, N, 3)
    nominal_length,     # (B, N-1)
    scale,              # (2B, N-1)
    mass_scale,         # (2B, N-1, 3,3)
    zero_mask_num,      # (B, N-1)
    tolerance
):
    B = current_vertices.shape[0]
    N = current_vertices.shape[1]

    # Precompute
    nominal_length_sq = nominal_length * nominal_length

    # Main loop over edges i = 0..(N-2)
    for i in range(N-1):
        # updated_edges shape = (B,3)
        updated_edges = np.zeros((B,3), dtype=np.float64)
        for b in range(B):
            mask_val = zero_mask_num[b, i]
            for k in range(3):
                updated_edges[b,k] = (current_vertices[b, i+1, k]
                                      - current_vertices[b, i, k]) * mask_val

        # denominator = nominal_length_sq[:, i] + sum_of_squares
        denominator = np.zeros(B, dtype=np.float64)
        for b in range(B):
            e_sum = 0.0
            for k in range(3):
                e_sum += updated_edges[b,k]*updated_edges[b,k]
            denominator[b] = nominal_length_sq[b,i] + e_sum

        # l-values, shape (B,)
        l_vals = np.zeros(B, dtype=np.float64)
        for b in range(B):
            if zero_mask_num[b,i] != 0.0:
                # 1 - 2*(L0^2)/denominator
                l_vals[b] = 1.0 - 2.0*(nominal_length_sq[b,i]/denominator[b])

        # If ALL abs(l) < tolerance => skip entire iteration
        # (mimics: are_all_close_to_zero = torch.all(torch.abs(l)<tolerance))
        all_small = True
        for b in range(B):
            if abs(l_vals[b]) >= tolerance:
                all_small = False
                break
        if all_small:
            continue

        # Expand l into shape (2B,)
        l_cat = np.zeros(2*B, dtype=np.float64)
        for b in range(B):
            val = l_vals[b]
            l_cat[2*b]   = val
            l_cat[2*b+1] = val

        # Divide by scale[:, i] => also shape (2B,)
        for s_idx in range(2*B):
            l_cat[s_idx] /= scale[s_idx, i]

        # Multiply each by mass_scale[:, i] => shape(2B, 3,3)
        l_scale = np.zeros((2*B,3,3), dtype=np.float64)
        for s_idx in range(2*B):
            for r in range(3):
                for c in range(3):
                    l_scale[s_idx, r, c] = mass_scale[s_idx, i, r, c]*l_cat[s_idx]

        # We must replicate updated_edges(b, :) -> 2 copies => big_edges(2B,3)
        big_edges = np.zeros((2*B,3), dtype=np.float64)
        for b in range(B):
            for k in range(3):
                val = updated_edges[b,k]
                big_edges[2*b,   k] = val
                big_edges[2*b+1, k] = val

        # Matrix multiply l_scale[s_idx] (3x3) with big_edges[s_idx] (3,)
        # => out (3,). Then store in result[s_idx, :]
        result = np.zeros((2*B, 3), dtype=np.float64)
        for s_idx in range(2*B):
            for r in range(3):
                accum = 0.0
                for c in range(3):
                    accum += l_scale[s_idx, r, c]*big_edges[s_idx, c]
                result[s_idx, r] = accum

        # Reshape (2B,3)->(B,2,3) so we can add into current_vertices
        # exactly the same as your .view(-1,2,3) logic
        for b in range(B):
            # result row 2*b => parent’s i
            # result row 2*b+1 => parent’s i+1
            for k in range(3):
                current_vertices[b, i,   k] += result[2*b,   k]
                current_vertices[b, i+1, k] += result[2*b+1, k]

    return current_vertices

@njit
def _numba_coupling_core(
    p_vertices,        # shape (1, 13, 3) in your example
    c_vertices,        # shape (2, 13, 3)
    c_index,           # shape (2,) e.g. [4,8]
    c_mass_scale       # shape (2,2,3,3)
):
    """
    Replicates the arithmetic updates:
      updated_edges = child_vertices[:, 0] - parent_vertices[:, coupling_index].view(-1, 3)
      parent_vertices[:, coupling_index] += ...
      child_vertices[:, 0] += ...
    We do this in a Numba-friendly way.
    """

    # 1) Compute updated_edges = c_vertices[:,0] - p_vertices[:,c_index].view(-1,3)
    #    c_vertices[:,0] is (2,3)
    #    p_vertices[:,c_index] is (1,2,3), which we flatten -> shape(2,3).
    # Numba cannot do "advanced indexing" the same way as PyTorch, so we do it manually.

    # Let's gather p_vertices[:, c_index] => shape(1, len(c_index), 3).
    # In your example, p_vertices.shape=(1,13,3), c_index=[4,8] => that's shape(1,2,3).
    # We'll flatten that to (2,3).
    k = c_index.shape[0]  # e.g. 2
    # flatten parent slice
    parent_slice = np.zeros((k, 3), dtype=np.float64)
    for j in range(k):
        idx = c_index[j]
        # p_vertices[0, idx, :] => shape (3,) in your example
        parent_slice[j,0] = p_vertices[0, idx, 0]
        parent_slice[j,1] = p_vertices[0, idx, 1]
        parent_slice[j,2] = p_vertices[0, idx, 2]

    # child_vertices[:,0] => shape(2,3)
    updated_edges = np.zeros((c_vertices.shape[0], 3), dtype=np.float64)
    for row in range(c_vertices.shape[0]):
        for col in range(3):
            updated_edges[row,col] = c_vertices[row,0,col] - parent_slice[row,col]

    # 2) l1 = c_mass_scale[:,0], l2 = c_mass_scale[:,1]
    #    => shape(2,3,3) each
    # We'll do matmul with updated_edges => shape(2,3,1)
    # Then reshape => parent => (1,2,3), child => (2,3)
    # We'll do it explicitly:

    # parent update: l1 -> shape(2,3,3)
    # multiply l1[row,:,:] with updated_edges[row,:] => shape(3,)
    # result => shape(3,) => store in a small array => later we'll reshape to (1,2,3)

    l1_out = np.zeros((c_vertices.shape[0],3), dtype=np.float64)
    l2_out = np.zeros((c_vertices.shape[0],3), dtype=np.float64)
    for row in range(c_vertices.shape[0]):
        # matmul l1[row](3x3) * updated_edges[row](3,)
        for r in range(3):
            acc = 0.0
            for c in range(3):
                acc += c_mass_scale[row, 0, r, c]*updated_edges[row,c]
            l1_out[row,r] = acc

        # same for l2
        for r in range(3):
            acc = 0.0
            for c in range(3):
                acc += c_mass_scale[row, 1, r, c]*updated_edges[row,c]
            l2_out[row,r] = acc

    # l1_out => shape(2,3). In PyTorch we do .view(-1, len(coupling_index), 3) => (1,2,3)
    # so we fold dimension 0=2 -> dimension [1,2], i.e. we want final shape(1,2,3).
    # We'll do the same reorder: that means the first dimension becomes 1,
    # and second dimension = k=2, third dimension=3.
    # We'll just manually add to p_vertices[0, c_index[j], :] for j in [0..k-1].

    # add to parent
    for j in range(k):
        p_vertices[0, c_index[j], 0] += l1_out[j,0]
        p_vertices[0, c_index[j], 1] += l1_out[j,1]
        p_vertices[0, c_index[j], 2] += l1_out[j,2]

    # child update: l2_out => shape(2,3) => we add that to child_vertices[:,0]
    for row in range(c_vertices.shape[0]):
        for col in range(3):
            c_vertices[row,0,col] += l2_out[row,col]

    # done
    return p_vertices, c_vertices

"""rotation"""
@njit
def _quaternion_invert(q):
    """
    Inverse of a normalized quaternion q = (w, x, y, z).
    """
    return np.array([ q[0], -q[1], -q[2], -q[3]], dtype=np.float64)

@njit
def _quaternion_multiply(q1, q2):
    """
    Hamilton product of two quaternions q1 * q2.
    q1,q2: shape(4,) => [w, x, y, z].
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=np.float64)

@njit
def _quaternion_norm(q):
    return np.sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3])

@njit
def _quaternion_apply(q, v):
    """
    Rotate 3D vector v by quaternion q (assumed normalized).
    Returns the rotated vector.
    Formula: v' = v + 2 * cross(q.xyz, cross(q.xyz, v) + q.w*v)
    """
    w, x, y, z = q
    cx = y*v[2] - z*v[1]
    cy = z*v[0] - x*v[2]
    cz = x*v[1] - y*v[0]

    rx = cx + w*v[0]
    ry = cy + w*v[1]
    rz = cz + w*v[2]

    c2x = y*rz - z*ry
    c2y = z*rx - x*rz
    c2z = x*ry - y*rx

    return np.array([v[0] + 2*c2x,
                     v[1] + 2*c2y,
                     v[2] + 2*c2z], dtype=np.float64)


@njit
def _axis_angle_to_quaternion(axis_angle):
    """
    Convert a 3D axis-angle vector (axis * angle) of shape (3,)
    back to a quaternion [w, x, y, z].

    The length of axis_angle is the rotation angle,
    and its direction is the rotation axis.
    """
    eps = 1e-30
    rx, ry, rz = axis_angle
    theta = np.sqrt(rx * rx + ry * ry + rz * rz)
    if theta < eps:
        # angle ~ 0 => return identity quaternion
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    half = 0.5 * theta
    sin_ = np.sin(half)
    cos_ = np.cos(half)

    # normalized axis
    ux = rx / theta
    uy = ry / theta
    uz = rz / theta

    # quaternion = [cos(half), sin(half)*axis]
    return np.array([cos_, ux * sin_, uy * sin_, uz * sin_], dtype=np.float64)


@njit
def _quaternion_to_axis_angle(q):
    """
    Convert quaternion `q` => a 3D axis-angle vector of shape (3,).
    The magnitude of the returned vector = angle in radians,
    and the direction is the rotation axis.

    If angle is near zero, returns [0,0,0].
    """
    # Normalize q just in case
    eps = 1e-30
    norm_q = np.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    if norm_q < eps:
        # Degenerate quaternion => no rotation
        return np.zeros(3, dtype=np.float64)

    w = q[0] / norm_q
    x = q[1] / norm_q
    y = q[2] / norm_q
    z = q[3] / norm_q

    angle = 2.0 * np.arccos(w)
    s = np.sqrt(1.0 - w * w)

    if s < eps:
        # angle ~ 0 => axis can be anything, so we choose (0,0,0) as "no rotation"
        return np.zeros(3, dtype=np.float64)

    # unit axis
    ux = x / s
    uy = y / s
    uz = z / s
    # return the axis * angle
    return np.array([angle * ux, angle * uy, angle * uz], dtype=np.float64)

@njit
def _rotation_matrix_from_vectors(v1, v2):
    """
    Rotate v1 -> v2.
    Return a 3x3 rotation matrix using a standard cross/dot-based formula.
    If v1 or v2 is near 0-length or they are antiparallel, handle it with fallback.
    """
    eps = 1e-30
    norm1 = np.sqrt(np.sum(v1*v1))
    norm2 = np.sqrt(np.sum(v2*v2))
    if norm1<eps or norm2<eps:
        # no rotation
        return np.eye(3, dtype=np.float64)
    a = v1 / norm1
    b = v2 / norm2
    cross_ = np.array([a[1]*b[2] - a[2]*b[1],
                       a[2]*b[0] - a[0]*b[2],
                       a[0]*b[1] - a[1]*b[0]], dtype=np.float64)
    dot_ = a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
    s = np.sqrt(np.sum(cross_*cross_))

    eye = np.eye(3, dtype=np.float64)
    if s<1e-30:
        # parallel or antiparallel
        if dot_>0:
            # parallel => identity
            return eye
        else:
            # 180 deg => find axis
            # pick any perpendicular
            axis = np.array([1.0,0.0,0.0], dtype=np.float64)
            if np.abs(a[0])>0.9:
                axis = np.array([0.0,1.0,0.0], dtype=np.float64)
            perp = np.array([a[1]*axis[2] - a[2]*axis[1],
                             a[2]*axis[0] - a[0]*axis[2],
                             a[0]*axis[1] - a[1]*axis[0]], dtype=np.float64)
            normp = np.sqrt(np.sum(perp*perp))
            if normp<1e-30:
                return eye
            perp/=normp
            # R = I + 2*K^2
            K = np.zeros((3,3), dtype=np.float64)
            K[0,1] = -perp[2]
            K[0,2] =  perp[1]
            K[1,0] =  perp[2]
            K[1,2] = -perp[0]
            K[2,0] = -perp[1]
            K[2,1] =  perp[0]
            return eye + 2.0*(K @ K)
    # general Rodrigues
    K = np.zeros((3,3), dtype=np.float64)
    K[0,1] = -cross_[2]
    K[0,2] =  cross_[1]
    K[1,0] =  cross_[2]
    K[1,2] = -cross_[0]
    K[2,0] = -cross_[1]
    K[2,1] =  cross_[0]

    return eye + K + (K @ K)*((1.0 - dot_)/(s*s))

@njit
def _matrix_to_quaternion(mat):
    """
    Convert 3x3 rotation matrix to quaternion (w, x, y, z).
    """
    tr = mat[0,0] + mat[1,1] + mat[2,2]
    if tr > 0.0:
        S = np.sqrt(tr + 1.0)*2
        qw = 0.25*S
        qx = (mat[2,1] - mat[1,2])/S
        qy = (mat[0,2] - mat[2,0])/S
        qz = (mat[1,0] - mat[0,1])/S
    else:
        if (mat[0,0] > mat[1,1]) and (mat[0,0] > mat[2,2]):
            S = np.sqrt(1.0 + mat[0,0] - mat[1,1] - mat[2,2])*2
            qw = (mat[2,1] - mat[1,2])/S
            qx = 0.25*S
            qy = (mat[0,1] + mat[1,0])/S
            qz = (mat[0,2] + mat[2,0])/S
        elif mat[1,1] > mat[2,2]:
            S = np.sqrt(1.0 + mat[1,1] - mat[0,0] - mat[2,2])*2
            qw = (mat[0,2] - mat[2,0])/S
            qx = (mat[0,1] + mat[1,0])/S
            qy = 0.25*S
            qz = (mat[1,2] + mat[2,1])/S
        else:
            S = np.sqrt(1.0 + mat[2,2] - mat[0,0] - mat[1,1])*2
            qw = (mat[1,0] - mat[0,1])/S
            qx = (mat[0,2] + mat[2,0])/S
            qy = (mat[1,2] + mat[2,1])/S
            qz = 0.25*S
    return np.array([qw, qx, qy, qz], dtype=np.float64)

###############################################################################
# 2) Subfunction: _numba_apply_rotation_moi(...)
###############################################################################

@njit
def _numba_apply_rotation_moi(
    parent_rod_vertices,   # shape (B, 2*n_children, 3)
    child_rod_vertices,    # shape (B, 2*n_children, 3)
    q1_array,              # shape (B*n_children, 4) or similar
    q2_array,              # shape (B*n_children, 4) or similar
    big_mscale,            # shape (B*n_children, 3, 3) -> momentum_scale
    B,
    total_rc
):
    """
    Numba-accelerated version of the PyTorch apply_rotation_test.
    Returns:
       out_r   -> for example, updated rod vertices (could combine parent+child)
       out_q1  -> updated orientation of parent
       out_q2  -> updated orientation of child
    """

    # Number of "pairs" we are dealing with
    N = B * total_rc # e.g. the flatten dimension for q1_array, q2_array, etc.
    # 1) Compute updated quaternion = q1 * invert(q2)
    # We'll store them in updated_q
    updated_q = np.zeros((N, 4), dtype=q1_array.dtype)
    for i in range(N):
        inv_q2 = _quaternion_invert(q2_array[i])
        updated_q[i] = _quaternion_multiply(q1_array[i], inv_q2)

    # 2) Convert updated_q to axis_angle
    delta_angular = np.zeros((N, 3), dtype=q1_array.dtype)
    for i in range(N):
        delta_angular[i] = _quaternion_to_axis_angle(updated_q[i])

    # 3) Multiply delta_angular by big_mscale ( = momentum_scale ).
    #    i.e. for each i, do big_mscale[i,:,:] dot delta_angular[i].
    delta_angular_rod = np.zeros((N*2, 3), dtype=q1_array.dtype)
    for i in range(N*2):
        # shape is (3,) = (3x3) dot (3,)
        mm = big_mscale[i]
        da = delta_angular[i//N]
        delta_angular_rod[i,0] = mm[0,0]*da[0] + mm[0,1]*da[1] + mm[0,2]*da[2]
        delta_angular_rod[i,1] = mm[1,0]*da[0] + mm[1,1]*da[1] + mm[1,2]*da[2]
        delta_angular_rod[i,2] = mm[2,0]*da[0] + mm[2,1]*da[1] + mm[2,2]*da[2]
    # 4) axis_angle -> quaternion
    #    Then we'll interpret that as 2 rods (?), so shape (N, 2, 4) in the torch code
    #    but here we just keep it as something we can multiply with q1/q2
    #    We'll mimic the same shape logic.
    angular_change_q_rod = np.zeros((N, 2, 4), dtype=q1_array.dtype)
    index = 0
    for i in range(N):
        # We assume we have 2 rods per "slot", or parent/child rods
        # Typically in the original code it was "view(-1, 2, 4)"
        # We'll do the same, so each i -> [0,4], [1,4].
        # In the original code it’s a direct reshape, but let's fill them:
        # Usually the same delta_angular_rod is used for both rods, so we
        # produce the same quaternion for "slot=0" and "slot=1"
        for j in range(2):
            dq = _axis_angle_to_quaternion(delta_angular_rod[index])
            angular_change_q_rod[i,j] = dq
            index += 1
    # 5) Multiply angular_change_q_rod * (concatenated q1,q2).
    #    In the original code: orientation = quaternion_multiply(...).view(N, 2, 4)
    #    Then orientation[:,0], orientation[:,1] => rod_orientation1, rod_orientation2
    out_q1 = np.zeros((N, 4), dtype=q1_array.dtype)
    out_q2 = np.zeros((N, 4), dtype=q1_array.dtype)
    for i in range(N):
        # "edge_q" was cat(q1, q2). We'll just do them separately:
        # rod_orientation1 = angular_change_q_rod[i,0] * q1_array[i]
        # rod_orientation2 = angular_change_q_rod[i,1] * q2_array[i]
        # or you can do the original approach if needed.
        # The original code did:
        #   orientation = quaternion_multiply( angular_change_q_rod, edge_q )
        #   => but for clarity we handle parent vs child separately:
        out_q1[i] = _quaternion_multiply(angular_change_q_rod[i,0], q1_array[i])
        out_q2[i] = _quaternion_multiply(angular_change_q_rod[i,1], q2_array[i])
    # 6) Now do the rod vertices transformations.
    #    Original code: rods_vertices1/2 -> shape (B, n_children, 2, 3)
    #    We have shape (B, 2*n_children, 3). We'll reshape to (B, n_children, 2, 3).
    #    Then stack parent/child, subtract origin, rotate, add origin back, etc.

    # Reshape to (B, total_rc, 2, 3)
    # total_rc = n_children, so 2*n_children = parent_rod_vertices.shape[1]
    parent_rod_vertices_4d = parent_rod_vertices.reshape(B, total_rc, 2, 3)
    child_rod_vertices_4d  = child_rod_vertices.reshape(B, total_rc, 2, 3)

    # For demonstration, let's combine them [ parent, child ] along a new "rod index" dimension
    # shape => (B, total_rc, 2 rods, 2 vertices, 3)
    # (like torch.stack([rods_vertices1, rods_vertices2], dim=2))
    combined_rod_verts = np.zeros((B, total_rc, 2, 2, 3), dtype=parent_rod_vertices.dtype)
    for b in range(B):
        for rc in range(total_rc):
            # rod0 = parent, rod1 = child
            for v in range(2):  # vertex index
                combined_rod_verts[b, rc, 0, v] = parent_rod_vertices_4d[b, rc, v]
                combined_rod_verts[b, rc, 1, v] = child_rod_vertices_4d[b, rc, v]

    # rod_vertices_origin => the "first vertex" as an origin
    # shape (B, total_rc, 2, 1, 3)
    # We can just do it in a loop
    rods_vertices_out = np.zeros_like(combined_rod_verts)
    for b in range(B):
        for rc in range(total_rc):
            for rod_idx in range(2):
                origin = combined_rod_verts[b, rc, rod_idx, 0].copy()  # shape (3,)
                # subtract origin from both vertices
                shifted = np.zeros((2,3), dtype=origin.dtype)
                for v in range(2):
                    shifted[v] = combined_rod_verts[b, rc, rod_idx, v] - origin
                # get the quaternion that rotates this rod:
                # we need the angular_change_q_rod that corresponds to this (b, rc)
                # index in the flattened sense is i = b*total_rc + rc
                i = b*total_rc + rc
                q_rot = angular_change_q_rod[i, rod_idx]  # shape (4,)

                # rotate each vertex about origin
                rotated = np.zeros((2,3), dtype=origin.dtype)
                for v in range(2):
                    rotated[v] = _quaternion_apply(q_rot, shifted[v])
                # add origin back
                for v in range(2):
                    rods_vertices_out[b, rc, rod_idx, v] = rotated[v] + origin

    # Now rods_vertices_out has shape (B, total_rc, 2 rods, 2 vertices, 3).
    # If you want it in the shape (B, 2*total_rc, 3) per rod, you can separate them:
    updated_parent = np.zeros((B, 2*total_rc, 3), dtype=parent_rod_vertices.dtype)
    updated_child  = np.zeros((B, 2*total_rc, 3), dtype=parent_rod_vertices.dtype)
    for b in range(B):
        for rc in range(total_rc):
            # rod0 => parent, rod1 => child
            # each rod has 2 vertices
            updated_parent[b, rc*2+0] = rods_vertices_out[b, rc, 0, 0]
            updated_parent[b, rc*2+1] = rods_vertices_out[b, rc, 0, 1]
            updated_child[b, rc*2+0]  = rods_vertices_out[b, rc, 1, 0]
            updated_child[b, rc*2+1]  = rods_vertices_out[b, rc, 1, 1]

    # Decide how you want to combine them for the final out_r, or keep them separate.
    # For example, let's just return the updated_parent for "out_r".
    # Adjust to your needs:
    return updated_parent, updated_child, out_q1, out_q2

###############################################################################
# 3) Main function:
#    _numba_rotation_constraints_enforcement_parent_children(...)
###############################################################################

@njit
def _numba_rotation_constraints_enforcement_parent_children(
    parent_vertices,       # shape (1, 13, 3)
    parent_orientations,   # shape (1, 12, 4)
    prev_parent_vertices,  # shape (1, 13, 3)
    children_vertices,     # shape (1, 2, 13, 3)
    children_orientations, # shape (1, 2, 4)
    prev_children_vertices,# shape (1, 2, 13, 3)
    parent_MOIs,           # shape (4,3,3)
    children_MOIs,         # shape (2,3,3)
    index_selection,       # shape(2,) e.g. [3,7]
    parent_MOI_index,      # shape(2,) e.g. [0,2]
    momentum_scale,        # shape(4,3,3) or bigger
    tolerance=5e-3,
    big_scale=10.0
):
    B = parent_vertices.shape[0]  # should be 1
    C = children_vertices.shape[1]  # 2 child rods
    k = index_selection.shape[0]    # 2 rods in parent

    # 1) Build [previous_edges + current_edges] => shape( (k+C)*B, 3 )
    # We'll do single-batch loops.
    total_rows = B*k + B*C  # e.g. 1*(2) + 1*(2) = 4
    previous_edges = np.zeros((total_rows, 3), dtype=np.float64)
    current_edges  = np.zeros((total_rows, 3), dtype=np.float64)

    row_count = 0
    # fill from parent's rods
    for b in range(B):
        for i in range(k):
            idx = index_selection[i]
            for r in range(3):
                previous_edges[row_count,r] = (prev_parent_vertices[b, idx+1, r]
                                               - prev_parent_vertices[b, idx, r])
                current_edges[row_count,r]  = (parent_vertices[b, idx+1, r]
                                               - parent_vertices[b, idx, r])
            row_count += 1
    # fill from children rods
    for b in range(B):
        for c_i in range(C):
            for r in range(3):
                previous_edges[row_count,r] = (prev_children_vertices[b, c_i, 1, r]
                                               - prev_children_vertices[b, c_i, 0, r])
                current_edges[row_count,r]  = (children_vertices[b, c_i, 1, r]
                                               - children_vertices[b, c_i, 0, r])
            row_count += 1

    # 2) gather old orientations => shape( (k+C)*B, 4 )
    all_orient = np.zeros((total_rows, 4), dtype=np.float64)
    row_count = 0
    for b in range(B):
        # parent's rods
        for i in range(k):
            idx = index_selection[i]
            for r in range(4):
                all_orient[row_count, r] = parent_orientations[b, idx, r]
            row_count += 1
        # children's rods
        for c_i in range(C):
            for r in range(4):
                all_orient[row_count, r] = children_orientations[b, c_i, r]
            row_count += 1

    # 3) compute quaternions from previous->current
    quaternions = np.zeros((total_rows, 4), dtype=np.float64)
    for i in range(total_rows):
        p_ed = previous_edges[i]
        c_ed = current_edges[i]
        # norm_p = np.sqrt(np.sum(p_ed*p_ed))
        # norm_c = np.sqrt(np.sum(c_ed*c_ed))
        # if norm_p<1e-12 or norm_c<1e-12:
        #     quaternions[i,0] = 1.0
        #     continue
        rot_mat = _rotation_matrix_from_vectors(p_ed, c_ed)
        q_ = _matrix_to_quaternion(rot_mat)
        quaternions[i] = q_


    # 5) multiply onto old orientation => new_orient
    new_orient = np.zeros_like(all_orient)
    for i in range(total_rows):
        new_orient[i] = _quaternion_multiply(quaternions[i], all_orient[i])

    # write back
    row_count = 0
    for b in range(B):
        for i in range(k):
            idx = index_selection[i]
            for r in range(4):
                parent_orientations[b, idx, r] = new_orient[row_count,r]
            row_count += 1
        for c_i in range(C):
            for r in range(4):
                children_orientations[b, c_i, r] = new_orient[row_count,r]
            row_count += 1

    # 6) gather rods for "apply_rotation(...)"
    #    -> In your code:
    #       parent_desired_order = [i, i+1 for i in index_selection]
    #       parent_rod_vertices = parent_vertices[:, parent_desired_order]
    # We'll replicate that logic for the parent. For the child, we do 0..1.

    parent_desired_order = []
    for i in range(k):     # e.g. 2 rods
        idx = index_selection[i]
        parent_desired_order.append(idx)
        parent_desired_order.append(idx+1)
    parent_desired_order = np.array(parent_desired_order, dtype=np.int64)  # shape(2*k,)

    # shape(1, 2*k, 3)
    parent_rod_vertices = np.zeros((B, 2*k, 3), dtype=np.float64)
    for b in range(B):
        for jj in range(2*k):
            vidx = parent_desired_order[jj]
            for r in range(3):
                parent_rod_vertices[b,jj,r] = parent_vertices[b,vidx,r]

    # child_rod_vertices => shape(1, 2*C, 3) if we flatten rods
    child_rod_vertices = np.zeros((B, 2*C, 3), dtype=np.float64)
    for b in range(B):
        for c_i in range(C):
            for v_idx in range(2):
                for r in range(3):
                    child_rod_vertices[b, c_i*2+v_idx, r] = children_vertices[b, c_i, v_idx, r]

    # Next, gather parent's orientation for these rods => shape(1, k,4)
    # children_orient => shape(1, C,4).
    parent_sel_orient = np.zeros((B,k,4), dtype=np.float64)
    for b in range(B):
        for i in range(k):
            idx = index_selection[i]
            for r in range(4):
                parent_sel_orient[b,i,r] = parent_orientations[b, idx, r]
    # child_orient is just children_orientations[b,c_i], shape(1,2,4).

    # We'll flatten them so we can pass them to _numba_apply_rotation_moi in one pass.
    total_rc = k  # e.g. 2 + 2 = 4 rods total
    q1_array = np.zeros((B*total_rc, 4), dtype=np.float64)
    row_count = 0
    for b in range(B):
        for i in range(k):
            for r in range(4):
                q1_array[row_count,r] = parent_sel_orient[b,i,r]
            row_count += 1
    # We'll let q2_array be identity
    q2_array = np.zeros((B*total_rc, 4), dtype=np.float64)
    row_count = 0
    for b in range(B):
        for c_i in range(C):
            for r in range(4):
                q2_array[row_count, r] = children_orientations[b, c_i, r]
            row_count += 1


    # rods_vertices => shape(1, (k+C), 2, 3).
    # We have parent_rod_vertices => shape(1, 2*k,3) => that is k rods of 2 vertices each
    # child_rod_vertices => shape(1, 2*C,3) => that is C rods


    # Build is_parent_mask => shape (k+C,)
    # first k rods => True, next C rods => False
    is_parent_mask = np.zeros(k+C, dtype=np.bool_)
    for i in range(k):
        is_parent_mask[i] = True
    # moi_idx_array => e.g. for parent rods, we use parent_MOI_index[i]; for child rods, we use c_i
    moi_idx_array = np.zeros(k, dtype=np.int64)
    for i in range(k):
        moi_idx_array[i] = parent_MOI_index[i]

    # momentum_scale => shape(4,3,3) => broadcast to (B*total_rc,3,3)
    ms_shape_0 = momentum_scale.shape[0]  # e.g. 4
    big_mscale = np.zeros((B*total_rc*2,3,3), dtype=np.float64)
    for i in range(B*total_rc*2):
        big_mscale[i] = momentum_scale[i % ms_shape_0]

    # 7) call the subfunction
    # print("parent_rod_vertices ", parent_rod_vertices)
    # print("child_rod_vertices ", child_rod_vertices)
    # print("q1_array ", q1_array)
    # print("q2_array", q2_array)
    # print("big_mscale ", big_mscale)
    out_p, out_c, out_q1, out_q2 = _numba_apply_rotation_moi(
        parent_rod_vertices,
        child_rod_vertices,
        q1_array,
        q2_array,
        big_mscale,
        B,
        total_rc
    )
    # print("out_p")
    # print(out_p)
    # print(out_c)
    # print(out_q1.shape)
    # print(out_q2.shape)
    # print(parent_vertices.shape)
    # print(children_vertices.shape)
    # raise Exception('Stop here')

    # 8) write back rods => parent_vertices, children_vertices
    # parent rods => out_r[b, i], i in [0..k-1]
    # print("out_q1.shape", out_q1.shape)
    for b in range(B):
        for i in range(k):
            idx = index_selection[i]
            parent_vertices[b, idx]   = out_p[b, i*k]
            parent_vertices[b, idx+1] = out_p[b, i*k+1]
            parent_orientations[b, idx]   = out_q1[i]

        for c_i in range(C):
            children_vertices[b, c_i, 0] = out_c[b, c_i*C]
            children_vertices[b, c_i, 1] = out_c[b, c_i*C+1]
    # print("parent_orientations ", parent_orientations)
    # If you also want to store out_q1 => parent_orientations or children_orientations,
    # parse them out similarly.
    # We skip for brevity.

    return (parent_vertices, parent_orientations,
            children_vertices, out_q2)


class constraints_enforcement_numba(nn.Module):
    def __init__(self):
        super().__init__()
        self.tolerance = 5e-3
        self.scale = 10.0


    def Inextensibility_Constraint_Enforcement(
            self, batch, current_vertices, nominal_length,
             scale, mass_scale, zero_mask_num
    ):
        """
        Drop-in for your PyTorch method.
        We'll do the bridging in/out + call the Numba function.
        """

        cv_np = current_vertices
        nl_np = nominal_length
        sc_np = scale
        ms_np = mass_scale
        zm_np = zero_mask_num.detach().cpu().numpy().astype(np.float64)
        out_np = _numba_inextensibility_constraint_enforcement(
            cv_np, nl_np, sc_np, ms_np, zm_np,
            self.tolerance
        )
        # out_torch = torch.from_numpy(out_np).to(device)
        return out_np

    def Inextensibility_Constraint_Enforcement_Coupling(
            self,
            parent_vertices,  # shape (1,13,3)
            child_vertices,  # shape (2,13,3)
            coupling_index,  # e.g. [4,8]
            coupling_mass_scale,  # shape (2,2,3,3)
            selected_parent_index,  # e.g. [0]
            selected_children_index  # e.g. [1,2]
    ):


        # Convert to CPU numpy
        p_np = parent_vertices
        c_np = child_vertices

        # coupling_index might be Python list => make int64 array
        if isinstance(coupling_index, (list, tuple)):
            ci_np = np.array(coupling_index, dtype=np.int64)
        else:
            ci_np = coupling_index

        cms_np = coupling_mass_scale

        # 1) Call the Numba core
        p_upd, c_upd = _numba_coupling_core(p_np, c_np, ci_np, cms_np)

        # 2) Rebuild b_DLOs_vertices as in your PyTorch code:
        #    shape = (len(selected_parent_index)+ len(selected_children_index),
        #             parent_vertices.size(1), 3)
        out_size = len(selected_parent_index) + len(selected_children_index)
        n_verts = p_upd.shape[1]  # = 13
        out_np = np.empty((out_size, n_verts, 3), dtype=np.float64)

        # For convenience, assume selected_parent_index is e.g. [0] => we copy the
        # updated parent rods to out_np[0].
        for i, idx in enumerate(selected_parent_index):
            out_np[idx] = p_upd[i]  # i=0 => p_upd[0], shape(13,3)

        # Similarly for child rods
        for i, idx in enumerate(selected_children_index):
            out_np[idx] = c_upd[i]  # c_upd[i], shape(13,3)

        # Convert back to torch
        # b_DLOs_vertices = torch.from_numpy(out_np).to(device=device, dtype=dtype)
        return out_np

    def Rotation_Constraints_Enforcement_Parent_Children(
            self,
            parent_vertices,
            parent_rod_orientation,
            previous_parent_vertices,
            children_vertices,
            children_rod_orientation,
            previous_children_vertices,
            parent_MOI_matrix,
            children_MOI_matrix,
            rigid_body_coupling_index,
            parent_MOI_index,
            momentum_scale
    ):


        # 1) convert to NumPy
        pv_np = parent_vertices
        pro_np = parent_rod_orientation
        ppv_np = previous_parent_vertices
        cv_np = children_vertices
        cro_np = children_rod_orientation
        pcv_np = previous_children_vertices
        pMOI_np = parent_MOI_matrix
        cMOI_np = children_MOI_matrix
        if isinstance(rigid_body_coupling_index, (list, tuple)):
            rci_np = np.array(rigid_body_coupling_index, dtype=np.int64)
        else:
            rci_np = rigid_body_coupling_index

        if isinstance(parent_MOI_index, (list, tuple)):
            pmi_np = np.array(parent_MOI_index, dtype=np.int64)
        else:
            pmi_np = parent_MOI_index

        ms_np = momentum_scale

        # 2) call the numba core
        # print("pv_np", pv_np)
        # print(pro_np)
        # print("ppv_np", ppv_np)
        # print(pv_np)
        # print(rci_np)
        # print(pmi_np)
        out_pv, out_pro, out_cv, out_cro = _numba_rotation_constraints_enforcement_parent_children(
            pv_np, pro_np, ppv_np,
            cv_np, cro_np, pcv_np,
            pMOI_np, cMOI_np,
            rci_np, pmi_np, ms_np,
            tolerance=self.tolerance,
            big_scale=self.scale
        )

        # 3) convert back
        # out_pv = torch.from_numpy(out_pv)
        # out_pro_torch = torch.from_numpy(out_pro).to(device=device, dtype=dtype)
        # out_cv = torch.from_numpy(out_cv)
        # out_cro_torch = torch.from_numpy(out_cro).to(device=device, dtype=dtype)

        return out_pv, out_pro, out_cv, np.expand_dims(out_cro, axis=0)
