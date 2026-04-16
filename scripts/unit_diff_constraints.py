"""
Unit-level diff: compare numba vs torch implementations of the
constraint-enforcement helpers on the SAME input. If a helper agrees
to ~1e-12, it can't be the source of the BDLO5-only ~1e-6 divergence.
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)
np.random.seed(0)

from deft.solvers.constraints_solver import constraints_enforcement
from deft.solvers.constraints_enforcement_numba import constraints_enforcement_numba

t_solver = constraints_enforcement(n_branch=3)
n_solver = constraints_enforcement_numba()


def diff(a, b):
    a_np = a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else a
    b_np = b.detach().cpu().numpy() if isinstance(b, torch.Tensor) else b
    d = np.abs(a_np - b_np)
    return float(d.max()), float(np.sqrt((d**2).mean()))


# ---------------------------------------------------------------------------
# Test 1: Inextensibility_Constraint_Enforcement (the per-rod inext)
# ---------------------------------------------------------------------------
print("=" * 70)
print("Test 1: Inextensibility_Constraint_Enforcement")
print("=" * 70)

# Build a representative input: 3 rods, 12 verts, 3D
# scale and mass_scale need shape (2*batch, n_edge[, 3, 3]) per the function's
# l_cat repeat-by-2 logic.
batch = 3   # 3 rods stacked
n_vert = 12
n_edge = n_vert - 1

verts = (torch.randn(batch, n_vert, 3, dtype=torch.float64) * 0.05
         + torch.linspace(0., 1., n_vert).view(1, n_vert, 1).repeat(batch, 1, 3))
nominal_length = torch.full((batch, n_edge), 1.0 / (n_vert - 1), dtype=torch.float64)
inext_scale = torch.ones(2 * batch, n_edge, dtype=torch.float64)
mass_scale = (
    torch.eye(3, dtype=torch.float64).view(1, 1, 3, 3).repeat(2 * batch, n_edge, 1, 1) * 0.5
)
zero_mask_num = torch.ones(batch, n_edge, dtype=torch.uint8)

verts_torch = verts.clone()
verts_numba = verts.detach().cpu().numpy().astype(np.float64).copy()

mass_matrix = torch.eye(3, dtype=torch.float64).view(1, 1, 3, 3).repeat(batch, n_vert, 1, 1)
clamped = torch.zeros(batch, n_vert, dtype=torch.float64)

out_torch = t_solver.Inextensibility_Constraint_Enforcement(
    batch, verts_torch, nominal_length, mass_matrix, clamped,
    inext_scale, mass_scale, zero_mask_num,
)
out_numba = n_solver.Inextensibility_Constraint_Enforcement(
    batch, verts_numba,
    nominal_length.detach().cpu().numpy().astype(np.float64).copy(),
    inext_scale.detach().cpu().numpy().astype(np.float64).copy(),
    mass_scale.detach().cpu().numpy().astype(np.float64).copy(),
    zero_mask_num,
)

m, r = diff(out_torch, out_numba)
print(f"  max_abs={m:.3e}  rms={r:.3e}")
print()


# ---------------------------------------------------------------------------
# Test 2: Inextensibility_Constraint_Enforcement_Coupling (BDLO1, 2 children)
# ---------------------------------------------------------------------------
print("=" * 70)
print("Test 2: Coupling enforcement, BDLO1-style (bdlo5=False)")
print("=" * 70)

p_verts = (torch.randn(1, n_vert, 3, dtype=torch.float64) * 0.05)
c_verts = (torch.randn(2, n_vert, 3, dtype=torch.float64) * 0.05)
coupling_index = torch.tensor([4, 8], dtype=torch.int64)
cms = (
    torch.eye(3, dtype=torch.float64).view(1, 1, 3, 3).repeat(2, 2, 1, 1) * 0.5
)
selected_parent_index = [0]
selected_children_index = [1, 2]

out_torch = t_solver.Inextensibility_Constraint_Enforcement_Coupling(
    p_verts.clone(), c_verts.clone(), coupling_index,
    cms.clone(), selected_parent_index, selected_children_index,
    bdlo5=False, child2_coupling_mass_scale=None, skip_child2_coupling=False, bdlo6=False,
)
out_numba = n_solver.Inextensibility_Constraint_Enforcement_Coupling(
    p_verts.detach().cpu().numpy().astype(np.float64).copy(),
    c_verts.detach().cpu().numpy().astype(np.float64).copy(),
    coupling_index.detach().cpu().numpy().astype(np.int64).copy(),
    cms.detach().cpu().numpy().astype(np.float64).copy(),
    selected_parent_index, selected_children_index,
    bdlo5=0, child2_coupling_mass_scale=None, bdlo6=0,
)

m, r = diff(out_torch, out_numba)
print(f"  max_abs={m:.3e}  rms={r:.3e}")
print()


# ---------------------------------------------------------------------------
# Test 3: Inextensibility_Constraint_Enforcement_Coupling (BDLO5)
# ---------------------------------------------------------------------------
print("=" * 70)
print("Test 3: Coupling enforcement, BDLO5-style (bdlo5=True)")
print("=" * 70)

p_verts = (torch.randn(1, n_vert, 3, dtype=torch.float64) * 0.05)
c_verts = (torch.randn(2, n_vert, 3, dtype=torch.float64) * 0.05)
coupling_index = torch.tensor([5, 1], dtype=torch.int64)
cms_bdlo5 = (
    torch.eye(3, dtype=torch.float64).view(1, 1, 3, 3).repeat(1, 2, 1, 1) * 0.5
)  # shape (1, 2, 3, 3) for c1<->parent
c2_cms_bdlo5 = (
    torch.eye(3, dtype=torch.float64).view(1, 1, 3, 3).repeat(1, 2, 1, 1) * 0.5
)  # shape (1, 2, 3, 3) for c2<->c1

out_torch = t_solver.Inextensibility_Constraint_Enforcement_Coupling(
    p_verts.clone(), c_verts.clone(), coupling_index,
    cms_bdlo5.clone(), [0], [1, 2],
    bdlo5=True,
    child2_coupling_mass_scale=c2_cms_bdlo5.clone(),
    skip_child2_coupling=False, bdlo6=False,
)
out_numba = n_solver.Inextensibility_Constraint_Enforcement_Coupling(
    p_verts.detach().cpu().numpy().astype(np.float64).copy(),
    c_verts.detach().cpu().numpy().astype(np.float64).copy(),
    coupling_index.detach().cpu().numpy().astype(np.int64).copy(),
    cms_bdlo5.detach().cpu().numpy().astype(np.float64).copy(),
    [0], [1, 2],
    bdlo5=1,
    child2_coupling_mass_scale=c2_cms_bdlo5.detach().cpu().numpy().astype(np.float64).copy(),
    bdlo6=0,
)

m, r = diff(out_torch, out_numba)
print(f"  max_abs={m:.3e}  rms={r:.3e}")
print()

print("Done.")
