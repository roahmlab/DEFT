"""Quick eval of BDLO5 on training data — runs each sample through the sim and prints RMSE."""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

torch.set_default_dtype(torch.float64)
torch.manual_seed(1)

from deft.utils.util import DEFT_initialization, construct_b_DLOs, clamp_index, index_init, Train_DEFTData
from deft.core.DEFT_sim import DEFT_sim

# ---- BDLO5 parameters ----
n_branch = 3
n_parent_vertices = 12
n_child1_vertices = 4
n_child2_vertices = 4
cs_n_vert = (4, 4)
n_vert = 12
n_edge = 11
device = "cpu"
total_time = 500
rigid_body_coupling_index = [5, 1]
bdlo5 = True

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

# ---- Build sim ----
bend_stiffness_parent = nn.Parameter(1e-3 * torch.ones((1, 1, n_edge), device=device, dtype=torch.float64))
bend_stiffness_child1 = nn.Parameter(1e-3 * torch.ones((1, 1, n_edge), device=device, dtype=torch.float64))
bend_stiffness_child2 = nn.Parameter(1e-3 * torch.ones((1, 1, n_edge), device=device, dtype=torch.float64))
twist_stiffness = nn.Parameter(1e-4 * torch.ones((1, n_branch, n_edge), device=device, dtype=torch.float64))
damping = nn.Parameter(torch.tensor((3., 3., 3.), device=device, dtype=torch.float64))
learning_weight = nn.Parameter(torch.tensor(0.00, device=device, dtype=torch.float64))

parent_clamped_selection = torch.tensor((0, 1, -2, -1))
child1_clamped_selection = torch.tensor((2))
child2_clamped_selection = torch.tensor((2))

parent_v = undeformed_BDLO[:, :n_parent_vertices]
child1_v = undeformed_BDLO[:, n_parent_vertices:n_parent_vertices + n_child1_vertices - 1]
child2_v = undeformed_BDLO[:, n_parent_vertices + n_child1_vertices - 1:]

b_DLO_mass, parent_MOI, children_MOI, _, _, _ = DEFT_initialization(
    parent_v, child1_v, child2_v, n_branch, n_parent_vertices, cs_n_vert,
    rigid_body_coupling_index, 1., 10., (0.5, 0.5), (1, 1), 0.1, bdlo5=True)

b, _ = construct_b_DLOs(1, rigid_body_coupling_index, n_parent_vertices, cs_n_vert, n_branch,
    parent_v, parent_v, child1_v, child1_v, child2_v, child2_v, bdlo5=True)

# Transform (same as DEFT_train.py for BDLO5)
step1 = torch.zeros_like(b)
step1[:, :, :, 0] = -b[:, :, :, 2]
step1[:, :, :, 1] = -b[:, :, :, 0]
step1[:, :, :, 2] = b[:, :, :, 1]
t = torch.zeros_like(step1)
t[:, :, :, 0] = -step1[:, :, :, 0]
t[:, :, :, 1] = step1[:, :, :, 2]
t[:, :, :, 2] = -step1[:, :, :, 1]
b_undeformed_vert = t[0].view(n_branch, -1, 3)

index_selection1, index_selection2, parent_MOI_index1, parent_MOI_index2 = index_init(
    rigid_body_coupling_index, n_branch)

batch = 1
eval_time_horizon = total_time - 2

clamped_index, parent_theta_clamp, child1_theta_clamp, child2_theta_clamp = clamp_index(
    batch, parent_clamped_selection, child1_clamped_selection, child2_clamped_selection,
    n_branch, n_parent_vertices, True, False, False)

sim = DEFT_sim(
    batch=batch, n_branch=n_branch, n_vert=n_vert, cs_n_vert=cs_n_vert,
    b_init_n_vert=b_undeformed_vert, n_edge=n_edge, b_undeformed_vert=b_undeformed_vert,
    b_DLO_mass=b_DLO_mass, parent_DLO_MOI=parent_MOI, children_DLO_MOI=children_MOI,
    device=device, clamped_index=clamped_index, rigid_body_coupling_index=rigid_body_coupling_index,
    parent_MOI_index1=parent_MOI_index1, parent_MOI_index2=parent_MOI_index2,
    parent_clamped_selection=parent_clamped_selection,
    child1_clamped_selection=child1_clamped_selection,
    child2_clamped_selection=child2_clamped_selection,
    clamp_parent=True, clamp_child1=False, clamp_child2=False,
    index_selection1=index_selection1, index_selection2=index_selection2,
    bend_stiffness_parent=bend_stiffness_parent, bend_stiffness_child1=bend_stiffness_child1,
    bend_stiffness_child2=bend_stiffness_child2, twist_stiffness=twist_stiffness,
    damping=damping, learning_weight=learning_weight,
    use_orientation_constraints=True, use_attachment_constraints=True,
    bdlo5=True, use_child2_orientation=True,
    skip_child2_coupling=False)

# ---- Load eval data ----
from torch.utils.data import DataLoader
from deft.utils.util import Eval_DEFTData
print("Loading eval data...")
eval_dataset = Eval_DEFTData(
    5, n_parent_vertices, (n_child1_vertices, n_child2_vertices), n_branch,
    rigid_body_coupling_index, 25, total_time, eval_time_horizon, device,
    bdlo5=True)

loader = DataLoader(eval_dataset, batch_size=batch, shuffle=False, drop_last=True)

# Map sample idx → filename (using same glob order as Eval_DEFTData)
import glob
eval_files = glob.glob('../dataset/BDLO5/eval/*')
print(f"\nSample → file mapping:")
for si, f in enumerate(eval_files):
    print(f"  [{si:2d}] -> {os.path.basename(f)}")

loss_func = torch.nn.MSELoss()
print(f"\nTotal eval samples: {len(eval_dataset)}, batch={batch}")

RMSE_THRESHOLD = 1.0
bad_global_indices = []

# Single-sample debug run: take just one batch (= one sample, since batch=1),
# enable visualization, and stay on the pytorch path (inference_1_batch=False).
DEBUG_SAMPLE_IDX = 0  # which file in eval_files glob order to run

with torch.no_grad():
    for batch_idx, data in enumerate(tqdm(loader, desc="Eval")):
        if batch_idx != DEBUG_SAMPLE_IDX:
            continue
        prev, curr, target = data
        traj_loss, _ = sim.iterative_sim(
            eval_time_horizon, curr, prev, target,
            loss_func, 0.01,
            parent_theta_clamp, child1_theta_clamp, child2_theta_clamp,
            False, vis_type="test", vis=True)

        total_rmse = np.sqrt(traj_loss.cpu().numpy() / total_time)
        print(f"\nTotal RMSE: {total_rmse:.4f}")

        per_sample = sim.last_sample_traj_rmse
        if per_sample is not None:
            for local_si in range(per_sample.shape[0]):
                if per_sample[local_si].item() > RMSE_THRESHOLD:
                    bad_global_indices.append(batch_idx * batch + local_si)
        break

# ---- Move bad samples from eval/ to train/ ----
import shutil
eval_dir = '../dataset/BDLO5/eval'
train_dir = '../dataset/BDLO5/train'

print(f"\n{'='*60}")
print(f"Found {len(bad_global_indices)} samples with traj RMSE > {RMSE_THRESHOLD}")
print(f"{'='*60}")

to_move = []
for gi in bad_global_indices:
    if gi >= len(eval_files):
        print(f"  [skip] global idx {gi} out of range (only {len(eval_files)} files)")
        continue
    src = eval_files[gi]
    dst = os.path.join(train_dir, os.path.basename(src))
    to_move.append((gi, src, dst))
    print(f"  [{gi:2d}] {os.path.basename(src)}")

if to_move:
    resp = input(f"\nMove these {len(to_move)} files from eval/ to train/? [y/N]: ").strip().lower()
    if resp == 'y':
        for gi, src, dst in to_move:
            shutil.move(src, dst)
            print(f"  moved: {os.path.basename(src)}")
        print(f"\nDone. Moved {len(to_move)} files.")
    else:
        print("Aborted; no files moved.")
else:
    print("Nothing to move.")
