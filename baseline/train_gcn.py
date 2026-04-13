"""
Training script for standalone GCN baseline on BDLO dynamics prediction.

Uses the same BatchedGNNModel (GCN with tree-structured adjacency) from
deft/models/GNN_tree.py, but trains it as a standalone predictor (no physics).

Reuses the same dataset classes (Train_DEFTData, Eval_DEFTData) as DEFT training.
Supports both teacher-forced and autoregressive training.

Usage (run from baseline/ directory):
    python train_gcn.py --BDLO_type 1 --clamp_type ends
    python train_gcn.py --BDLO_type 5 --clamp_type ends
    python train_gcn.py --BDLO_type 6 --clamp_type ends
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import sys
import os
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deft.utils.util import Train_DEFTData, Eval_DEFTData, save_pickle
from deft.models.GNN_tree import BatchedGNNModel
from tqdm import tqdm


# BDLO configurations per type and clamp mode
BDLO_CONFIGS = {
    (1, "ends"): {
        "n_parent": 13,
        "cs_n_vert": (5, 4),
        "coupling": [4, 8],
        "train_n": 77,
        "eval_n": 24,
        "parent_clamp": [0, 1, -2, -1],
        "clamp_parent": True, "clamp_child1": False, "clamp_child2": False,
    },
    (1, "middle"): {
        "n_parent": 13,
        "cs_n_vert": (5, 4),
        "coupling": [4, 8],
        "train_n": 71,
        "eval_n": 18,
        "parent_clamp": [2, -2, -1],
        "clamp_parent": True, "clamp_child1": False, "clamp_child2": False,
    },
    (2, "ends"): {
        "n_parent": 12,
        "cs_n_vert": (5, 5),
        "coupling": [4, 7],
        "train_n": 110,
        "eval_n": 37,
        "parent_clamp": [0, 1, -2, -1],
        "clamp_parent": True, "clamp_child1": False, "clamp_child2": False,
    },
    (3, "ends"): {
        "n_parent": 12,
        "cs_n_vert": (3, 4),
        "coupling": [4, 7],
        "train_n": 103,
        "eval_n": 26,
        "parent_clamp": [0, 1, -2, -1],
        "clamp_parent": True, "clamp_child1": False, "clamp_child2": False,
    },
    (3, "middle"): {
        "n_parent": 12,
        "cs_n_vert": (3, 4),
        "coupling": [4, 7],
        "train_n": 39,
        "eval_n": 12,
        "parent_clamp": [3, -2, -1],
        "clamp_parent": True, "clamp_child1": False, "clamp_child2": False,
    },
    (4, "ends"): {
        "n_parent": 12,
        "cs_n_vert": (3, 3),
        "coupling": [4, 8],
        "train_n": 74,
        "eval_n": 25,
        "parent_clamp": [0, 1, -2, -1],
        "clamp_parent": True, "clamp_child1": False, "clamp_child2": False,
    },
    (5, "ends"): {
        "n_parent": 12,
        "cs_n_vert": (4, 4),
        "coupling": [5, 1],
        "train_n": 75,
        "eval_n": 26,
        "parent_clamp": [0, 1, -2, -1],
        "clamp_parent": True, "clamp_child1": False, "clamp_child2": False,
        "bdlo5": True,
    },
    (6, "ends"): {
        "n_parent": 12,
        "cs_n_vert": (5, 3, 4),
        "coupling": [2, 7, 7],
        "train_n": 66,
        "eval_n": 21,
        "parent_clamp": [0, 1, -2, -1],
        "clamp_parent": True, "clamp_child1": False, "clamp_child2": False,
        "n_branch": 4,
        "bdlo6": True,
    },
}


def compute_index_selections(n_parent, cs_n_vert, n_branch, batch):
    """Compute the index arrays that BatchedGNNModel needs for reshaping."""
    n_vert = n_parent
    # selected_parent_index: indices of parent branch rows in (batch*n_branch, n_vert, 3)
    selected_parent_index = torch.arange(0, batch * n_branch, n_branch)
    # selected_children_index: indices of all child branch rows
    children_indices = []
    for b in range(batch):
        for c in range(1, n_branch):
            children_indices.append(b * n_branch + c)
    selected_children_index = torch.tensor(children_indices)
    # Per-child indices
    selected_child1_index = torch.arange(1, batch * n_branch, n_branch)
    selected_child2_index = torch.arange(2, batch * n_branch, n_branch)
    selected_child3_index = None
    if n_branch >= 4:
        selected_child3_index = torch.arange(3, batch * n_branch, n_branch)
    return (selected_child1_index, selected_child2_index, selected_child3_index,
            selected_parent_index, selected_children_index)


def create_valid_mask(n_parent_vertices, cs_n_vert, n_branch):
    """Create a binary mask for valid (non-padded) vertices."""
    mask = torch.zeros(n_branch, n_parent_vertices)
    mask[0, :] = 1.0
    for i, c_n in enumerate(cs_n_vert):
        mask[i + 1, :c_n] = 1.0
    return mask


def train(args):
    config_key = (args.BDLO_type, args.clamp_type)
    if config_key not in BDLO_CONFIGS:
        raise ValueError(
            f"No config for BDLO_type={args.BDLO_type}, clamp_type={args.clamp_type}. "
            f"Available: {list(BDLO_CONFIGS.keys())}"
        )

    config = BDLO_CONFIGS[config_key]
    n_parent = config["n_parent"]
    cs_n_vert = config["cs_n_vert"]
    coupling = config["coupling"]
    train_n = config["train_n"]
    eval_n = config["eval_n"]
    parent_clamp = config["parent_clamp"]
    clamp_parent = config["clamp_parent"]
    clamp_child1 = config["clamp_child1"]
    clamp_child2 = config["clamp_child2"]
    n_branch = config.get("n_branch", 3)
    bdlo5 = config.get("bdlo5", False)
    bdlo6 = config.get("bdlo6", False)
    n_children = cs_n_vert

    parent_clamped_selection = torch.tensor(parent_clamp)
    child1_clamped_selection = torch.tensor((2))
    child2_clamped_selection = torch.tensor((2))

    valid_mask = create_valid_mask(n_parent, cs_n_vert, n_branch)

    # Load datasets
    if args.BDLO_type == 6:
        from deft.utils.util import Train_DEFTData_BDLO6, Eval_DEFTData_BDLO6
        print("Loading BDLO6 training data...")
        train_dataset = Train_DEFTData_BDLO6(
            args.total_time,
            args.train_time_horizon,
            args.device,
        )
        print(f"Training samples: {len(train_dataset)}")

        print("Loading BDLO6 evaluation data...")
        eval_time_horizon = args.total_time - 2
        eval_dataset = Eval_DEFTData_BDLO6(
            args.total_time,
            eval_time_horizon,
            args.device,
        )
        print(f"Evaluation samples: {len(eval_dataset)}")
    else:
        if args.clamp_type == "ends":
            bdlo_key = args.BDLO_type
        else:
            bdlo_key = str(args.BDLO_type) + "_mid_clamp"

        print("Loading training data...")
        train_dataset = Train_DEFTData(
            bdlo_key, n_parent, n_children, n_branch, coupling,
            train_n, args.total_time, args.train_time_horizon, args.device,
            bdlo5=bdlo5,
        )
        print(f"Training samples: {len(train_dataset)}")

        print("Loading evaluation data...")
        eval_time_horizon = args.total_time - 2
        eval_dataset = Eval_DEFTData(
            bdlo_key, n_parent, n_children, n_branch, coupling,
            eval_n, args.total_time, eval_time_horizon, args.device,
            bdlo5=bdlo5,
        )
        print(f"Evaluation samples: {len(eval_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=args.train_batch, shuffle=True, drop_last=True
    )

    # Compute index selections for the GNN
    (selected_child1_index, selected_child2_index, selected_child3_index,
     selected_parent_index, selected_children_index) = compute_index_selections(
        n_parent, cs_n_vert, n_branch, args.train_batch
    )

    # Model: standalone GCN with tree-structured adjacency
    # Input: current pos (3) + previous pos (3) = 6 features per node
    model = BatchedGNNModel(
        batch=args.train_batch,
        in_features=6,
        hidden_features=args.hidden_size,
        out_features=3,
        n_vert=n_parent,
        cs_n_vert=cs_n_vert,
        rigid_body_coupling_index=coupling,
        clamp_parent=clamp_parent,
        clamp_child1=clamp_child1,
        clamp_child2=clamp_child2,
        parent_clamped_selection=parent_clamped_selection,
        child1_clamped_selection=child1_clamped_selection,
        child2_clamped_selection=child2_clamped_selection,
        selected_child1_index=selected_child1_index,
        selected_child2_index=selected_child2_index,
        selected_parent_index=selected_parent_index,
        selected_children_index=selected_children_index,
        bdlo5=bdlo5,
        bdlo6=bdlo6,
        selected_child3_index=selected_child3_index,
    )
    model = model.double()
    model = model.to(args.device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # For eval, we need a separate model instance with eval batch size
    # (or we rebuild adjacency). We'll handle this in evaluate().

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_func = nn.MSELoss()

    train_losses = []
    train_steps = []
    eval_losses = []
    eval_epochs = []
    iteration = 0
    training_case = 1

    for epoch in range(args.train_epochs):
        model.train()
        bar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for data in bar:
            prev_traj, curr_traj, target_traj, _ = data
            # Shapes: [batch, time_horizon, n_branch, n_vert, 3]

            if iteration % args.evaluate_period == 0:
                model.eval()
                eval_loss = evaluate(
                    model, eval_dataset, eval_n, n_parent, cs_n_vert,
                    valid_mask, parent_clamp, eval_time_horizon, args,
                    n_branch=n_branch, bdlo5=bdlo5, bdlo6=bdlo6,
                    clamp_parent=clamp_parent, clamp_child1=clamp_child1,
                    clamp_child2=clamp_child2,
                    parent_clamped_selection=parent_clamped_selection,
                    child1_clamped_selection=child1_clamped_selection,
                    child2_clamped_selection=child2_clamped_selection,
                    coupling=coupling,
                )
                eval_losses.append(eval_loss)
                eval_epochs.append(iteration)
                print(f"\n[Iter {iteration}] Eval RMSE: {np.sqrt(eval_loss):.6f}")

                torch.save(
                    model.state_dict(),
                    os.path.join(
                        "../save_model/",
                        "GCN_%s_%s_%s_%s.pth"
                        % (args.clamp_type, args.BDLO_type, iteration, training_case),
                    ),
                )
                save_pickle(
                    eval_losses,
                    "../training_record/eval_%s_loss_GCN_%s_%s.pkl"
                    % (args.clamp_type, training_case, args.BDLO_type),
                )
                save_pickle(
                    eval_epochs,
                    "../training_record/eval_%s_epoches_GCN_%s_%s.pkl"
                    % (args.clamp_type, training_case, args.BDLO_type),
                )
                model.train()

            # --- Training step ---
            total_loss = 0.0
            batch_size = prev_traj.size(0)

            # Build clamped input hints for the full trajectory
            inputs = torch.zeros_like(target_traj)
            if clamp_parent:
                inputs[:, :, 0, parent_clamped_selection] = target_traj[:, :, 0, parent_clamped_selection]

            for t in range(args.train_time_horizon):
                if t == 0:
                    curr = curr_traj[:, t]   # [batch, n_branch, n_vert, 3]
                    prev = prev_traj[:, t]
                target = target_traj[:, t]
                inp = inputs[:, t]

                # Flatten for GNN: [batch, n_branch*n_vert, 3]
                curr_flat = curr.reshape(batch_size, -1, 3)
                prev_flat = prev.reshape(batch_size, -1, 3)
                inp_flat = inp.reshape(batch_size, -1, 3)

                # GNN input: [batch, n_branch*n_vert, 6]
                x = torch.cat([curr_flat, prev_flat], dim=-1)
                pred_flat = model.inference(x, inp_flat)  # [batch, n_branch*n_vert, 3]

                # Masked loss
                target_flat = target.reshape(batch_size, -1, 3)
                mask = (
                    valid_mask.reshape(1, -1, 1)
                    .to(pred_flat.device, dtype=pred_flat.dtype)
                    .expand_as(pred_flat)
                )
                step_loss = loss_func(pred_flat * mask, target_flat * mask)
                total_loss += step_loss

                # Prepare next step
                if args.teacher_forcing:
                    if t + 1 < args.train_time_horizon:
                        prev = prev_traj[:, t + 1]
                        curr = curr_traj[:, t + 1]
                else:
                    prev = curr
                    pred_branched = pred_flat.reshape(batch_size, n_branch, n_parent, 3)
                    curr = pred_branched.detach() if args.detach_autoregressive else pred_branched

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            avg_loss = total_loss.item() / args.train_time_horizon
            train_losses.append(avg_loss)
            train_steps.append(iteration)
            bar.set_postfix(loss=f"{avg_loss:.6f}")

            save_pickle(
                train_losses,
                "../training_record/train_%s_loss_GCN_%s_%s.pkl"
                % (args.clamp_type, training_case, args.BDLO_type),
            )
            save_pickle(
                train_steps,
                "../training_record/train_%s_step_GCN_%s_%s.pkl"
                % (args.clamp_type, training_case, args.BDLO_type),
            )

            iteration += 1


def evaluate(
    model, eval_dataset, eval_n, n_parent, cs_n_vert, valid_mask,
    parent_clamp, eval_time_horizon, args, n_branch=3,
    bdlo5=False, bdlo6=False,
    clamp_parent=True, clamp_child1=False, clamp_child2=False,
    parent_clamped_selection=None, child1_clamped_selection=None,
    child2_clamped_selection=None, coupling=None,
):
    """Autoregressive evaluation using a fresh GNN with eval batch size."""
    eval_loader = DataLoader(
        eval_dataset, batch_size=eval_n, shuffle=False, drop_last=True
    )
    loss_func = nn.MSELoss()

    # Build eval-batch-sized GNN (adjacency depends on batch size)
    (sel_c1, sel_c2, sel_c3, sel_p, sel_ch) = compute_index_selections(
        n_parent, cs_n_vert, n_branch, eval_n
    )
    eval_model = BatchedGNNModel(
        batch=eval_n,
        in_features=6,
        hidden_features=args.hidden_size,
        out_features=3,
        n_vert=n_parent,
        cs_n_vert=cs_n_vert,
        rigid_body_coupling_index=coupling,
        clamp_parent=clamp_parent,
        clamp_child1=clamp_child1,
        clamp_child2=clamp_child2,
        parent_clamped_selection=parent_clamped_selection,
        child1_clamped_selection=child1_clamped_selection,
        child2_clamped_selection=child2_clamped_selection,
        selected_child1_index=sel_c1,
        selected_child2_index=sel_c2,
        selected_parent_index=sel_p,
        selected_children_index=sel_ch,
        bdlo5=bdlo5,
        bdlo6=bdlo6,
        selected_child3_index=sel_c3,
    )
    eval_model = eval_model.double().to(args.device)
    # Skip adjacency_batch buffer (different batch size)
    state = {k: v for k, v in model.state_dict().items() if k != 'adjacency_batch'}
    eval_model.load_state_dict(state, strict=False)
    eval_model.eval()

    total_eval_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for eval_data in eval_loader:
            prev_traj, curr_traj, target_traj = eval_data
            batch_size = prev_traj.size(0)
            if batch_size != eval_n:
                continue  # skip incomplete batch

            # Build clamped hints
            inputs = torch.zeros_like(target_traj)
            if clamp_parent:
                inputs[:, :, 0, parent_clamped_selection] = target_traj[:, :, 0, parent_clamped_selection]

            traj_loss = 0.0
            curr = curr_traj[:, 0]
            prev = prev_traj[:, 0]

            for t in range(eval_time_horizon):
                target = target_traj[:, t]
                inp = inputs[:, t]

                curr_flat = curr.reshape(batch_size, -1, 3)
                prev_flat = prev.reshape(batch_size, -1, 3)
                inp_flat = inp.reshape(batch_size, -1, 3)

                x = torch.cat([curr_flat, prev_flat], dim=-1)
                pred_flat = eval_model.inference(x, inp_flat)

                target_flat = target.reshape(batch_size, -1, 3)
                mask = (
                    valid_mask.reshape(1, -1, 1)
                    .to(pred_flat.device, dtype=pred_flat.dtype)
                    .expand_as(pred_flat)
                )
                step_loss = loss_func(pred_flat * mask, target_flat * mask)
                traj_loss += step_loss.item()

                # Autoregressive
                prev = curr
                curr = pred_flat.reshape(batch_size, n_branch, n_parent, 3)

            total_eval_loss += traj_loss / eval_time_horizon
            n_batches += 1

    return total_eval_loss / max(n_batches, 1)


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(1)
    np.random.seed(1)

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    parser = argparse.ArgumentParser(description="GCN baseline for BDLO dynamics")

    parser.add_argument("--BDLO_type", type=int, default=1, choices=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--clamp_type", type=str, default="ends", choices=["ends", "middle"])
    parser.add_argument("--total_time", type=int, default=500)
    parser.add_argument("--train_time_horizon", type=int, default=50)
    parser.add_argument("--train_batch", type=int, default=32)

    parser.add_argument("--hidden_size", type=int, default=128)

    parser.add_argument("--train_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--evaluate_period", type=int, default=100)
    parser.add_argument(
        "--teacher_forcing",
        type=lambda x: x.lower() == "true",
        default=True,
    )
    parser.add_argument(
        "--detach_autoregressive",
        type=lambda x: x.lower() == "true",
        default=True,
    )

    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    train(args)
