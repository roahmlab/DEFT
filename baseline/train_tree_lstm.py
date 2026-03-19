"""
Training script for Tree-LSTM baseline on BDLO dynamics prediction.

Reuses the same dataset classes (Train_DEFTData, Eval_DEFTData) as DEFT training.
Supports both teacher-forced and autoregressive training.

Usage (run from baseline/ directory):
    python train_tree_lstm.py --BDLO_type 1 --clamp_type ends
    python train_tree_lstm.py --BDLO_type 3 --clamp_type middle --teacher_forcing false
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
from tree_lstm import BDLOTreeLSTM
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
    },
    (1, "middle"): {
        "n_parent": 13,
        "cs_n_vert": (5, 4),
        "coupling": [4, 8],
        "train_n": 71,
        "eval_n": 18,
        "parent_clamp": [2, -2, -1],
    },
    (2, "ends"): {
        "n_parent": 12,
        "cs_n_vert": (5, 5),
        "coupling": [4, 7],
        "train_n": 110,
        "eval_n": 37,
        "parent_clamp": [0, 1, -2, -1],
    },
    (3, "ends"): {
        "n_parent": 12,
        "cs_n_vert": (3, 4),
        "coupling": [4, 7],
        "train_n": 103,
        "eval_n": 26,
        "parent_clamp": [0, 1, -2, -1],
    },
    (3, "middle"): {
        "n_parent": 12,
        "cs_n_vert": (3, 4),
        "coupling": [4, 7],
        "train_n": 39,
        "eval_n": 12,
        "parent_clamp": [3, -2, -1],
    },
    (4, "ends"): {
        "n_parent": 12,
        "cs_n_vert": (3, 3),
        "coupling": [4, 8],
        "train_n": 74,
        "eval_n": 25,
        "parent_clamp": [0, 1, -2, -1],
    },
}


def create_valid_mask(n_parent_vertices, cs_n_vert, n_branch):
    """
    Create a binary mask for valid (non-padded) vertices.

    Returns:
        mask: [n_branch, n_parent_vertices] - 1.0 for valid vertices, 0.0 for padding
    """
    mask = torch.zeros(n_branch, n_parent_vertices)
    mask[0, :] = 1.0  # All parent vertices are valid
    for i, c_n in enumerate(cs_n_vert):
        mask[i + 1, :c_n] = 1.0
    return mask


def build_clamped_target_hints(target, parent_clamp_indices, n_branch, n_vert):
    """
    Build a tensor of target position hints for clamped vertices.
    Non-clamped vertices get zeros.

    Args:
        target: [batch, n_branch, n_vert, 3] - ground truth next positions
        parent_clamp_indices: list of int - clamped vertex indices in parent branch

    Returns:
        hints: [batch, n_branch, n_vert, 3]
    """
    hints = torch.zeros_like(target)
    for idx in parent_clamp_indices:
        hints[:, 0, idx] = target[:, 0, idx]
    return hints


def enforce_clamps(pred, target, parent_clamp_indices):
    """Overwrite clamped parent vertices with ground truth target positions."""
    pred = pred.clone()
    for idx in parent_clamp_indices:
        pred[:, 0, idx] = target[:, 0, idx]
    return pred


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
    n_branch = 3
    n_children = cs_n_vert

    # Valid vertex mask (excludes zero-padded child vertices)
    valid_mask = create_valid_mask(n_parent, cs_n_vert, n_branch)

    # Dataset key for path resolution
    if args.clamp_type == "ends":
        bdlo_key = args.BDLO_type
    else:
        bdlo_key = str(args.BDLO_type) + "_mid_clamp"

    # Load datasets
    print("Loading training data...")
    train_dataset = Train_DEFTData(
        bdlo_key,
        n_parent,
        n_children,
        n_branch,
        coupling,
        train_n,
        args.total_time,
        args.train_time_horizon,
        args.device,
    )
    print(f"Training samples: {len(train_dataset)}")

    print("Loading evaluation data...")
    eval_time_horizon = args.total_time - 2
    eval_dataset = Eval_DEFTData(
        bdlo_key,
        n_parent,
        n_children,
        n_branch,
        coupling,
        eval_n,
        args.total_time,
        eval_time_horizon,
        args.device,
    )
    print(f"Evaluation samples: {len(eval_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=args.train_batch, shuffle=True, drop_last=True
    )

    # Model
    model = BDLOTreeLSTM(
        hidden_size=args.hidden_size,
        n_parent_vertices=n_parent,
        cs_n_vert=cs_n_vert,
        rigid_body_coupling_index=coupling,
        input_size=9,  # position(3) + velocity(3) + clamped_target_hint(3)
    )
    model = model.double()  # Match DEFT's float64 precision
    model = model.to(args.device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_func = nn.MSELoss()

    # Training records
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

            # Evaluate periodically
            if iteration % args.evaluate_period == 0:
                model.eval()
                eval_loss = evaluate(
                    model,
                    eval_dataset,
                    eval_n,
                    n_parent,
                    cs_n_vert,
                    valid_mask,
                    parent_clamp,
                    eval_time_horizon,
                    args,
                )
                eval_losses.append(eval_loss)
                eval_epochs.append(iteration)
                print(f"\n[Iter {iteration}] Eval RMSE: {np.sqrt(eval_loss):.6f}")

                # Save model
                torch.save(
                    model.state_dict(),
                    os.path.join(
                        "../save_model/",
                        "TreeLSTM_%s_%s_%s_%s.pth"
                        % (args.clamp_type, args.BDLO_type, iteration, training_case),
                    ),
                )

                # Save records
                save_pickle(
                    eval_losses,
                    "../training_record/eval_%s_loss_TreeLSTM_%s_%s.pkl"
                    % (args.clamp_type, training_case, args.BDLO_type),
                )
                save_pickle(
                    eval_epochs,
                    "../training_record/eval_%s_epoches_TreeLSTM_%s_%s.pkl"
                    % (args.clamp_type, training_case, args.BDLO_type),
                )
                model.train()

            # --- Training step ---
            total_loss = 0.0

            # Initialize from ground truth
            prev = prev_traj[:, 0]  # [batch, n_branch, n_vert, 3]
            curr = curr_traj[:, 0]

            for t in range(args.train_time_horizon):
                target = target_traj[:, t]

                # Build clamped target hints (boundary conditions)
                hints = build_clamped_target_hints(
                    target, parent_clamp, n_branch, n_parent
                )

                # Forward pass
                pred = model(curr, prev, clamped_target_hints=hints)

                # Enforce clamp constraints on prediction
                pred = enforce_clamps(pred, target, parent_clamp)

                # Masked MSE loss (only valid vertices)
                mask = (
                    valid_mask.unsqueeze(0)
                    .unsqueeze(-1)
                    .to(pred.device, dtype=pred.dtype)
                    .expand_as(pred)
                )
                step_loss = loss_func(pred * mask, target * mask)
                total_loss += step_loss

                # Prepare next step inputs
                if args.teacher_forcing:
                    # Teacher forcing: use ground truth for next step
                    if t + 1 < args.train_time_horizon:
                        prev = prev_traj[:, t + 1]
                        curr = curr_traj[:, t + 1]
                else:
                    # Autoregressive: use own predictions
                    prev = curr
                    curr = pred.detach() if args.detach_autoregressive else pred

            # Backprop
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            avg_loss = total_loss.item() / args.train_time_horizon
            train_losses.append(avg_loss)
            train_steps.append(iteration)
            bar.set_postfix(loss=f"{avg_loss:.6f}")

            # Save training records
            save_pickle(
                train_losses,
                "../training_record/train_%s_loss_TreeLSTM_%s_%s.pkl"
                % (args.clamp_type, training_case, args.BDLO_type),
            )
            save_pickle(
                train_steps,
                "../training_record/train_%s_step_TreeLSTM_%s_%s.pkl"
                % (args.clamp_type, training_case, args.BDLO_type),
            )

            iteration += 1


def evaluate(
    model,
    eval_dataset,
    eval_n,
    n_parent,
    cs_n_vert,
    valid_mask,
    parent_clamp,
    eval_time_horizon,
    args,
):
    """
    Autoregressive evaluation over the full eval time horizon.

    Starts from ground truth initial frames, then rolls out using
    model predictions (no teacher forcing).
    """
    eval_loader = DataLoader(
        eval_dataset, batch_size=eval_n, shuffle=False, drop_last=True
    )
    loss_func = nn.MSELoss()
    n_branch = 3

    total_eval_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for eval_data in eval_loader:
            prev_traj, curr_traj, target_traj = eval_data

            traj_loss = 0.0
            prev = prev_traj[:, 0]
            curr = curr_traj[:, 0]

            for t in range(eval_time_horizon):
                target = target_traj[:, t]

                # Build clamped hints from GT target
                hints = build_clamped_target_hints(
                    target, parent_clamp, n_branch, n_parent
                )

                pred = model(curr, prev, clamped_target_hints=hints)
                pred = enforce_clamps(pred, target, parent_clamp)

                mask = (
                    valid_mask.unsqueeze(0)
                    .unsqueeze(-1)
                    .to(pred.device, dtype=pred.dtype)
                    .expand_as(pred)
                )
                step_loss = loss_func(pred * mask, target * mask)
                traj_loss += step_loss.item()

                # Autoregressive: use predictions for next step
                prev = curr
                curr = pred

            total_eval_loss += traj_loss / eval_time_horizon
            n_batches += 1

    return total_eval_loss / max(n_batches, 1)


if __name__ == "__main__":
    # Match DEFT's precision and reproducibility settings
    torch.set_default_dtype(torch.float64)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(1)
    np.random.seed(1)

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    parser = argparse.ArgumentParser(description="Tree-LSTM baseline for BDLO dynamics")

    # Data args (match DEFT_train.py)
    parser.add_argument("--BDLO_type", type=int, default=1, choices=[1, 2, 3, 4])
    parser.add_argument("--clamp_type", type=str, default="ends", choices=["ends", "middle"])
    parser.add_argument("--total_time", type=int, default=500)
    parser.add_argument("--train_time_horizon", type=int, default=50)
    parser.add_argument("--train_batch", type=int, default=32)

    # Model args
    parser.add_argument("--hidden_size", type=int, default=128)

    # Training args
    parser.add_argument("--train_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--evaluate_period", type=int, default=100)
    parser.add_argument(
        "--teacher_forcing",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Use teacher forcing during training (default: True)",
    )
    parser.add_argument(
        "--detach_autoregressive",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Detach predictions in autoregressive mode to prevent full BPTT (default: True)",
    )

    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    train(args)
