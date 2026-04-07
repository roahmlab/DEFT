"""
Visualize BDLO5 trajectory data after preprocessing (same pipeline as Train/Eval_DEFTData).
Generates mp4 videos for one training and one eval sequence.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pandas as pd
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from deft.utils.util import construct_BDLOs_data

# BDLO5 parameters
n_parent_vertices = 12
n_child1_vertices = 4
n_child2_vertices = 4
n_children_vertices = (n_child1_vertices, n_child2_vertices)
n_branch = 3
rigid_body_coupling_index = [5, 1]
bdlo5 = True
total_time = 500


def load_and_preprocess(pkl_path, apply_coord_transform=True):
    """Load a pickle file and run the same preprocessing as Train_DEFTData."""
    verts = torch.tensor(pd.read_pickle(pkl_path)).view(3, total_time, -1).permute(1, 2, 0)

    parent_vertices = verts[:, :n_parent_vertices]
    child1_vertices = verts[:, n_parent_vertices: n_parent_vertices + n_child1_vertices - 1]
    child2_vertices = verts[:, n_parent_vertices + n_child1_vertices - 1:]

    BDLO_vert_no_trans = construct_BDLOs_data(
        total_time, rigid_body_coupling_index,
        n_parent_vertices, n_children_vertices,
        n_branch, parent_vertices, child1_vertices, child2_vertices,
        bdlo5=bdlo5
    )

    if apply_coord_transform:
        BDLO_vert = torch.zeros_like(BDLO_vert_no_trans)
        BDLO_vert[:, :, :, 0] = -BDLO_vert_no_trans[:, :, :, 2]
        BDLO_vert[:, :, :, 1] = -BDLO_vert_no_trans[:, :, :, 0]
        BDLO_vert[:, :, :, 2] = BDLO_vert_no_trans[:, :, :, 1]
    else:
        BDLO_vert = BDLO_vert_no_trans

    return BDLO_vert  # shape: [total_time, n_branch, n_parent_vertices, 3]


def make_video(BDLO_vert, output_path, title="BDLO5 Trajectory", fps=100):
    """
    Generate an mp4 from preprocessed BDLO data with 3 different views.
    BDLO_vert: [total_time, n_branch, n_vert, 3]
    """
    n_frames = BDLO_vert.shape[0]
    colors = ['red', 'green', 'blue']
    branch_names = ['Parent', 'Child1', 'Child2']
    branch_n_verts = [n_parent_vertices, n_child1_vertices, n_child2_vertices]

    # Compute global axis limits from all frames
    all_points = []
    for b in range(n_branch):
        nv = branch_n_verts[b]
        all_points.append(BDLO_vert[:, b, :nv, :].reshape(-1, 3).numpy())
    all_points = np.concatenate(all_points, axis=0)
    margin = 0.05
    x_min, x_max = all_points[:, 0].min() - margin, all_points[:, 0].max() + margin
    y_min, y_max = all_points[:, 1].min() - margin, all_points[:, 1].max() + margin
    z_min, z_max = all_points[:, 2].min() - margin, all_points[:, 2].max() + margin

    # Make all axes the same scale
    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
    x_mid = (x_min + x_max) / 2
    y_mid = (y_min + y_max) / 2
    z_mid = (z_min + z_max) / 2
    x_min, x_max = x_mid - max_range / 2, x_mid + max_range / 2
    y_min, y_max = y_mid - max_range / 2, y_mid + max_range / 2
    z_min, z_max = z_mid - max_range / 2, z_mid + max_range / 2

    # 3 views: (elev, azim)
    views = [
        (25, -60, "View 1 (Front)"),
        (25, 30, "View 2 (Side)"),
        (80, -60, "View 3 (Top)"),
    ]

    fig = plt.figure(figsize=(24, 7))
    axes = []
    for i, (elev, azim, view_title) in enumerate(views):
        ax = fig.add_subplot(1, 3, i + 1, projection='3d')
        axes.append(ax)

    def update(frame):
        for i, ax in enumerate(axes):
            ax.cla()
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_zlim(z_min, z_max)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            elev, azim, view_title = views[i]
            ax.view_init(elev=elev, azim=azim)
            ax.set_title(f'{view_title} - Frame {frame}/{n_frames}')

            for b in range(n_branch):
                nv = branch_n_verts[b]
                pts = BDLO_vert[frame, b, :nv, :].numpy()
                ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                        '-o', color=colors[b], markersize=3, linewidth=1.5,
                        label=branch_names[b])
            if i == 0:
                ax.legend(loc='upper right', fontsize=8)
        fig.suptitle(title, fontsize=14)

    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 // fps)
    anim.save(output_path, writer='ffmpeg', fps=fps, dpi=100)
    plt.close(fig)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Pick one training and one eval file
    train_files = sorted(glob.glob("../dataset/BDLO5/train/*.pkl"))
    eval_files = sorted(glob.glob("../dataset/BDLO5/eval/*.pkl"))

    os.makedirs("../visualizations", exist_ok=True)

    if train_files:
        print(f"Loading training file: {os.path.basename(train_files[0])}")
        train_data = load_and_preprocess(train_files[0])
        print(f"  Shape after preprocessing: {train_data.shape}")
        print(f"  X range: [{train_data[:,:,:,0].min():.3f}, {train_data[:,:,:,0].max():.3f}]")
        print(f"  Y range: [{train_data[:,:,:,1].min():.3f}, {train_data[:,:,:,1].max():.3f}]")
        print(f"  Z range: [{train_data[:,:,:,2].min():.3f}, {train_data[:,:,:,2].max():.3f}]")
        make_video(train_data, "../visualizations/bdlo5_train.mp4",
                   title=f"BDLO5 Train - {os.path.basename(train_files[0])}")

    if eval_files:
        print(f"Loading eval file: {os.path.basename(eval_files[0])}")
        eval_data = load_and_preprocess(eval_files[0])
        print(f"  Shape after preprocessing: {eval_data.shape}")
        make_video(eval_data, "../visualizations/bdlo5_eval.mp4",
                   title=f"BDLO5 Eval - {os.path.basename(eval_files[0])}")
