"""
Visualize BDLO6 trajectory data and the undeformed reference pose.

BDLO6 topology:
    parent  : 12 vertices
    child1  :  4 vertices, attaches at parent[2]   (3rd vertex)
    child2  :  2 vertices, attaches at parent[7]   (8th vertex)
    child3  :  3 vertices, attaches at parent[7]   (8th vertex)

Total stored vertices per frame: 12 + 4 + 2 + 3 = 21.

Unlike BDLO5, the BDLO6 dataset stores all 21 vertices independently
(child[0] is its own vertex near the parent attachment, not the shared
coupling vertex). So we don't need construct_BDLOs_data — we plot the
raw stored vertices directly and draw an explicit dashed line from each
child[0] back to its parent attachment vertex to make the topology
visible.

Outputs:
    visualizations/bdlo6_undeformed.mp4   (single static frame, just the rest pose)
    visualizations/bdlo6_train.mp4        (one sample training trajectory)
    visualizations/bdlo6_eval.mp4         (one sample eval trajectory)
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ---- BDLO6 topology ----
N_PARENT  = 12
N_CHILD1  = 4
N_CHILD2  = 2
N_CHILD3  = 3
N_TOTAL   = N_PARENT + N_CHILD1 + N_CHILD2 + N_CHILD3   # 21

# Slice ranges in the stored 21-vertex array
PARENT_SLICE = slice(0,                                  N_PARENT)
C1_SLICE     = slice(N_PARENT,                           N_PARENT + N_CHILD1)
C2_SLICE     = slice(N_PARENT + N_CHILD1,                N_PARENT + N_CHILD1 + N_CHILD2)
C3_SLICE     = slice(N_PARENT + N_CHILD1 + N_CHILD2,     N_TOTAL)

# Parent attachment indices for each child
C1_ATTACH = 2
C2_ATTACH = 7
C3_ATTACH = 7

TOTAL_TIME = 500   # timesteps per dataset pkl


def apply_bdlo5_coord_transform(verts):
    """
    BDLO5-style transform from DEFT_train.py composed with an extra +90° rotation
    around the world X axis (applied last):
        stage1 (DEFT_train.py): (x, y, z) -> (-z, -x,  y)
        stage2 (DEFT_train.py): (a, b, c) -> (-a,  c, -b)
        stage1+stage2 simplifies to: (x, y, z) -> (z, y, x)
        post-rotation Rx(+90°):     (x', y', z') -> (x', -z', y')
        full composed:              (x, y, z) -> (z, -x, y)
    `verts` may be [..., 3]; only the last axis is permuted.
    """
    out = np.empty_like(verts)
    out[..., 0] =  verts[..., 2]
    out[..., 1] = -verts[..., 0]
    out[..., 2] =  verts[..., 1]
    return out


def load_trajectory(pkl_path):
    """Load a BDLO6 .pkl. Returns array of shape [time, 21, 3] in the
    transformed (BDLO5-style) coordinate frame."""
    arr = np.asarray(pd.read_pickle(pkl_path))   # (3, time * 21)
    raw = arr.reshape(3, TOTAL_TIME, N_TOTAL).transpose(1, 2, 0)
    return apply_bdlo5_coord_transform(raw)


def load_undeformed(pkl_path='../dataset/BDLO6_undeformed.pkl'):
    """Load the undeformed reference. Returns array of shape [21, 3] in
    the transformed (BDLO5-style) coordinate frame."""
    arr = np.asarray(pd.read_pickle(pkl_path))   # (3, 21)
    return apply_bdlo5_coord_transform(arr.T)


def make_video(trajectory, output_path, title="BDLO6 Trajectory", fps=100):
    """
    Render a 3-view 3D animation of a BDLO6 trajectory.

    trajectory: [n_frames, 21, 3] numpy array. If n_frames == 1 the output is
                a single still frame video (useful for the undeformed pose).
    """
    n_frames = trajectory.shape[0]

    # Compute consistent global axis limits across the whole trajectory
    pts_all = trajectory.reshape(-1, 3)
    margin = 0.05
    mins = pts_all.min(0) - margin
    maxs = pts_all.max(0) + margin
    max_range = float((maxs - mins).max())
    mids = (mins + maxs) / 2
    xlim = (mids[0] - max_range / 2, mids[0] + max_range / 2)
    ylim = (mids[1] - max_range / 2, mids[1] + max_range / 2)
    zlim = (mids[2] - max_range / 2, mids[2] + max_range / 2)

    # 3 viewpoints
    views = [
        (25, -60, "View 1 (Front)"),
        (25,  30, "View 2 (Side)"),
        (80, -60, "View 3 (Top)"),
    ]

    # Branch styling
    branch_colors = {
        'parent': 'red',
        'c1':     'green',
        'c2':     'blue',
        'c3':     'magenta',
    }

    fig = plt.figure(figsize=(24, 7))
    axes = [fig.add_subplot(1, 3, i + 1, projection='3d') for i in range(3)]

    def draw_frame(frame_idx):
        verts = trajectory[frame_idx]    # [21, 3]
        parent = verts[PARENT_SLICE]
        c1 = verts[C1_SLICE]
        c2 = verts[C2_SLICE]
        c3 = verts[C3_SLICE]

        for ax_i, ax in enumerate(axes):
            ax.cla()
            ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_zlim(*zlim)
            ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
            elev, azim, view_title = views[ax_i]
            ax.view_init(elev=elev, azim=azim)
            ax.set_title(f'{view_title} - Frame {frame_idx}/{n_frames}')

            # Plot each rod as a polyline of its own vertices
            ax.plot(parent[:, 0], parent[:, 1], parent[:, 2],
                    '-o', color=branch_colors['parent'], markersize=3, linewidth=1.5,
                    label=f'parent ({N_PARENT}v)')
            ax.plot(c1[:, 0], c1[:, 1], c1[:, 2],
                    '-o', color=branch_colors['c1'], markersize=3, linewidth=1.5,
                    label=f'c1 ({N_CHILD1}v)')
            ax.plot(c2[:, 0], c2[:, 1], c2[:, 2],
                    '-o', color=branch_colors['c2'], markersize=3, linewidth=1.5,
                    label=f'c2 ({N_CHILD2}v)')
            ax.plot(c3[:, 0], c3[:, 1], c3[:, 2],
                    '-o', color=branch_colors['c3'], markersize=3, linewidth=1.5,
                    label=f'c3 ({N_CHILD3}v)')

            # Dashed lines from each child[0] back to its parent attachment,
            # so the topology is unambiguous in the visualization.
            for child_first, parent_idx, color in [
                (c1[0], C1_ATTACH, branch_colors['c1']),
                (c2[0], C2_ATTACH, branch_colors['c2']),
                (c3[0], C3_ATTACH, branch_colors['c3']),
            ]:
                p_attach = parent[parent_idx]
                ax.plot([p_attach[0], child_first[0]],
                        [p_attach[1], child_first[1]],
                        [p_attach[2], child_first[2]],
                        '--', color=color, linewidth=1, alpha=0.6)

            if ax_i == 0:
                ax.legend(loc='upper right', fontsize=8)

        fig.suptitle(title, fontsize=14)

    if n_frames == 1:
        # Static still: render once and save as a 1-frame mp4 (or fall back to png)
        draw_frame(0)
        png_path = output_path.replace('.mp4', '.png')
        fig.savefig(png_path, dpi=100)
        print(f"Saved still: {png_path}")
        plt.close(fig)
        return

    anim = FuncAnimation(fig, draw_frame, frames=n_frames, interval=1000 // fps)
    anim.save(output_path, writer='ffmpeg', fps=fps, dpi=100)
    plt.close(fig)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs("../visualizations", exist_ok=True)

    # 1) Render the undeformed reference pose as a short looping mp4
    # (replicate the single rest pose across a few frames so it shows up as
    #  a video file in visualizations/ alongside the train/eval clips).
    undeformed = load_undeformed()
    undeformed_loop = np.broadcast_to(undeformed[None], (100, 21, 3)).copy()
    make_video(undeformed_loop,
               "../visualizations/bdlo6_undeformed.mp4",
               title="BDLO6 Undeformed reference pose")

    # 2) One training trajectory
    train_files = sorted(glob.glob("../dataset/BDLO6/train/*.pkl"))
    if train_files:
        traj = load_trajectory(train_files[0])
        make_video(traj, "../visualizations/bdlo6_train.mp4",
                   title=f"BDLO6 Train - {os.path.basename(train_files[0])}")

    # 3) One eval trajectory
    eval_files = sorted(glob.glob("../dataset/BDLO6/eval/*.pkl"))
    if eval_files:
        traj = load_trajectory(eval_files[0])
        make_video(traj, "../visualizations/bdlo6_eval.mp4",
                   title=f"BDLO6 Eval - {os.path.basename(eval_files[0])}")
