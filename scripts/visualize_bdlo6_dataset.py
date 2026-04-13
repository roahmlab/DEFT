"""
Visualize one BDLO6 trajectory loaded *through the new dataset class*
(`Eval_DEFTData_BDLO6`) — verifies that the constructed dataset produces
sensible geometry end-to-end (raw .pkl → coord transform → padded layout).

Outputs:
    visualizations/bdlo6_dataset_eval.mp4
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from deft.utils.util import Eval_DEFTData_BDLO6


# BDLO6 topology (assembled lengths after BDLO1–5-style prepend)
N_PARENT  = 12
N_C1      = 5    # 4 raw + 1 prepended parent[2]
N_C2      = 3    # 2 raw + 1 prepended parent[7]
N_C3      = 4    # 3 raw + 1 prepended parent[7]
N_BRANCH  = 4

# Parent attach indices for the dashed connection lines
ATTACH = [None, 2, 7, 7]   # parent[*] for c1, c2, c3

BRANCH_COLORS = ['red', 'green', 'blue', 'magenta']
BRANCH_NAMES  = ['parent', 'c1', 'c2', 'c3']
BRANCH_NV     = [N_PARENT, N_C1, N_C2, N_C3]


def make_video_padded(traj_padded, output_path, title="BDLO6", fps=100,
                      draw_attach_lines=True):
    """
    traj_padded: torch.Tensor or ndarray of shape [n_frames, 4, 12, 3] in
                 the padded layout produced by Eval_DEFTData_BDLO6.

    Each branch's first `BRANCH_NV[b]` vertices are real; the rest are
    zero-padding and are skipped at plot time.
    """
    if hasattr(traj_padded, 'numpy'):
        traj = traj_padded.numpy()
    else:
        traj = np.asarray(traj_padded)
    n_frames = traj.shape[0]
    assert traj.shape[1:] == (N_BRANCH, N_PARENT, 3), f"unexpected shape {traj.shape}"

    # Compute a single global axis range from the active vertices across all frames
    pts = np.concatenate(
        [traj[:, b, :BRANCH_NV[b]].reshape(-1, 3) for b in range(N_BRANCH)],
        axis=0,
    )
    margin = 0.05
    mins = pts.min(0) - margin
    maxs = pts.max(0) + margin
    max_range = float((maxs - mins).max())
    mids = (mins + maxs) / 2
    xlim = (mids[0] - max_range / 2, mids[0] + max_range / 2)
    ylim = (mids[1] - max_range / 2, mids[1] + max_range / 2)
    zlim = (mids[2] - max_range / 2, mids[2] + max_range / 2)

    # Single 3D plot at matplotlib's default 3D view (elev=30, azim=-60), so it
    # matches what DEFT_train.py's undeform_vis pops up by default.
    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(111, projection='3d')

    def draw_frame(frame_idx):
        verts = traj[frame_idx]   # [4, 12, 3]
        ax.cla()
        ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_zlim(*zlim)
        ax.set_box_aspect((1, 1, 1))
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.view_init(elev=30, azim=-60)   # matplotlib 3D default
        ax.set_title(f'{title} (frame {frame_idx})')

        for b in range(N_BRANCH):
            pts_b = verts[b, :BRANCH_NV[b]]
            ax.plot(pts_b[:, 0], pts_b[:, 1], pts_b[:, 2], '-o',
                    color=BRANCH_COLORS[b],
                    label=f'{BRANCH_NAMES[b]} ({BRANCH_NV[b]}v)',
                    markersize=4, linewidth=1.5)

        if draw_attach_lines:
            for b in range(1, N_BRANCH):
                cp = verts[b, 0]
                pp = verts[0, ATTACH[b]]
                ax.plot([pp[0], cp[0]], [pp[1], cp[1]], [pp[2], cp[2]],
                        '--', color=BRANCH_COLORS[b], linewidth=1, alpha=0.6)

        ax.legend(loc='upper right', fontsize=9)

    if n_frames == 1:
        # Render a single still frame as a PNG
        draw_frame(0)
        png_path = output_path.replace('.mp4', '.png')
        fig.savefig(png_path, dpi=100)
        plt.close(fig)
        print(f"Saved still: {png_path}")
        return

    anim = FuncAnimation(fig, draw_frame, frames=n_frames, interval=1000 // fps)
    anim.save(output_path, writer='ffmpeg', fps=fps, dpi=100)
    plt.close(fig)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs("../visualizations", exist_ok=True)

    # Load the BDLO6 eval split via the new dataset class.
    # Use a short horizon so we don't sit through the full 498-frame trajectory.
    eval_ds = Eval_DEFTData_BDLO6(total_time=500, eval_time_horizon=498, device="cpu")
    print(f"loaded {len(eval_ds)} eval samples")

    # Each item is (previous, current, target). We render the "current" stream
    # of sample 0 — that's the trajectory the sim would predict against.
    _prev, current, _target = eval_ds[0]
    print(f"sample 0 'current' shape: {tuple(current.shape)}  (expect (498, 4, 12, 3))")

    # Single still frame, no dashed attach lines (so we can confirm child[0]
    # truly coincides with parent[attach] in the constructed dataset).
    make_video_padded(current[:1],
                      "../visualizations/bdlo6_dataset_eval.mp4",
                      title="BDLO6 Eval frame 0 — Eval_DEFTData_BDLO6",
                      draw_attach_lines=False)
