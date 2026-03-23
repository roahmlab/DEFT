"""
Script to read and visualize the thread insertion pickle data.
Visualizes all mocap configs and height data from BDLO_main_branch_thread_insertion.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import os
import argparse

DATASET_DIR = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'BDLO_main_branch_thread_insertion')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'visualization_output')

# Wire connections (1-indexed) for 20-vertex BDLO
CONNECTIONS_20 = [
    {'points': [1, 2, 3, 4, 5], 'color': 'red'},
    {'points': [5, 6, 7, 8, 9], 'color': 'red'},
    {'points': [9, 10, 11, 12, 13], 'color': 'red'},
    {'points': [5, 14, 15, 16, 17], 'color': 'blue'},
    {'points': [9, 18, 19, 20], 'color': 'blue'}
]

# For 8-vertex height data, just connect all points sequentially
CONNECTIONS_8 = [
    {'points': list(range(1, 9)), 'color': 'green'}
]


def load_pkl(filepath, n_points=20):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    bdlo_data = np.array(data).squeeze()
    bdlo_data = bdlo_data.T.reshape(-1, n_points, 3)
    return bdlo_data


def set_equal_axes_3d(ax):
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    center = limits.mean(axis=1)
    span = (limits[:, 1] - limits[:, 0]).max() / 2
    for set_lim, c in zip([ax.set_xlim3d, ax.set_ylim3d, ax.set_zlim3d], center):
        set_lim([c - span, c + span])


def compute_axis_limits(all_bdlo_data_list):
    """Compute global axis limits from a list of bdlo_data arrays."""
    all_points = np.concatenate([d.reshape(-1, 3) for d in all_bdlo_data_list], axis=0)
    valid_mask = np.isfinite(all_points).all(axis=1)
    valid_points = all_points[valid_mask]

    if valid_points.shape[0] > 0:
        mins = valid_points.min(axis=0)
        maxs = valid_points.max(axis=0)
    else:
        mins = np.array([-1, -1, -1])
        maxs = np.array([1, 1, 1])

    padding = 0.05
    pad = (maxs - mins) * padding
    return {
        'x': (mins[0] - pad[0], maxs[0] + pad[0]),
        'y': (mins[1] - pad[1], maxs[1] + pad[1]),
        'z': (mins[2] - pad[2], maxs[2] + pad[2])
    }


def plot_frame(ax, data_chunk, connections, axis_limits, title=None):
    """Plot a single frame on a given axes."""
    ax.scatter(data_chunk[:, 0], data_chunk[:, 1], data_chunk[:, 2], color='black', s=20)

    for conn in connections:
        points = np.array(conn['points']) - 1
        ax.plot(data_chunk[points, 0], data_chunk[points, 1], data_chunk[points, 2],
                color=conn['color'], linewidth=2)

    ax.set_xlim(axis_limits['x'])
    ax.set_ylim(axis_limits['y'])
    ax.set_zlim(axis_limits['z'])
    ax.view_init(elev=20, azim=-60, roll=0)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    set_equal_axes_3d(ax)
    if title:
        ax.set_title(title)


def save_frame_image(data_chunk, connections, save_path, axis_limits, frame_idx=None, total_frames=None):
    """Save a single frame as image."""
    if not np.isfinite(data_chunk).all():
        print(f"  Skipping frame {frame_idx} - contains NaN/Inf")
        return False

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    title = f'Frame {frame_idx}/{total_frames}' if frame_idx is not None else None
    plot_frame(ax, data_chunk, connections, axis_limits, title=title)

    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    return True


def create_video(list_of_data, connections, output_path, axis_limits, fps=30, step=5):
    """Create video by saving frames and combining with cv2."""
    frames_dir = os.path.join(OUTPUT_DIR, "temp_frames")
    os.makedirs(frames_dir, exist_ok=True)

    frame_paths = []
    n_frames = len(list_of_data)
    total_video_frames = len(range(0, n_frames, step))

    print(f"Saving {total_video_frames} frames...")

    for idx, i in enumerate(range(0, n_frames, step)):
        frame_path = os.path.join(frames_dir, f"frame_{i:05d}.png")
        success = save_frame_image(list_of_data[i], connections, frame_path, axis_limits,
                                   frame_idx=i, total_frames=n_frames)
        if success:
            frame_paths.append(frame_path)

        if idx % 50 == 0:
            print(f"  Frame {idx}/{total_video_frames}")

    if len(frame_paths) == 0:
        print("No valid frames to create video!")
        return

    print(f"Combining {len(frame_paths)} frames into video...")

    first_frame = cv2.imread(frame_paths[0])
    height, width, _ = first_frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame_path in frame_paths:
        frame = cv2.imread(frame_path)
        video.write(frame)

    video.release()

    # Clean up temp frames
    for frame_path in frame_paths:
        os.remove(frame_path)
    os.rmdir(frames_dir)

    print(f"Video saved to: {output_path}")


def plot_combination(ax, bdlo_pts, hole_pts, connections, axis_limits, title=None):
    """Plot a BDLO config with a target hole overlaid."""
    # Plot BDLO wire
    ax.scatter(bdlo_pts[:, 0], bdlo_pts[:, 1], bdlo_pts[:, 2], color='black', s=15, alpha=0.8)
    for conn in connections:
        points = np.array(conn['points']) - 1
        ax.plot(bdlo_pts[points, 0], bdlo_pts[points, 1], bdlo_pts[points, 2],
                color=conn['color'], linewidth=1.5)

    # Plot hole as triangle (first 3 points form triangle, 4th could be center)
    triangle_idx = [0, 1, 2, 0]  # close the triangle
    ax.plot(hole_pts[triangle_idx, 0], hole_pts[triangle_idx, 1], hole_pts[triangle_idx, 2],
            'g^-', color='green', linewidth=2, markersize=8)
    # Plot the 4th point (center or additional vertex)
    if len(hole_pts) >= 4:
        ax.scatter(*hole_pts[3], color='lime', s=60, marker='*', edgecolors='black', zorder=5)

    ax.set_xlim(axis_limits['x'])
    ax.set_ylim(axis_limits['y'])
    ax.set_zlim(axis_limits['z'])
    ax.view_init(elev=20, azim=-60, roll=0)
    ax.set_xlabel('X', fontsize=7)
    ax.set_ylabel('Y', fontsize=7)
    ax.set_zlabel('Z', fontsize=7)
    ax.tick_params(labelsize=6)
    set_equal_axes_3d(ax)
    if title:
        ax.set_title(title, fontsize=8)


def main():
    parser = argparse.ArgumentParser(description='Visualize BDLO thread insertion data in 3D')
    parser.add_argument('--save', action='store_true', help='Save figures instead of showing')
    parser.add_argument('--video', action='store_true', help='Also create videos for each config')
    parser.add_argument('--fps', type=int, default=30, help='Video FPS')
    parser.add_argument('--step', type=int, default=20, help='Frame step for video')
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load 5 initial configs
    configs = {}
    for i in range(1, 6):
        fname = f'mocap_in_base_config{i}.pkl'
        fpath = os.path.join(DATASET_DIR, fname)
        if os.path.exists(fpath):
            configs[f'Config {i}'] = load_pkl(fpath)[0]  # (20, 3) - frame 0
            print(f'{fname}: {configs[f"Config {i}"].shape}')

    # Load 8 target holes: 4 height files × 2 holes each (points 0-3 = hole A, points 4-7 = hole B)
    holes = {}
    for i in range(1, 5):
        fname = f'height_{i}_0319_in_base.pkl'
        fpath = os.path.join(DATASET_DIR, fname)
        if os.path.exists(fpath):
            pts = load_pkl(fpath, n_points=8)[0]  # (8, 3)
            holes[f'H{i}A'] = pts[:4]   # hole A (4 points)
            holes[f'H{i}B'] = pts[4:]   # hole B (4 points)
            print(f'{fname}: hole A center={pts[:4].mean(0).round(3)}, hole B center={pts[4:].mean(0).round(3)}')

    print(f'\n{len(configs)} configs x {len(holes)} holes = {len(configs) * len(holes)} combinations')

    # Compute global axis limits
    all_pts_list = [c.reshape(-1, 3) for c in configs.values()] + [h.reshape(-1, 3) for h in holes.values()]
    all_pts = np.concatenate(all_pts_list, axis=0)
    valid = all_pts[np.isfinite(all_pts).all(axis=1)]
    mins, maxs = valid.min(0), valid.max(0)
    pad = (maxs - mins) * 0.05
    axis_limits = {
        'x': (mins[0] - pad[0], maxs[0] + pad[0]),
        'y': (mins[1] - pad[1], maxs[1] + pad[1]),
        'z': (mins[2] - pad[2], maxs[2] + pad[2])
    }

    # --- 40 combinations: 5 configs (rows) × 8 holes (cols) ---
    config_names = list(configs.keys())
    hole_names = list(holes.keys())
    nrows = len(config_names)  # 5
    ncols = len(hole_names)    # 8

    fig = plt.figure(figsize=(4 * ncols, 3.5 * nrows))
    fig.suptitle('Thread Insertion: 5 Configs × 8 Target Holes = 40 Combinations', fontsize=14, y=0.98)

    for row, cfg_name in enumerate(config_names):
        for col, hole_name in enumerate(hole_names):
            ax = fig.add_subplot(nrows, ncols, row * ncols + col + 1, projection='3d')
            plot_combination(ax, configs[cfg_name], holes[hole_name],
                             CONNECTIONS_20, axis_limits,
                             title=f'{cfg_name} → {hole_name}')

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if args.save:
        out_path = os.path.join(OUTPUT_DIR, '40_combinations.png')
        plt.savefig(out_path, dpi=150)
        print(f'\nSaved to {out_path}')
    else:
        plt.show()

    # --- Also save X-Z plane view of all 40 combinations ---
    fig2, axes2 = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    fig2.suptitle('Thread Insertion X-Z View: 5 Configs × 8 Target Holes', fontsize=14, y=0.98)

    for row, cfg_name in enumerate(config_names):
        for col, hole_name in enumerate(hole_names):
            ax = axes2[row, col]
            bdlo_pts = configs[cfg_name]
            hole_pts = holes[hole_name]

            # Plot BDLO in X-Z
            ax.scatter(bdlo_pts[:, 0], bdlo_pts[:, 2], color='black', s=10)
            for conn in CONNECTIONS_20:
                pts = np.array(conn['points']) - 1
                ax.plot(bdlo_pts[pts, 0], bdlo_pts[pts, 2], color=conn['color'], linewidth=1.5)

            # Plot hole triangle in X-Z
            tri = [0, 1, 2, 0]
            ax.plot(hole_pts[tri, 0], hole_pts[tri, 2], 'g-', linewidth=2)
            ax.scatter(hole_pts[:3, 0], hole_pts[:3, 2], color='green', s=40, marker='^', zorder=5)
            if len(hole_pts) >= 4:
                ax.scatter(hole_pts[3, 0], hole_pts[3, 2], color='lime', s=50, marker='*',
                           edgecolors='black', zorder=5)

            ax.set_title(f'{cfg_name}→{hole_name}', fontsize=7)
            ax.set_aspect('equal')
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.tick_params(labelsize=5)
            if row == nrows - 1:
                ax.set_xlabel('X', fontsize=7)
            if col == 0:
                ax.set_ylabel('Z', fontsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if args.save:
        xz_path = os.path.join(OUTPUT_DIR, '40_combinations_xz.png')
        plt.savefig(xz_path, dpi=150)
        print(f'Saved to {xz_path}')
    else:
        plt.show()


if __name__ == '__main__':
    main()
