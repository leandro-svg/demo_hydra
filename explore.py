# kitti_exploration.py
# KITTI Dataset Exploration Script (Standalone)

import os
import numpy as np
import pykitti
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from moviepy.editor import ImageSequenceClip
from source import parseTrackletXML as xmlParser
from source.utilities import print_progress

# Directory where KITTI data is stored
basedir = 'data'

def load_dataset(date, drive, calibrated=False, frame_range=None):
    dataset = pykitti.raw(basedir, date, drive)

    if calibrated:
        dataset.load_calib()

    np.set_printoptions(precision=4, suppress=True)
    print('\nDrive:', dataset.drive)
    print('\nFrame range:', dataset.frames)

    if calibrated:
        print('\nIMU-to-Velodyne transformation:\n', dataset.calib.T_velo_imu)
        print('\nGray stereo pair baseline [m]:', dataset.calib.b_gray)
        print('\nRGB stereo pair baseline [m]:', dataset.calib.b_rgb)

    return dataset

def load_tracklets_for_frames(n_frames, xml_path):
    tracklets = xmlParser.parseXML(xml_path)
    frame_tracklets = {}
    frame_tracklets_types = {}
    for i in range(n_frames):
        frame_tracklets[i] = []
        frame_tracklets_types[i] = []

    for tracklet in tracklets:
        h, w, l = tracklet.size
        trackletBox = np.array([
            [-l/2, -l/2,  l/2,  l/2, -l/2, -l/2,  l/2,  l/2],
            [ w/2, -w/2, -w/2,  w/2,  w/2, -w/2, -w/2,  w/2],
            [ 0.0,  0.0,  0.0,  0.0,  h,    h,    h,    h  ]
        ])

        for translation, rotation, state, occlusion, truncation, amtOcclusion, amtBorders, absFrameNum in tracklet:
            if truncation not in (xmlParser.TRUNC_IN_IMAGE, xmlParser.TRUNC_TRUNCATED):
                continue

            yaw = rotation[2]
            assert np.abs(rotation[:2]).sum() == 0, 'Invalid object rotation!'

            rotMat = np.array([
                [np.cos(yaw), -np.sin(yaw), 0.0],
                [np.sin(yaw), np.cos(yaw), 0.0],
                [0.0, 0.0, 1.0]
            ])
            cornerPosInVelo = np.dot(rotMat, trackletBox) + np.tile(translation, (8, 1)).T
            frame_tracklets[absFrameNum].append(cornerPosInVelo)
            frame_tracklets_types[absFrameNum].append(tracklet.objectType)

    return frame_tracklets, frame_tracklets_types

colors = {
    'Car': 'b',
    'Tram': 'r',
    'Cyclist': 'g',
    'Van': 'c',
    'Truck': 'm',
    'Pedestrian': 'y',
    'Sitter': 'k'
}

axes_limits = [[-20, 80], [-20, 20], [-3, 10]]
axes_str = ['X', 'Y', 'Z']

def draw_box(ax, vertices, axes=[0, 1, 2], color='black'):
    vertices = vertices[axes, :]
    connections = [
        [0,1],[1,2],[2,3],[3,0],
        [4,5],[5,6],[6,7],[7,4],
        [0,4],[1,5],[2,6],[3,7]
    ]
    for connection in connections:
        ax.plot(*vertices[:, connection], c=color, lw=0.5)

def display_frame_statistics(dataset, tracklet_rects, tracklet_types, frame, points=0.2):
    dataset_gray = list(dataset.gray)
    dataset_rgb = list(dataset.rgb)
    dataset_velo = list(dataset.velo)
    print('Frame timestamp:', dataset.timestamps[frame])

    f, ax = plt.subplots(2, 2, figsize=(15, 5))
    ax[0,0].imshow(dataset_gray[frame][0], cmap='gray')
    ax[0,0].set_title('Left Gray Image (cam0)')
    ax[0,1].imshow(dataset_gray[frame][1], cmap='gray')
    ax[0,1].set_title('Right Gray Image (cam1)')
    ax[1,0].imshow(dataset_rgb[frame][0])
    ax[1,0].set_title('Left RGB Image (cam2)')
    ax[1,1].imshow(dataset_rgb[frame][1])
    ax[1,1].set_title('Right RGB Image (cam3)')
    plt.show()

    points_step = int(1. / points)
    point_size = 0.01 * (1. / points)
    velo_range = range(0, dataset_velo[frame].shape[0], points_step)
    velo_frame = dataset_velo[frame][velo_range, :]

    def draw_point_cloud(ax, title, axes=[0,1,2]):
        ax.scatter(*np.transpose(velo_frame[:, axes]), s=point_size, c=velo_frame[:, 3], cmap='gray')
        ax.set_title(title)
        ax.set_xlabel(f'{axes_str[axes[0]]} axis')
        ax.set_ylabel(f'{axes_str[axes[1]]} axis')
        if len(axes) > 2:
            ax.set_xlim3d(*axes_limits[axes[0]])
            ax.set_ylim3d(*axes_limits[axes[1]])
            ax.set_zlim3d(*axes_limits[axes[2]])
            ax.set_zlabel(f'{axes_str[axes[2]]} axis')
        else:
            ax.set_xlim(*axes_limits[axes[0]])
            ax.set_ylim(*axes_limits[axes[1]])
        for t_rects, t_type in zip(tracklet_rects[frame], tracklet_types[frame]):
            draw_box(ax, t_rects, axes=axes, color=colors[t_type])

    # 3D view
    f2 = plt.figure(figsize=(15,8))
    ax2 = f2.add_subplot(111, projection='3d')
    draw_point_cloud(ax2, 'Velodyne scan')
    plt.show()

    # Plane projections
    f, ax3 = plt.subplots(3, 1, figsize=(15, 25))
    draw_point_cloud(ax3[0], 'XZ projection (Y=0)', axes=[0,2])
    draw_point_cloud(ax3[1], 'XY projection (Z=0)', axes=[0,1])
    draw_point_cloud(ax3[2], 'YZ projection (X=0)', axes=[1,2])
    plt.show()

def draw_3d_plot(frame, dataset, tracklet_rects, tracklet_types, points=0.2):
    dataset_velo = list(dataset.velo)
    f = plt.figure(figsize=(12,8))
    axis = f.add_subplot(111, projection='3d', xticks=[], yticks=[], zticks=[])

    points_step = int(1. / points)
    point_size = 0.01 * (1. / points)
    velo_range = range(0, dataset_velo[frame].shape[0], points_step)
    velo_frame = dataset_velo[frame][velo_range, :]
    axis.scatter(*np.transpose(velo_frame[:, [0,1,2]]), s=point_size, c=velo_frame[:, 3], cmap='gray')
    axis.set_xlim3d(*axes_limits[0])
    axis.set_ylim3d(*axes_limits[1])
    axis.set_zlim3d(*axes_limits[2])
    for t_rects, t_type in zip(tracklet_rects[frame], tracklet_types[frame]):
        draw_box(axis, t_rects, axes=[0,1,2], color=colors[t_type])
    filename = f'video/frame_{frame:04d}.png'
    os.makedirs('video', exist_ok=True)
    plt.savefig(filename)
    plt.close(f)
    return filename

if __name__ == "__main__":
    date = '2011_09_26'
    drive = '0048'
    dataset = load_dataset(date, drive)
    xml_path = f'data/{date}/{date}_drive_{drive}_sync/tracklet_labels.xml'
    tracklet_rects, tracklet_types = load_tracklets_for_frames(len(list(dataset.velo)), xml_path)

    frame = 10
    display_frame_statistics(dataset, tracklet_rects, tracklet_types, frame)

    print('Preparing animation frames...')
    frames = []
    n_frames = len(list(dataset.velo))
    for i in range(n_frames):
        print_progress(i, n_frames - 1)
        filename = draw_3d_plot(i, dataset, tracklet_rects, tracklet_types)
        frames.append(filename)
    print('...Animation frames ready.')

    clip = ImageSequenceClip(frames, fps=5)
    clip.write_gif('pcl_data.gif', fps=5)
    
