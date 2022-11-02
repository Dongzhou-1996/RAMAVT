import sys

import numpy as np
import os
import glob
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as mticker


def trajectory_3d_ploting(lines={}, save_path=''):
    fig = plt.figure()
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    ax.view_init(30, -70)
    # ax.axis([-1600, 800, -800, 200])
    # ax.set_zlim(1500, 3300)
    plt.ion()
    # cmap = plt.cm.get_cmap('Set1', len(lines))
    cmap = ['red', 'green']
    linestyles = ['-', '--', ':']
    params = {'font.family': 'serif',
              'font.serif': 'Times New Roman',
              'font.style': 'normal',
              'font.weight': 'normal',  # or 'blod'
              }
    matplotlib.rcParams.update(params)
    for k, key in enumerate(lines.keys()):
        label_value = key
        tracklet_display(ax, lines[key], color=cmap[k], linestyle=linestyles[k],
                         alpha=0.7, marker='.', label=label_value)
    # ax.grid(True)
    # plt.axis('off')
    matplotlib.rcParams.update({'font.size': 7.4})
    plt.legend(loc='upper right')
    # 添加坐标轴(顺序是Z, Y, X)
    ax.tick_params(labelsize=8, pad=7)
    # ax.set_xticks(np.arange(-1600, 800, 400))
    # ax.set_yticks(np.arange(-700, 200, 200))
    # ax.set_zticks(np.arange(1800, 3200, 400))
    labels = ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    ax.set_title('3D trajectories plots', fontdict={'family': 'Times New Roman', 'weight': 'normal', 'size': 14, 'color': 'black'})
    ax.set_zlabel('Z', fontdict={'family': 'Times New Roman', 'weight': 'normal', 'size': 12, 'color': 'blue'}, labelpad=10)
    ax.set_ylabel('Y', fontdict={'family': 'Times New Roman', 'weight': 'normal', 'size': 12, 'color': 'blue'}, labelpad=10)
    ax.set_xlabel('X', fontdict={'family': 'Times New Roman', 'weight': 'normal', 'size': 12, 'color': 'blue'}, labelpad=8)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%d m'))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%d m'))
    ax.zaxis.set_major_formatter(mticker.FormatStrFormatter('%d m'))

    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

    plt.ioff()
    # plt.show()
    print('Saving success plots to', save_path)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')


def tracklet_display(ax, tracklet, color='r', alpha=0.5, marker='o', marker_size=2, label='', linestyle ='-'):
    x = tracklet[::8, 0]
    y = tracklet[::8, 1]
    z = tracklet[::8, 2]
    ax.plot(x, y, z, c=color, alpha=alpha, markersize=marker_size, label=label, linestyle=linestyle)
    ax.scatter(x[0], y[0], z[0], s=25, c='orange', alpha=1, marker='^')
    ax.text(x[0] + 0.05, y[0] + 0.05, z[0] + 0.05, 'start point', fontsize=10)
    ax.scatter(x[-1], y[-1], z[-1], s=25, c='blue', alpha=1, marker='v')
    ax.text(x[-1] + 0.05, y[-1] + 0.05, z[-1] + 0.05, 'end point', fontsize=10)
    return


def error_3d_plotting(lines={}, save_path=''):
    fig, axes = plt.subplots(3, 1)
    plt.ion()
    # cmap = plt.cm.get_cmap('Set1', len(lines))
    cmap = ['red', 'green']
    linestyles = ['-', '--', ':']
    params = {'font.family': 'serif',
              'font.serif': 'Times New Roman',
              'font.style': 'normal',
              'font.weight': 'normal',  # or 'blod'
              }
    matplotlib.rcParams.update(params)
    labels = ['X', 'Y', 'Z']
    plt.suptitle('3D error plots', y=0.92, fontsize=12)
    expected_state = [0, 0, 5]

    for i, ax in enumerate(axes):
        matplotlib.rcParams.update({'font.size': 7.4})
        errors = (lines['target'][:, i] - lines['chaser'][:, i]) - expected_state[i]
        x = np.arange(len(errors))
        ax.plot(x, errors, color='red', linestyle=linestyles[0], label=labels[i])
        ax.set_xlabel('timesteps')
        ax.set_ylabel('error')
        ax.legend(loc='upper right')
    plt.ioff()
    # plt.show()
    print('Saving success plots to', save_path)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    root_dir = '../eval/result/DQN_Conv_RGBD_Position'
    scene_dirs = os.listdir(root_dir)
    trajectory_files = []
    for scene_dir in scene_dirs:
        trajectory_dir = os.path.join(root_dir, scene_dir, 'record')
        trajectory_files.append(glob.glob(os.path.join(trajectory_dir, '*_trajectory.txt')))
    trajectory_files = np.concatenate(trajectory_files)
    print('=> {} trajectory file have been found!'.format(len(trajectory_files)))

    for i, trajectory_file in enumerate(trajectory_files):
        trajectory_data = np.loadtxt(trajectory_file, delimiter=',')

        target_trajectory = trajectory_data[:, 1:4]
        chaser_trajectory = trajectory_data[:, 7:10]
        for j in range(len(chaser_trajectory) - 8):
            chaser_trajectory[j] = sum(chaser_trajectory[j:j+8]) / 8
        lines = {'target': target_trajectory,
                 'chaser': chaser_trajectory}
        trajectory_img_file = trajectory_file[:trajectory_file.rfind('_')] + '_trajectory.pdf'
        trajectory_3d_ploting(lines, save_path=trajectory_img_file)
        error_img_file = trajectory_file[:trajectory_file.rfind('_')] + '_error.pdf'
        error_3d_plotting(lines, save_path=error_img_file)

    sys.exit()