import numpy as np
import os
import sys

sys.path.append('/home/group1/dzhou/RAMAVT')

os.environ['COPPELIASIM_ROOT'] = os.environ['HOME'] + '/CoppeliaSim4.2'
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = os.environ['COPPELIASIM_ROOT']

TRAIN_SCENES_DIR = os.environ['HOME'] + '/RAMAVT/Scenes/train'
TEST_SCENES_DIR = os.environ['HOME'] + '/RAMAVT/Scenes/eval'

SCENES = [
    'SNCOAT-Asteroid-v0.ttt', 'SNCOAT-Asteroid-v1.ttt', 'SNCOAT-Asteroid-v2.ttt',
    'SNCOAT-Asteroid-v3.ttt', 'SNCOAT-Asteroid-v4.ttt', 'SNCOAT-Asteroid-v5.ttt',
    'SNCOAT-Capsule-v0.ttt', 'SNCOAT-Capsule-v1.ttt', 'SNCOAT-Capsule-v2.ttt',
    'SNCOAT-Rocket-v0.ttt', 'SNCOAT-Rocket-v1.ttt', 'SNCOAT-Rocket-v2.ttt',
    'SNCOAT-Satellite-v0.ttt', 'SNCOAT-Satellite-v1.ttt', 'SNCOAT-Satellite-v2.ttt',
    'SNCOAT-Station-v0.ttt', 'SNCOAT-Station-v1.ttt', 'SNCOAT-Station-v2.ttt',
]

# action format: force in x, y, z axis

DISCRETE_ACTIONS = np.array([
    [0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0],
    [-1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [0, -1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0],
    [0, 0, -1, 0, 0, 0],
    [1, 1, 0, 0, 0, 0],
    [-1, 1, 0, 0, 0, 0],
    [1, -1, 0, 0, 0, 0],
    [-1, -1, 0, 0, 0, 0],
])

IMAGE_CHANNELS = {
    'Color': 3,
    'Depth': 1,
    'RGBD': 4
}
