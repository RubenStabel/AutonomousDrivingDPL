import numpy as np

from deepproblog.examples.Autonomous_driving.experimental.networks.network import AD_V0_0_net, AD_V1_1_net
from deepproblog.examples.Autonomous_driving.version_0.networks.network_NeSy import AD_V0_NeSy_1_net
from deepproblog.examples.Autonomous_driving.version_0.networks.network_baseline import AD_V0_baseline_net
from deepproblog.examples.Autonomous_driving.version_2.networks.network_NeSy import AD_V2_NeSy_2_net_x_rel, \
    AD_V2_NeSy_2_net_y_rel, AD_V2_NeSy_1_net_ped, AD_V2_NeSy_1_net_speed_zone
from deepproblog.examples.Autonomous_driving.version_2.networks.network_baseline import AD_V2_baseline_net_1, \
    AD_V2_baseline_net_0, AD_V2_baseline_net_2
from deepproblog.examples.Autonomous_driving.version_3.networks.network_baseline import AD_V3_baseline_net_0
from deepproblog.examples.Autonomous_driving.version_5.networks.network_NeSy import AD_V5_NeSy_0_net_danger_pedestrian, \
    AD_V5_NeSy_0_net_speed_zone, AD_V5_NeSy_0_net_traffic_light, AD_V5_NeSy_0_net_danger_intersection, \
    AD_V5_NeSy_0_net_traffic_sign, AD_V5_NeSy_0_net_danger_distance
from deepproblog.examples.Autonomous_driving.version_5.networks.network_baseline import AD_V5_baseline_net_0

"""
MODE:
0 --> Drive sim with keys
1 --> DEV [Currently - Advanced rule based driving]
2 --> NeSy self driving
3 --> PyTorch self driving
4 --> Version 0 rule based self driving
5 --> Version 1 rule based self driving
6 --> Version 2 rule based self driving
7 --> Version 3 rule based self driving
8 --> Version 4 rule based self driving
9 --> Version 5 rule based self driving
10 --> Version 6 rule based self driving
11 --> Version 7 rule based self driving
"""
MODE = 10
NN_INPUTS = 5

"""
DISPLAY SETTINGS
"""
DYNAMIC_SIMULATION = True

"""
GAMEPLAY SETTINGS
"""
MAX_VEL = 8
OCCLUDED_OBJ_VISIBLE = True

"""
DATA COLLECTION
"""
COLLECT_DATA = False
DATA_ANALYSIS = False
SIMULATION_METRICS = False

PREFIX = '1'
DATA_FOLDER = "general/version_5_env_5_tl_pos"
ENV = 'env_5'

IMAGE_DIM = 360
NUMBER_ITERATIONS = 50

"""
ENVIRONMENT SETTINGS
"""
NUMBER_PEDESTRIANS = 1
NUMBER_STATIC_CARS = 15
NUMBER_SPEED_ZONES = 1
NUMBER_TRAFFIC_LIGHTS = 1
TRAFFIC_LIGHT_INTERSECTION = True
TRAFFIC_LIGHT_ORANGE = False
NUMBER_INTERSECTIONS = 1
NUMBER_DYNAMIC_CARS = 8
DYNAMIC_CARS_TRAFFIC_LIGHTS = True
DYNAMIC_CARS_TRAFFIC_LIGHTS_DRAW = False
NUMBER_TRAFFIC_SIGNS = 2
TRAFFIC_SIGNS_LEFT = False
NUMBER_SCENERY = 0
EXTENDED_ROAD = False

"""
TESTING
"""
SCENARIO_MODE = False

SCENARIO = {
    'low pass': [(30, 70), (29, 69), (28, 68), (27, 67), (26, 66), (25, 65), (24, 64), (23, 63), (22, 62), (21, 61), (20, 61), (19, 60), (18, 59), (17, 58), (16, 57), (15, 57), (14, 56), (13, 56), (12, 55), (11, 54), (10, 53), (9, 52), (8, 51), (7, 50)],
    'high pass': [(30, 35), (29, 34), (28, 33), (27, 32), (26, 31), (25, 30), (24, 29), (23, 28), (22, 27), (21, 26), (20, 25), (19, 24), (18, 23), (17, 22), (16, 21), (15, 20), (15, 19), (14, 18), (13, 17), (12, 16), (11, 15), (11, 14), (11, 13), (11, 12), (10, 11), (9, 10), (8, 9), (8, 8), (8, 7), (8, 6), (7, 5)],
    'mid pass': [(30, 50), (29, 50), (28, 50), (27, 50), (26, 50), (25, 50), (24, 50), (23, 50), (22, 50), (21, 50), (20, 50), (19, 50), (18, 50), (17, 50), (16, 50), (15, 50), (14, 50), (13, 50), (12, 50), (11, 50), (10, 50), (9, 50), (8, 50), (7, 50)],
    'low high': [(28, 70), (28, 69), (28, 68), (28, 67), (28, 66), (28, 65), (28, 64), (28, 63), (28, 62), (28, 61), (28, 60), (28, 59), (28, 58), (28, 57), (28, 56), (28, 55), (27, 54), (26, 53), (26, 52), (26, 51), (26, 50), (26, 49), (26, 48), (26, 47), (26, 46), (26, 45), (26, 44), (26, 43), (26, 42), (26, 41), (26, 40), (25, 39), (24, 38), (24, 37), (23, 36), (22, 35), (22, 34), (22, 33), (21, 32), (20, 31), (19, 30), (18, 29), (17, 28), (16, 27), (15, 26), (14, 25), (13, 24), (12, 23), (11, 22), (10, 21), (9, 20), (8, 19), (7, 18), (6, 17), (6, 16), (6, 15), (5, 14), (4, 13), (3, 12), (3, 11), (3, 10), (3, 9), (3, 8), (2, 7), (1, 6), (1, 5)],
    'high low': [(34, 35), (33, 36), (32, 37), (31, 38), (30, 39), (29, 39), (28, 40), (27, 41), (26, 41), (25, 41), (24, 41), (23, 42), (22, 43), (21, 43), (20, 44), (19, 44), (18, 44), (17, 44), (16, 44), (15, 44), (14, 44), (13, 45), (12, 45), (11, 46), (10, 47), (9, 48), (8, 49), (7, 49), (6, 49), (5, 50)]
}

# # Baseline NeSy
# NETWORK = [AD_V1_net()]
# MODEL_NAME = "NN"
# MODEL_PATH = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/deepproblog/examples/Autonomous_driving/models/autonomous_driving_baseline.pl'
# NN_PATH = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/deepproblog/examples/Autonomous_driving/snapshot/baseline/autonomous_driving_baseline_NeSy_9.pth'
# NN_NAME = ['ad_baseline_net']

# # NeSy V1.1
# NETWORK = [AD_V0_NeSy_1_net()]
# MODEL_NAME = "version_0_NeSy_1"
# MODEL_PATH = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/deepproblog/examples/Autonomous_driving/version_0/models/collision_NeSy.pl'
# NN_PATH = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/deepproblog/examples/Autonomous_driving/version_0/snapshot/neuro_symbolic/test/autonomous_driving_NeSy_1_complete_0.pth'
# NN_NAME = ['perc_net_version_0_NeSy_1']

# # V2 - baseline_0
# MODEL_NAME = "NeSy"
# NETWORK = [AD_V2_baseline_net_0()]
# MODEL_PATH = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/deepproblog/examples/Autonomous_driving/version_2/models/autonomous_driving_baseline_0.pl'
# NN_PATH = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/deepproblog/examples/Autonomous_driving/version_2/snapshot/baseline/train/autonomous_driving_baseline_0_complete_env_1_0.pth'
# NN_NAME = ['perc_net_version_2_baseline_0']

# # V2 - baseline_1
# MODEL_NAME = "NeSy"
# NETWORK = [AD_V2_baseline_net_1()]
# MODEL_PATH = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/deepproblog/examples/Autonomous_driving/version_2/models/autonomous_driving_baseline_1.pl'
# NN_PATH = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/deepproblog/examples/Autonomous_driving/version_2/snapshot/baseline/train/autonomous_driving_baseline_1_complete_env_1_0.pth'
# NN_NAME = ['perc_net_version_2_baseline_1']

# # V2 - baseline_2
# MODEL_NAME = "NeSy"
# NETWORK = [AD_V2_baseline_net_2()]
# MODEL_PATH = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/deepproblog/examples/Autonomous_driving/version_2/models/autonomous_driving_baseline_2.pl'
# NN_PATH = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/deepproblog/examples/Autonomous_driving/version_2/snapshot/baseline/test/autonomous_driving_baseline_2_complete_env_2_0.pth'
# NN_NAME = ['perc_net_version_2_baseline_2']

# # V2 - NeSy_1
# MODEL_NAME = "NeSy"
# NETWORK = [AD_V2_NeSy_1_net_ped(), AD_V2_NeSy_1_net_speed_zone()]
# MODEL_PATH = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/deepproblog/examples/Autonomous_driving/version_2/models/autonomous_driving_NeSy_1.pl'
# NN_PATH = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/deepproblog/examples/Autonomous_driving/version_2/snapshot/neuro_symbolic/test/autonomous_driving_NeSy_1_complete_env_2_0.pth'
# NN_NAME = ['perc_net_version_2_NeSy_ped', 'perc_net_version_2_NeSy_speed_zone']

# # V2 - NeSy_3
# MODEL_NAME = "NeSy"
# NETWORK = [AD_V2_NeSy_2_net_x_rel(), AD_V2_NeSy_2_net_y_rel()]
# MODEL_PATH = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/deepproblog/examples/Autonomous_driving/version_2/models/autonomous_driving_NeSy_3.pl'
# NN_PATH = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/deepproblog/examples/Autonomous_driving/version_2/snapshot/neuro_symbolic/train/autonomous_driving_NeSy_3_complete_env_1_0.pth'
# NN_NAME = ['perc_net_version_2_NeSy_x_rel', 'perc_net_version_2_NeSy_y_rel']

# # V3 - baseline_0
# MODEL_NAME = "NeSy"
# NETWORK = [AD_V3_baseline_net_0()]
# MODEL_PATH = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/deepproblog/examples/Autonomous_driving/version_3/models/autonomous_driving_baseline_0.pl'
# NN_PATH = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/deepproblog/examples/Autonomous_driving/version_3/snapshot/baseline/test/autonomous_driving_baseline_0_medium_env_3_0.pth'
# NN_NAME = ['perc_net_version_3_baseline_0']

# V5 - NeSy_1
MODEL_NAME = "NeSy"
# ENV = 5
NETWORK = [AD_V5_NeSy_0_net_danger_pedestrian(), AD_V5_NeSy_0_net_speed_zone(), AD_V5_NeSy_0_net_traffic_light(), AD_V5_NeSy_0_net_danger_intersection(), AD_V5_NeSy_0_net_traffic_sign()]
OUTPUT_DATA = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/data/output_data/output_10_env_5_tl_pos.txt'
MODEL_PATH = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/deepproblog/examples/Autonomous_driving/version_5/models/autonomous_driving_NeSy_1.pl'
NN_PATH = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/deepproblog/examples/Autonomous_driving/version_5/snapshot/neuro_symbolic/test/autonomous_driving_NeSy_1_medium_env_5_pretrain_4.pth'
NN_NAME = ['perc_net_version_5_NeSy_danger_pedestrian', 'perc_net_version_5_NeSy_speed_zone', 'perc_net_version_5_NeSy_traffic_light', 'perc_net_version_5_NeSy_intersection', 'perc_net_version_5_NeSy_traffic_sign']

# # V5 - NeSy_0
# MODEL_NAME = "NeSy"
# NETWORK = [AD_V5_NeSy_0_net_danger_pedestrian(), AD_V5_NeSy_0_net_speed_zone(), AD_V5_NeSy_0_net_traffic_light(), AD_V5_NeSy_0_net_danger_distance(), AD_V5_NeSy_0_net_danger_intersection(), AD_V5_NeSy_0_net_traffic_sign()]
# OUTPUT_DATA = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/data/output_data/output_10_env_5.txt'
# MODEL_PATH = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/deepproblog/examples/Autonomous_driving/version_5/models/autonomous_driving_NeSy_0.pl'
# NN_PATH = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/deepproblog/examples/Autonomous_driving/version_5/snapshot/neuro_symbolic/test/autonomous_driving_NeSy_0_complete_env_5_0.pth'
# NN_NAME = ['perc_net_version_5_NeSy_danger_pedestrian', 'perc_net_version_5_NeSy_speed_zone', 'perc_net_version_5_NeSy_traffic_light', 'perc_net_version_5_NeSy_danger_distance', 'perc_net_version_5_NeSy_intersection', 'perc_net_version_5_NeSy_traffic_sign']

#
# # V5 - baseline
# MODEL_NAME = "NeSy"
# # ENV = 7
# NETWORK = [AD_V5_baseline_net_0()]
# OUTPUT_DATA = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/data/output_data/output_10_env_7.txt'
# MODEL_PATH = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/deepproblog/examples/Autonomous_driving/version_5/models/autonomous_driving_baseline_0.pl'
# NN_PATH = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/deepproblog/examples/Autonomous_driving/version_5/snapshot/baseline/test/autonomous_driving_baseline_0_complete_env_7_0.pth'
# NN_NAME = ['perc_net_version_5_baseline_0']

# # # Baseline PyTorch
# NETWORK = [AD_V1_net()]
# MODEL_NAME = "NN"
# NN_PATH = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/deepproblog/examples/Autonomous_driving/snapshot/baseline/test/autonomous_driving_baseline_V0_0.pth'
# MODEL_PATH = ''
# NN_NAME = ['']
