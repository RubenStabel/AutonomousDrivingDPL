from deepproblog.examples.AD_V0.network import AD_V1_net

"""
0 --> Drive sim with arrows
1 --> Rule based driving (for data collection)
2 --> NN self driving
3 --> Simple rule based driving (for data collection)
4 --> Baseline NN self driving
"""
MODE = 4
RULE_BASED = True
MAX_VEL = 8
OCCLUDED_OBJ_VISIBLE = True
IMAGE_DIM = 360
DATA_FOLDER = "train_simple_yellow_1"
PREFIX = '0'
COLLECT_DATA = False
NUMBER_STATIC_CARS = 0
DATA_ANALYSIS = False

# # Baseline
# NETWORK = AD_V1_net()
# MODEL_NAME = "NN"
# MODEL_PATH = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/deepproblog/examples/AD_V0/models/autonomous_driving_baseline.pl'
# NN_PATH = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/deepproblog/examples/AD_V0/snapshot/baseline/autonomous_driving_baseline_NeSy_9.pth'
# NN_NAME = 'ad_baseline_net'

# NeSy V1
# NETWORK = AD_V1_net()
# MODEL_NAME = "NeSy"
# MODEL_PATH = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/deepproblog/examples/AD_V0/models/autonomous_driving_V0.0.pl'
# NN_PATH = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/deepproblog/examples/AD_V0/snapshot/neuro_symbolic/autonomous_driving_NeSy_15.pth'
# NN_NAME = 'perc_net_AD_V1'

# # Baseline PyTorch
NETWORK = AD_V1_net()
MODEL_NAME = "NN"
NN_PATH = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/deepproblog/examples/AD_V0/snapshot/baseline/test/autonomous_driving_baseline_V0_0.pth'
MODEL_PATH = ''
NN_NAME = ''
