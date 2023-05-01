from deepproblog.examples.AD_V0.network import AD_V1_net

"""
0 --> Drive sim with arrows
1 --> Rule based driving (for data collection)
2 --> NN self driving
"""
MODE = 1
RULE_BASED = True
MAX_VEL = 8
OCCLUDED_OBJ_VISIBLE = True
IMAGE_DIM = 360
DATA_FOLDER = "train"
PREFIX = '0'
COLLECT_DATA = True

# Baseline NeSy
# MODEL_PATH = '/Users/rubenstabel/Documents/universiteit/AD_V0.2 kopie/Traffic_simulation_V0/deepproblog/src/deepproblog/examples/AD_V0/models/autonomous_driving_baseline.pl'
# NN_PATH = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/deepproblog/examples/AD_V0/snapshot/baseline/autonomous_driving_baseline_NeSy_1.pth'
# NN_NAME = 'ad_baseline_net'

# NeSy V1
MODEL_PATH = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/deepproblog/examples/AD_V0/models/autonomous_driving_V1.0.pl'
NN_PATH = '/Users/rubenstabel/Documents/Thesis/Implementation/AutonomousDrivingDPL/src/deepproblog/examples/AD_V0/snapshot/neuro_symbolic/autonomous_driving_NeSy_1.pth'
NN_NAME = 'perc_net_AD_V1'
NETWORK = AD_V1_net()