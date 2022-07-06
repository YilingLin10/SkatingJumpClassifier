from easydict import EasyDict as edict
import os

# Get the configuration dictionary whose keys can be accessed with dot.
CONFIG = edict()

# ========TRANSFORMER PARAMETERs=========
CONFIG.NUM_HEADS = 3
CONFIG.NUM_ENCODER_LAYERS = 2
CONFIG.DIM_FEEDFORWARD = 256
CONFIG.LAYER_NORM_EPS = 1e-5


# ========Hyperparameters============
CONFIG.LR = 1e-3
CONFIG.EVAL_STEPS = 1000
CONFIG.SAVE_STEPS = 2000
CONFIG.NUM_EPOCHS = 1000
CONFIG.BATCH_SIZE = 4

