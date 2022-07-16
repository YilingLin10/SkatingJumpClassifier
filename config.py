from easydict import EasyDict as edict
import os

# Get the configuration dictionary whose keys can be accessed with dot.
CONFIG = edict()

# ========TRANSFORMER PARAMETERs=========
CONFIG.NUM_HEADS = 3
CONFIG.NUM_ENCODER_LAYERS = 2
CONFIG.DIM_FEEDFORWARD = 256
CONFIG.LAYER_NORM_EPS = 1e-5
CONFIG.NUM_CLASS = 4
CONFIG.USE_CRF = True

# ========Hyperparameters============
CONFIG.LR = 1e-5
CONFIG.EVAL_STEPS = 200
CONFIG.SAVE_STEPS = 2000
CONFIG.NUM_EPOCHS = 2000
CONFIG.BATCH_SIZE = 32

# =========FILE PATHS============
CONFIG.TRAIN_DIR = '/home/lin10/projects/SkatingJumpClassifier/data/train/'
CONFIG.TEST_DIR = '/home/lin10/projects/SkatingJumpClassifier/data/test/'
CONFIG.CSV_FILE = '/home/lin10/projects/SkatingJumpClassifier/data/iceskatingjump.csv'
CONFIG.JSON_FILE = '/home/lin10/projects/SkatingJumpClassifier/data/skating_data.jsonl'
CONFIG.TAG2IDX_FILE = '/home/lin10/projects/SkatingJumpClassifier/data/tag2idx.json'
