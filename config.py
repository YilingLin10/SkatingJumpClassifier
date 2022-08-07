from easydict import EasyDict as edict
import os

# Get the configuration dictionary whose keys can be accessed with dot.
CONFIG = edict()

# ========TRANSFORMER PARAMETERs=========
CONFIG.USE_EMBS = False

if CONFIG.USE_EMBS:
    #### USE_EMBS:
    CONFIG.D_MODEL = 16
    CONFIG.NUM_HEADS = 2
    CONFIG.DIM_FEEDFORWARD = 128
else:
    #### USE_KEYPOINTS:
    CONFIG.D_MODEL = 51
    CONFIG.NUM_HEADS = 3
    CONFIG.DIM_FEEDFORWARD = 256

CONFIG.NUM_ENCODER_LAYERS = 2
CONFIG.LAYER_NORM_EPS = 1e-5
CONFIG.NUM_CLASS = 4
CONFIG.USE_CRF = True
CONFIG.ADD_NOISE = False
# ========Hyperparameters============
CONFIG.LR = 1e-5
CONFIG.EVAL_STEPS = 500
CONFIG.SAVE_STEPS = 2000
CONFIG.NUM_EPOCHS = 2000
CONFIG.BATCH_SIZE = 32

# =========FILE PATHS============
CONFIG.TRAIN_DIR = '/home/lin10/projects/SkatingJumpClassifier/data/train/'
CONFIG.TEST_DIR = '/home/lin10/projects/SkatingJumpClassifier/data/test/'
CONFIG.CSV_FILE = '/home/lin10/projects/SkatingJumpClassifier/data/iceskatingjump.csv'
CONFIG.JSON_FILE = '/home/lin10/projects/SkatingJumpClassifier/data/train_data_aug2.jsonl'
CONFIG.TAG2IDX_FILE = '/home/lin10/projects/SkatingJumpClassifier/data/tag2idx.json'

# ========= 220801/Loop FILE PATHS ======
# CONFIG.TRAIN_DIR = '/home/lin10/projects/SkatingJumpClassifier/20220801/Loop/'
# CONFIG.TEST_DIR = '/home/lin10/projects/SkatingJumpClassifier/data/test/'
# CONFIG.CSV_FILE = '/home/lin10/projects/SkatingJumpClassifier/data/iceskatingjump.csv'
# CONFIG.JSON_FILE = '/home/lin10/projects/SkatingJumpClassifier/data/loop_data_aug2.jsonl'
# CONFIG.TAG2IDX_FILE = '/home/lin10/projects/SkatingJumpClassifier/data/tag2idx.json'