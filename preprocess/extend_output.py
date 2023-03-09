import numpy as np
import os 
import glob
import pandas as pd
import json
import cv2
from pathlib import Path
import csv
from read_skeleton import *
from absl import flags
from absl import app
from tqdm import tqdm
import pickle

flags.DEFINE_string('action', None, 'axel, flip, loop, lutz, old, salchow, toe')
flags.DEFINE_string('feature_type', None, 'raw_skeletons or embeddings')
FLAGS = flags.FLAGS

def writePickle(pickle_file, samples):
    with open(pickle_file, "wb") as f:
        pickle.dump(samples, f)

def get_original_sample_list(action, feature_type, split):
    original_sample_file = os.path.join("/home/lin10/projects/SkatingJumpClassifier/data", action, feature_type, f"{split}.pkl")
    with open(original_sample_file, 'rb') as f:
        original_sample_list = pickle.load(f)
    return original_sample_list

def extend_output(original_sample_list):
    """
        original_sample = {
            "video_name":{}
            "features": {},
            "output": {} 
        }
    """
    start_action_duration = 5
    end_action_duration = 5
    for original_sample in tqdm(original_sample_list):
        start_jump_frames = [i for i, label in enumerate(original_sample["output"]) if label == 1]
        end_jump_frames = [i for i, label in enumerate(original_sample["output"]) if label == 3]
        
        for start_frame in start_jump_frames:
            original_sample["output"][start_frame-start_action_duration+1:start_frame+1] = np.ones((start_action_duration))
        for end_frame in end_jump_frames:
            original_sample["output"][end_frame: end_frame + end_action_duration] = 3 * np.ones((end_action_duration))
    return original_sample_list

def generate_extended_data(action, feature_type, split):
    extended_output_file = os.path.join("/home/lin10/projects/SkatingJumpClassifier/data", action, f"extended_{feature_type}", f"{split}.pkl")
    
    original_sample_list = get_original_sample_list(action, feature_type, split)
    extended_sample_list = extend_output(original_sample_list)
    
    if not os.path.exists(os.path.dirname(extended_output_file)):
        os.makedirs(os.path.dirname(extended_output_file))
    writePickle(extended_output_file, extended_sample_list)
    
def main(_argv):
    for split in ["test", "train"]:
        print("Extending output for ACTION: {}, FEATURE_TYPE: {}, SPLIT: {}".format(FLAGS.action, FLAGS.feature_type, split))
        generate_extended_data(FLAGS.action, FLAGS.feature_type, split)

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass