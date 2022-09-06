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
flags.DEFINE_string('estimator', 'alphapose', 'alphapose or posetriplet or normalized')
FLAGS = flags.FLAGS

tag_mapping_file = "/home/lin10/projects/SkatingJumpClassifier/data/tag2idx.json"
tag_mapping = json.loads(Path(tag_mapping_file).read_text())
def tag2idx(tag: str):
    return tag_mapping[tag]

def writePickle(pickle_file, samples):
    with open(pickle_file, "wb") as f:
        pickle.dump(samples, f)

def load_data(json_file, split):
    output_file = os.path.join(os.path.dirname(json_file), '{}.pkl'.format(split))
    root_dir = os.path.join(os.path.dirname(json_file), split)
    video_data_list = []
    with open(json_file, 'r') as f:
        for line in f:
            video_data_list.append(json.loads(line))
    samples_list = []
    for video_data in tqdm(video_data_list):
        video_name = video_data['id']
        original_video = video_data['video_name']
        action = original_video.split('_')[0]
        frames = list(Path(os.path.join(f'/home/lin10/projects/SkatingJumpClassifier/20220801/{action}', original_video)).glob("*.jpg"))
        start_frame = video_data['start_frame']
        end_frame = video_data['end_frame']
        start_jump_1 = video_data['start_jump_1']
        end_jump_1 = video_data['end_jump_1']
        start_jump_2 = video_data['start_jump_2']
        end_jump_2 = video_data['end_jump_2']
        tags = []
        frameNumber_list = []
        # create tags and frameNumber_list
        for i in range(start_frame, start_jump_1):
            tags.append(tag2idx('O'))
            frameNumber_list.append(i)
        tags.append(tag2idx('B'))
        frameNumber_list.append(start_jump_1)
        for i in range(start_jump_1 + 1, end_jump_1):
            tags.append(tag2idx('I'))
            frameNumber_list.append(i)
        tags.append(tag2idx('E'))
        frameNumber_list.append(end_jump_1)
        if start_jump_2 == -1:
            ## 1 Jump
            for i in range(end_jump_1 + 1, end_frame + 1):
                tags.append(tag2idx('O'))
                frameNumber_list.append(i)
        else:
            ## 2 jumps
            for i in range(end_jump_1 + 1, start_jump_2):
                tags.append(tag2idx('O'))
                frameNumber_list.append(i)
            tags.append(tag2idx('B'))
            frameNumber_list.append(start_jump_2)
            for i in range(start_jump_2 + 1, end_jump_2):
                tags.append(tag2idx('I'))
                frameNumber_list.append(i)
            tags.append(tag2idx('E'))
            frameNumber_list.append(end_jump_2)
            for i in range(end_jump_2 + 1, end_frame + 1):
                tags.append(tag2idx('O'))
                frameNumber_list.append(i)
        tags = np.array(tags)
        if FLAGS.estimator == 'alphapose':
            alphaposeResults = get_main_skeleton(os.path.join(f'/home/lin10/projects/SkatingJumpClassifier/20220801/{action}', original_video, "alphapose-results.json"))
            assert len(frames) == len(alphaposeResults), "{} error".format(video_name)
            subtractions_list = [subtract_features(alphaposeResults[frameNumber][1]) for frameNumber in frameNumber_list]
            keypoints_list = [np.delete(alphaposeResults[frameNumber][1], 2, axis=1).reshape(-1) for frameNumber in frameNumber_list]
            subtraction_features_list = np.append(keypoints_list, subtractions_list, axis=1)  
            features_list = keypoints_list
            sample = {"subtraction_features": subtraction_features_list, "features": features_list, "video_name": video_name, "output": tags}
        elif FLAGS.estimator == 'posetriplet':
            posetripletResults = get_posetriplet(os.path.join(f'/home/lin10/projects/SkatingJumpClassifier/20220801/{action}', original_video, "{}_pred3D.pkl".format(original_video)))
            assert len(frames) == len(posetripletResults), "{} error, frames: {}, results: {}".format(video_name, len(frames), len(posetripletResults))
            subtractions_list = [subtract_pose3d_features(posetripletResults[frameNumber]) for frameNumber in frameNumber_list]
            keypoints_list = [np.delete(posetripletResults[frameNumber], 2, axis=1).reshape(-1) for frameNumber in frameNumber_list]
            subtraction_features_list = np.append(keypoints_list, subtractions_list, axis=1)  
            features_list = keypoints_list
            sample = {"subtraction_features": subtraction_features_list, "features": features_list, "video_name": video_name, "output": tags}
        else:
            # keypoints_list: [N, 16, 2]
            normalizedResults = get_normalized_keypoints(os.path.join(f'/home/lin10/projects/SkatingJumpClassifier/20220801/{action}', original_video, "alphapose-results.json"))
            tmp_keypoints_list = normalizedResults.copy()
            assert len(frames) == len(normalizedResults), "{} error".format(video_name)
            subtractions_list = [subtract_pose3d_features(tmp_keypoints_list[frameNumber]) for frameNumber in frameNumber_list]
            keypoints_list = [normalizedResults[frameNumber].reshape(-1) for frameNumber in frameNumber_list]
            subtraction_features_list = np.append(keypoints_list, subtractions_list, axis=1)  
            features_list = keypoints_list
            sample = {"subtraction_features": subtraction_features_list, "features": features_list, "video_name": video_name, "output": tags}
        samples_list.append(sample)
    writePickle(output_file, samples_list)

def main(_argv):
    for split in ["test", "train"]:
        print("Preprocessing {} {}ing data".format(FLAGS.action, split))
        load_data("/home/lin10/projects/SkatingJumpClassifier/data/{}/{}/{}_aug.jsonl".format(FLAGS.action, FLAGS.estimator, split), split)

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass