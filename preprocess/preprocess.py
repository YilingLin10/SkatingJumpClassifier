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
flags.DEFINE_string('reverse', 'false', 'whether to include features with reversed sequence order')
FLAGS = flags.FLAGS

tag_mapping_file = "/home/lin10/projects/SkatingJumpClassifier/data/tag2idx.json"
tag_mapping = json.loads(Path(tag_mapping_file).read_text())
def tag2idx(tag: str):
    return tag_mapping[tag]

def writePickle(pickle_file, samples):
    with open(pickle_file, "wb") as f:
        pickle.dump(samples, f)

def reverse_feature(keypoints_list):
    # keypoints_list: [len, feature_size]
    reversed_list = []
    for i in range(len(keypoints_list), 0, -1):
        reversed_list.append(keypoints_list[i-1])
    return reversed_list

def reverse_output(keypoints_list):
    # keypoints_list: [len, feature_size]
    reversed_list = []
    for i in range(len(keypoints_list), 0, -1):
        reversed_list.append(keypoints_list[i-1])
    for i, tag in enumerate(reversed_list):
        if tag == 3:
            reversed_list[i] = 1
        elif tag == 1:
            reversed_list[i] = 3
    return reversed_list

def get_raw_skeletons(action, original_video):
    """
        T: # of frames of original_video
        return np.array (T, 34)
    """
    alphapose_results = get_main_skeleton(os.path.join(f'/home/lin10/projects/SkatingJumpClassifier/20220801/{action}', original_video, "alphapose-results.json"))
    keypoints_list = [np.delete(alphapose_result[1], 2, axis=1).reshape(-1) for alphapose_result in alphapose_results]
    keypoints_array = np.stack(keypoints_list)
    return keypoints_array

def get_subtraction_features(action, original_video):
    """
        T: # of frames of original_video
        return np.array (T, 34+8)
    """
    alphapose_results = get_main_skeleton(os.path.join(f'/home/lin10/projects/SkatingJumpClassifier/20220801/{action}', original_video, "alphapose-results.json"))
    subtractions_list = [subtract_features(alphapose_result[1]) for alphapose_result in alphapose_results]
    keypoints_list = [np.delete(alphapose_result[1], 2, axis=1).reshape(-1) for alphapose_result in alphapose_results]
    subtraction_features_list = np.append(keypoints_list, subtractions_list, axis=1)  
    subtraction_features_array = np.stack(subtraction_features_list)
    return subtraction_features_array

def get_posetransformer_embeddings(action, original_video):
    """
        T: # of frames of original_video
        return np.array (T, 544)
    """
    embedding_file = os.path.join(f'/home/lin10/projects/SkatingJumpClassifier/20220801/{action}', original_video, "skeleton_embedding.pkl")
    with open(embedding_file, 'rb') as f:
        embeddings = pickle.load(f)
    return embeddings

def get_poem_embeddings(action, original_video):
    embedding_file = os.path.join(f'/home/lin10/projects/SkatingJumpClassifier/20220801/{action}', original_video, "unnormalized_embeddings.csv")
    with open(embedding_file, 'rt') as f:
        csv_reader = csv.reader(f, delimiter=',')
        all_embeddings = []
        for embeddings in csv_reader:
            all_embeddings.append(np.array(embeddings))
    return np.stack(all_embeddings)
        
def normalize_keypoints(X, w, h):
    """
    divide by [w,h] and map to [-1,1]
    """
    X = np.divide(X, np.array([w, h]))
    return X*2 - [1,1]

def get_video_hw(action, video):
    if (os.path.exists(os.path.join("/home/lin10/projects/SkatingJumpClassifier/20220801/", action, video, "0.jpg"))):
        img = cv2.imread(os.path.join("/home/lin10/projects/SkatingJumpClassifier/20220801/", action, video, "0.jpg"))
    else:
        img = cv2.imread(os.path.join("/home/lin10/projects/SkatingJumpClassifier/20220801/", action, video, "0000.jpg"))
    height, width, _ = img.shape
    return height, width

def get_normalized_skeletons(action, original_video):
    alphaposeResults = get_main_skeleton(os.path.join(f'/home/lin10/projects/SkatingJumpClassifier/20220801/{action}', original_video, "alphapose-results.json"))
    keypoints_list = [np.delete(alphaposeResult[1], 2, axis=1) for alphaposeResult in alphaposeResults]
    keypoints_array = np.stack(keypoints_list)
    ############## normalize joint coordinates to [-1, 1] #############
    # get height, width of video
    height, width = get_video_hw(action, original_video)
    normalized_keypoints = normalize_keypoints(keypoints_array[..., :2], width, height).astype(np.float32)
    normalized_keypoints = normalized_keypoints.reshape((-1 ,34))
    return normalized_keypoints
    
# def get_posetriplet_skeleton(action, original_video):
#     """
#         T: # of frames of original_video
#         return np.array (T, 32)
#     """
#     posetripletResults = get_posetriplet(os.path.join(f'/home/lin10/projects/SkatingJumpClassifier/20220801/{action}', original_video, "{}_pred3D.pkl".format(original_video)))
#     keypoints_list = [np.delete(posetripletResult, 2, axis=1).reshape(-1) for posetripletResult in posetripletResults]
#     keypoints_array = np.stack(keypoints_list)
#     return keypoints_array

def get_tags(video_data):
    """
        T: # of frames of the augmented sample
        return: (T)
    """
    start_frame = video_data['start_frame']
    end_frame = video_data['end_frame']
    start_jump_1 = video_data['start_jump_1']
    end_jump_1 = video_data['end_jump_1']
    start_jump_2 = video_data['start_jump_2']
    end_jump_2 = video_data['end_jump_2']
    tags = []
    # create tags
    for i in range(start_frame, start_jump_1):
        tags.append(tag2idx('O'))
    tags.append(tag2idx('B'))
    for i in range(start_jump_1 + 1, end_jump_1):
        tags.append(tag2idx('I'))
    tags.append(tag2idx('E'))
    if start_jump_2 == -1:
        ## 1 Jump
        for i in range(end_jump_1 + 1, end_frame + 1):
            tags.append(tag2idx('O'))
    else:
        ## 2 jumps
        for i in range(end_jump_1 + 1, start_jump_2):
            tags.append(tag2idx('O'))
        tags.append(tag2idx('B'))
        for i in range(start_jump_2 + 1, end_jump_2):
            tags.append(tag2idx('I'))
        tags.append(tag2idx('E'))
        for i in range(end_jump_2 + 1, end_frame + 1):
            tags.append(tag2idx('O'))
    tags = np.array(tags)
    return tags

def load_data(json_file, split):
    raw_skeletons_output_file = os.path.join(os.path.dirname(json_file), "raw_skeletons", '{}.pkl'.format(split))
    embeddings_output_file = os.path.join(os.path.dirname(json_file), "embeddings", '{}.pkl'.format(split))
    normalized_skeletons_output_file = os.path.join(os.path.dirname(json_file), "normalized_skeletons", '{}.pkl'.format(split))
    poem_embeddings_output_file = os.path.join(os.path.dirname(json_file), "poem_embeddings", '{}.pkl'.format(split))
    video_data_list = []
    with open(json_file, 'r') as f:
        for line in f:
            video_data_list.append(json.loads(line))
    raw_skeleton_sample_list = []
    embedding_sample_list = []
    normalized_skeleton_sample_list = []
    poem_embedding_sample_list = []
    for video_data in tqdm(video_data_list):
        video_name = video_data['id']
        original_video = video_data['video_name']
        split_video_name = original_video.split('_')
        if (len(split_video_name) == 3):
            action = f"{split_video_name[0]}_{split_video_name[1]}"
        else:
            action = split_video_name[0]
        start_frame = video_data['start_frame']
        end_frame = video_data['end_frame']
        tags = get_tags(video_data)
        
        ## write raw_skeletons.pkl
        raw_skeletons = get_raw_skeletons(action, original_video)[start_frame: end_frame+1]
        raw_skeleton_sample = {
            "video_name": video_name,
            "features": raw_skeletons,
            "output": tags 
        }
        raw_skeleton_sample_list.append(raw_skeleton_sample)
        
        ### write embeddings.pkl
        embeddings = get_posetransformer_embeddings(action, original_video)[start_frame: end_frame+1]
        embedding_sample = {
            "video_name": video_name,
            "features": embeddings,
            "output": tags 
        }
        embedding_sample_list.append(embedding_sample)
        
        ## write normalized_skeletons.pkl
        normalized_skeletons = get_normalized_skeletons(action, original_video)[start_frame: end_frame+1]
        
        normalized_skeleton_sample = {
            "video_name": video_name,
            "features": normalized_skeletons,
            "output": tags 
        }
        normalized_skeleton_sample_list.append(normalized_skeleton_sample)
        
        ### write poem_embeddings.pkl
        poem_embeddings = get_poem_embeddings(action, original_video)[start_frame: end_frame+1]
        
        poem_embedding_sample = {
            "video_name": video_name,
            "features": poem_embeddings,
            "output": tags 
        }
        poem_embedding_sample_list.append(poem_embedding_sample)
        if (poem_embeddings.shape[0] != raw_skeletons.shape[0]):
            print(f"{video_name} error!")
        
    # ### Write raw_skeletons.pkl
    if not os.path.exists(os.path.dirname(raw_skeletons_output_file)):
        os.makedirs(os.path.dirname(raw_skeletons_output_file))
    writePickle(raw_skeletons_output_file, raw_skeleton_sample_list)
    
    # ### Write embeddings.pkl
    if not os.path.exists(os.path.dirname(embeddings_output_file)):
        os.makedirs(os.path.dirname(embeddings_output_file))
    writePickle(embeddings_output_file, embedding_sample_list)
    
    ### Write normalized_skeletons.pkl
    if not os.path.exists(os.path.dirname(normalized_skeletons_output_file)):
        os.makedirs(os.path.dirname(normalized_skeletons_output_file))
    writePickle(normalized_skeletons_output_file, normalized_skeleton_sample_list)
    
    ### Write poem_embeddings.pkl
    if not os.path.exists(os.path.dirname(poem_embeddings_output_file)):
        os.makedirs(os.path.dirname(poem_embeddings_output_file))
    writePickle(poem_embeddings_output_file, poem_embedding_sample_list)
    

def main(_argv):
    for split in ["test", "train"]:
        print("Preprocessing {} {}ing data".format(FLAGS.action, split))
        load_data("/home/lin10/projects/SkatingJumpClassifier/data/{}/{}_aug.jsonl".format(FLAGS.action, split), split)

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass