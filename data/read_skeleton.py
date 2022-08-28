import numpy as np 
import json
import numpy as np
import os
import cv2
from scipy.spatial import distance
import warnings
import pickle
import glob
from pathlib import Path

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 
SKELETON_CONF_THRESHOLD=0.0
def get_main_skeleton(path_to_json):
  root_dir = os.path.dirname(path_to_json)
  frames = list(Path(f'{root_dir}').glob("*.jpg"))
  last_frame = last_frame = sorted([int(os.path.split(frame)[1].replace('.jpg','')) for frame in frames])[-1]
  with open(path_to_json) as f:
    json_skeleton = json.load(f)
  
  # Get first image id
  prev_frame = int(json_skeleton[0]['image_id'].rstrip('.jpg'))
  score_list = []
  main_skeleton_list = []
  for skeleton in json_skeleton:
    # Get current image id
    cur_frame = int(skeleton['image_id'].rstrip('.jpg'))
    # Get all skeleton in current frame
    if cur_frame == prev_frame:
      score_list.append([skeleton['score'], np.reshape(skeleton['keypoints'], (17,3))])
    # Get the main skeleton in previous frame by score or distance
    else:
      if main_skeleton_list == []:
        main_skeleton_list.append([prev_frame, np.array(max(score_list))])
      else:
        dist_list = []
        # Pick a skeleton that is the closest to the previous one
        for score in score_list:
          ske_distance = distance.euclidean(np.delete(score[1], (2), axis=1).ravel(), np.delete(main_skeleton_list[-1][1][1], (2), axis=1).ravel())
          dist_list.append([ske_distance, score])
        main_skeleton_list.append([prev_frame, np.array(min(dist_list)[1])])
      # Clear score list and append first skeleton for current frame
      score_list.clear()
      score_list.append([skeleton['score'], np.reshape(skeleton['keypoints'], (17,3))])
    prev_frame = cur_frame
  # append max score for the last frame
  main_skeleton_list.append([cur_frame, np.array(max(score_list))])
  main_skeleton_list = np.reshape(main_skeleton_list, (-1,2))

  keypoints_info = []
  for main_skeleton in main_skeleton_list:
    delete_pt = []
    keypoints = main_skeleton[1][1]
    for row in range(len(keypoints)):
      if keypoints[row][2] < SKELETON_CONF_THRESHOLD:
        delete_pt.append(row)
    keypoints = np.delete(keypoints, (delete_pt), axis=0)
    keypoints_info.append([main_skeleton[0], keypoints])
  keypoints_info = np.array(keypoints_info)

  # Insert missing frame info
  for i in range(last_frame+1):
    if i not in keypoints_info[:,0]:
      insert_row = np.concatenate([[i],keypoints_info[i-1,1:]], axis=0)
      keypoints_info = np.insert(keypoints_info, i, insert_row, axis=0)

  return keypoints_info

# subtract right joint x from left joint x...
def subtract_features(keypoints):
  # input: [17, 2]
  # output: [8]
  subtracted_keypoints = []
  for i in range(1, 17, 2):
    subtraction = keypoints[i][0] - keypoints[i+1][0]
    subtracted_keypoints.append(subtraction)
  
  return np.array(subtracted_keypoints)

def get_posetriplet(path_to_pickle):
  with open(path_to_pickle, 'rb') as fp:
      # [num_frames, 16, 3]
      pose3d = pickle.load(fp)

  results = pose3d['result']
  return results
  

if __name__ == '__main__':
  path_to_json = "/home/lin10/projects/SkatingJumpClassifier/data/loop/train/Loop_22/alphapose-results.json"
  alphaposeResults = get_main_skeleton(path_to_json)
  print(len(alphaposeResults))