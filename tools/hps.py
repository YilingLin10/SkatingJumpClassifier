import numpy as np 
import json
import numpy as np
import os
import cv2
from scipy.spatial import distance
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

SKELETON_CONF_THRESHOLD=0.0
# Return keypoint data for each frame [image_id, array(17, 3)]
def get_main_skeleton(path_to_json):
  with open(os.path.join(path_to_json, 'alphapose-results.json')) as f:
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
  for i in range(cur_frame+1):
    if i not in keypoints_info[:,0]:
      insert_row = np.concatenate([[i],keypoints_info[i-1,1:]], axis=0)
      keypoints_info = np.insert(keypoints_info, i, insert_row, axis=0)

  return keypoints_info

def hps(X):
  X_mean = np.mean(X, axis=0)
  X_centered = X -  X_mean
  X_cov = np.cov(X_centered.T)

  # * 3.1. Get Eigen Values and Vectors
  eigen_values, eigen_vectors = np.linalg.eig(X_cov)
  # * 3.2. Get the top M eigen values indices
  top_M_idx = np.argsort(eigen_values)[::-1][-1]

  # return np.dot(X_mean, top_M_idx)
  return eigen_values[top_M_idx]
    
if __name__ == '__main__':
    # results = json.load(open("/home/lin10/projects/SkatingJumpClassifier/data/train/d05/alphapose-results.json"))
    results = get_main_skeleton("/home/lin10/projects/SkatingJumpClassifier/data/train/d05")
    hpss = []
    frames = []
    print(len(results))
    for i in range(0, len(results)):
        keypoints = np.array(results[i][1]).reshape(17,3)[:,:2]
        hps_val = hps(keypoints)
        hpss.append(hps_val)
        frames.append(i)
    
    fig = plt.figure()
    ax = plt.subplot(111)

    # der = np.gradient(hpss,frames)

    ax.plot(frames, hpss)
    plt.xticks(np.arange(min(frames), max(frames) + 1, 10))
    plt.grid()
    fig.savefig('./tools/hps_d05_dif.png')