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
  last_frame = sorted([int(os.path.split(frame)[1].replace('.jpg','')) for frame in frames])[-1]
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

# 0: hip, 1: r_hip, 2: r_knee, 3: r_ankle,
# 4: l_hip, 5: l_knee, 6: l_ankle, 7: spine,
# 8: neck, 9: head, 10: l_shoulder, 11: l_elbow,
# 12: l_wrist, 13: r_shoulder, 14: r_elbow, 15: r_wrist
def subtract_pose3d_features(keypoints):
  # input: [16, 3]
  # output: [6] (x value subtractions)
  subtracted_keypoints = []
  lr_pair = [(4, 1), (5, 2), (6, 3), (10, 13), (11, 14), (12, 15)]
  for pair in lr_pair:
    subtraction = keypoints[pair[0]][0] - keypoints[pair[1]][0]
    subtracted_keypoints.append(subtraction)
  
  return np.array(subtracted_keypoints)

def convert_AlphaOpenposeCoco_to_standard16Joint(pose_x):
  """
  pose_x: nx17x2
  https://zhuanlan.zhihu.com/p/367707179
  """
  hip = 0.5 * (pose_x[:, 11] + pose_x[:, 12])
  neck = 0.5 * (pose_x[:, 5] + pose_x[:, 6])
  spine = 0.5 * (neck + hip)

  # head = 0.5 * (pose_x[:, 1] + pose_x[:, 2])

  head_0 = pose_x[:, 0]  # by noise
  head_1 = (neck - hip)*0.5 + neck  # by backbone
  head_2 = 0.5 * (pose_x[:, 1] + pose_x[:, 2])  # by two eye
  head_3 = 0.5 * (pose_x[:, 3] + pose_x[:, 4])  # by two ear
  head = head_0 * 0.1 + head_1 * 0.6 + head_2 * 0.1 + head_3 * 0.2

  combine = np.stack([hip, spine, neck, head])  # 0 1 2 3 ---> 17, 18, 19 ,20
  combine = np.transpose(combine, (1, 0, 2))
  combine = np.concatenate([pose_x, combine], axis=1)
  reorder = [17, 12, 14, 16, 11, 13, 15, 18, 19, 20, 5, 7, 9, 6, 8, 10]
  standart_16joint = combine[:, reorder]
  return standart_16joint

def get_video_wh(viz_video_path):
  vid = cv2.VideoCapture(viz_video_path)
  height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
  width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)

  return int(width), int(height)

def keypoint_smoothing(keypoints):
  x = keypoints.copy()
  window_length = 5
  polyorder = 2
  out = scipy.signal.savgol_filter(x, window_length, polyorder, deriv=0, delta=1.0, axis=0, mode='interp', cval=0.0)
  return out
      
def keypoint_square_padding(keypoint, video_width, video_height):
  """
  square_padding
  the same as take the longer one as width.
  """
  tmp_keypoint = keypoint.copy()
  if video_width > video_height:  # up down padding
      pad = int((video_width - video_height)*0.5)
      tmp_keypoint[:, :, 1] =  tmp_keypoint[:, :, 1] + pad
      video_height = video_width
  elif video_width < video_height:  # left right padding
      pad = int((video_height - video_width)*0.5)
      tmp_keypoint[:, :, 0] =  tmp_keypoint[:, :, 0] + pad
      video_width = video_height
  else:
      print('image are square, no need padding')
  return tmp_keypoint

def normalize_screen_coordinates(X, w, h):
    assert X.shape[-1] == 2

    # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
    return X / w * 2 - [1, h / w]

def normalize_keypoints(keypoints, video_width, video_height):
  keypoints = convert_AlphaOpenposeCoco_to_standard16Joint(
      keypoints.copy())  # Nx16x2
  keypoints_imgunnorm = keypoint_square_padding(keypoints, video_width, video_height)
  # normlization keypoints
  keypoints_imgnorm = normalize_screen_coordinates(keypoints_imgunnorm[..., :2], w=video_width,
                                                        h=video_height)
  # analysis scale
  pose2dscale = 1
  keypoints_imgnorm = keypoints_imgnorm * pose2dscale
  return keypoints_imgnorm

def get_normalized_keypoints(path_to_json):
  alphaposeResults = get_main_skeleton(path_to_json)
  alphaposeResults = [np.delete(alphaposeResult[1], 2, axis=1) for alphaposeResult in alphaposeResults]
  alphaposeResults = np.array(alphaposeResults).astype(np.float32)
  viz_video_path = get_video_path(path_to_json)
  video_width, video_height = get_video_wh(viz_video_path)
  normalized_keypoints = normalize_keypoints(alphaposeResults, video_width, video_height)

  # np.array: [N, 16, 2] 
  return normalized_keypoints

def get_video_path(path_to_json):
  path = Path(path_to_json)
  video_name = path.parts[-2]
  action_name = video_name.split('_')[0]
  video_path = f'/home/lin10/projects/SkatingJumpClassifier/20220801_Jump_重新命名/{action_name}/{video_name}.MOV'
  return video_path

if __name__ == '__main__':
  # path_to_pickle = "/home/lin10/projects/PoseTriplet/estimator_inference/wild_eval/pred3D_pose/loop/Loop_1_pred3D.pkl"
  # posetripletResults = get_posetriplet(path_to_pickle)
  # subtractions_list = [subtract_pose3d_features(posetripletResult) for posetripletResult in posetripletResults]
  # print(subtractions_list[0].shape)
  # print(len(subtractions_list))

  path_to_json = "/home/lin10/projects/SkatingJumpClassifier/20220801/Lutz/Lutz_2/alphapose-results.json"
  normalized_keypoints = get_normalized_keypoints(path_to_json)
  with open('./lutz_2.pkl', 'wb') as handle:
    pickle.dump(normalized_keypoints, handle, protocol=pickle.HIGHEST_PROTOCOL)



