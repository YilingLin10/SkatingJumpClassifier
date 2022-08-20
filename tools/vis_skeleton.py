import numpy as np 
import json
import numpy as np
import os
import cv2
from scipy.spatial import distance
from matplotlib import pyplot as plt
import warnings
import math
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

def draw_xy_time(frames, joints):
    plt.figure()

    xs = []
    ys = []
    for i in range(len(frames)):
        x = joints[i][0]
        y = joints[i][1]
        xs.append(x)
        ys.append(y)
        plt.text(x, y, str(frames[i]), color="red",fontdict={'weight': 'bold', 'size': 8})
    plt.xticks(np.arange(math.floor(min(xs)), math.ceil(max(xs) + 1), math.floor((math.ceil(max(xs) + 1) - math.floor(min(xs))) / 10)))
    plt.yticks(np.arange(math.floor(min(ys)), math.ceil(max(ys) + 1), math.floor((math.ceil(max(ys) + 1) - math.floor(min(ys))) / 10)))
    plt.xlim([math.floor(min(xs)), math.ceil(max(xs)+1)])
    plt.ylim([math.floor(min(ys)), math.ceil(max(ys)+1)])
    plt.savefig('./tools/right_ankle.png')

def draw_y_time(frames, joints):
    fig = plt.figure()
    ax = plt.subplot(111)

    ys = [joint[1] for joint in joints]

    ax.plot(frames, ys)
    plt.xticks(np.arange(min(frames), max(frames) + 1, 10))
    plt.grid()
    fig.savefig('./tools/joint_y.png')

def draw_x_time(frames, joints):
  fig = plt.figure()
  ax = plt.subplot(111)

  xs = [joint[0] for joint in joints]
  der = np.gradient(xs,frames)
  ax.plot(frames, xs)
  plt.xticks(np.arange(min(frames), max(frames) + 1, 10))
  plt.grid()
  fig.savefig('./tools/joint_x.png')

def draw_x_time_leftright(frames, left_joints, right_joints):
  fig = plt.figure()
  ax = plt.subplot(111)

  xs_1 = [joint[0] for joint in left_joints]
  xs_2 = [joint[0] for joint in right_joints]
  ax.plot(frames, xs_1, label="left")
  ax.plot(frames, xs_2, label="right")

  # der_1 = np.gradient(xs_1, frames)
  # der_2 = np.gradient(xs_2, frames)
  # ax.plot(frames, der_1, label="left")
  # ax.plot(frames, der_2, label="right")

  plt.xticks(np.arange(min(frames), max(frames) + 1, 10))
  plt.grid()
  plt.legend()
  fig.savefig('./tools/joint_x_leftright.png')


if __name__ == '__main__':
    # results = get_main_skeleton("/home/lin10/projects/SkatingJumpClassifier/data/train/d05")
    results = get_main_skeleton("/home/lin10/projects/SkatingJumpClassifier/20220801/Loop/Loop_190")
    # results = get_main_skeleton("/home/lin10/projects/tcc/alpha_pose/outputs/alpha_pose_Toe_1")
    frames = []
    left_ankles = []
    right_ankles = []
    left_shoulders = []
    right_shoulders = []
    left_hips = []
    right_hips = []

    hips_differences = []
    shoulders_differences = []

    for i in range(0, len(results)):
        keypoints = np.array(results[i][1]).reshape(17,3)[:,:2]
        left_ankles.append(keypoints[15])
        right_ankles.append(keypoints[16])
        left_shoulders.append(keypoints[5])
        right_shoulders.append(keypoints[6])
        left_hips.append(keypoints[11])
        right_hips.append(keypoints[12])

        hips_differences.append(np.subtract(keypoints[11], keypoints[12]))
        shoulders_differences.append(np.subtract(keypoints[5], keypoints[6]))
        frames.append(i)
    
    # draw_y_time(frames, right_ankles)
    draw_x_time(frames, shoulders_differences)
    # draw_x_time_leftright(frames, left_shoulders, right_shoulders)
    # draw_x_time_leftright(frames, left_hips, right_hips)
    