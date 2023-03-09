import os
import numpy as np
from tqdm import tqdm 
from absl import flags
from absl import app
from read_skeleton import *


flags.DEFINE_string('action', None, 'axel, flip, loop, lutz, old, salchow, toe')
FLAGS = flags.FLAGS

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def pad(array, target_shape):
    return np.pad(
        array,
        [(0, target_shape[i] - array.shape[i]) for i in range(len(array.shape))],
        "constant",
    )
    
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

def main(_argv):
    root_dir='/home/lin10/projects/SkatingJumpClassifier/data/all_jump/alphapose'
    with open(f'{root_dir}/train_list.txt') as f:
        videos = f.read().splitlines()
    for video in tqdm(videos):
        split_video_name = video.split('_')
        if (len(split_video_name) == 3):
            action = f"{split_video_name[0]}_{split_video_name[1]}"
        else:
            action = split_video_name[0]
        
        alphaposeResults = get_main_skeleton(os.path.join("/home/lin10/projects/SkatingJumpClassifier/20220801/", action, video, "alphapose-results.json"))
        keypoints_list = [np.delete(alphaposeResult[1], 2, axis=1) for alphaposeResult in alphaposeResults]
        valid_len = len(keypoints_list)
        
        ############## convert list of float64 np.array to a single float32 np.array ##############
        keypoints_array = np.stack(keypoints_list)
        
        ############## normalize joint coordinates to [-1, 1] #############
        # get height, width of video
        height, width = get_video_hw(action, video)
        normalized_keypoints = normalize_keypoints(keypoints_array[..., :2], width, height).astype(np.float32)
        
        ############## pad (T, 17, 2) to (300, 17, 2) then unsqueeze to (1, 300, 17, 2) ##############
        normalized_keypoints = pad(normalized_keypoints, (300, 17, 2))
        normalized_keypoints = np.expand_dims(normalized_keypoints, axis=0)

        ##### save output
        finetune_data_file = os.path.join("/home/lin10/projects/SkatingJumpClassifier/data/skating_finetune_data_220222.npy")
        with open(finetune_data_file, "wb") as f:
            np.save(f, normalized_keypoints)
    
if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
