import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import datetime
import pandas as pd
import glob
import cv2
import json
from absl import app
from absl import flags

# Specify GPUs
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4, 5, 6"

flags.DEFINE_string('action', 'loop', 'action_name')
flags.DEFINE_string('video_name', 'Loop_187', 'name of the video to test')
flags.DEFINE_string('estimator', 'alphapose', 'name of the estimator')
FLAGS = flags.FLAGS

model_list = ['loop_alphapose_42', 'loop_alphapose_34']

# 1. get_prediction
# 2. visualize

def get_answer(model_name, video_name):
    path_to_prediction = f'./experiments/{model_name}/{FLAGS.action}_test_pred.csv'
    predictions = pd.read_csv(path_to_prediction, header=None, usecols=[0, 1, 2], names=['video_name', 'answer', 'prediction'])
    video_prediction = predictions.loc[predictions['video_name'] == video_name]
    answer = json.loads(video_prediction['answer'].item())
    return answer

def get_prediction(model_name, video_name):
    path_to_prediction = f'./experiments/{model_name}/{FLAGS.action}_test_pred.csv'
    predictions = pd.read_csv(path_to_prediction, header=None, usecols=[0, 1, 2], names=['video_name', 'answer', 'prediction'])
    video_prediction = predictions.loc[predictions['video_name'] == video_name]
    prediction = json.loads(video_prediction['prediction'].item())
    return prediction

def get_frames(video_name):
    root_dir = "/home/lin10/projects/SkatingJumpClassifier/20220801"
    action_name = video_name.split('_')[0]
    video_dir = f"{root_dir}/{action_name}/{video_name}"
    files = os.listdir(video_dir)
    imgs = [f for f in files if '.jpg' in f]
    for img in imgs:
        new_name = '{0:04d}'.format(int(img.rstrip('.jpg')))+'.jpg'
        old = os.path.join(video_dir, img)
        new = os.path.join(video_dir, new_name)
        os.rename(old, new)

    # Preprocessing raw frames
    framefiles = sorted(glob.glob(os.path.join(video_dir, '*.jpg')))
    frames_raw = []
    for framefile in framefiles:
        frame_raw = cv2.imread(framefile)
        frame_raw = cv2.cvtColor(frame_raw, cv2.COLOR_BGR2RGB)
        frames_raw.append(frame_raw)
    
    return frames_raw

def gray_scale(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return gray

def gen_result(raw_frames, answer, prediction_list):
    frames = raw_frames.copy()
    # now_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # OUTPUT_PATH = './result/vis_{}_{}.mp4'.format(FLAGS.video_name, now_time)
    OUTPUT_PATH = './result/vis_{}.mp4'.format(FLAGS.video_name)
    
    label_list = prediction_list
    label_list.insert(0, answer)
    
    jumps_list = []
    for label in label_list:
        jumps = [i for i, lab in enumerate(label) if lab != 0]
        jumps_list.append(jumps)

    # Create subplots
    nrows = len(label_list)
    height, width, channels = raw_frames[0].shape

    fig, ax = plt.subplots(
        ncols=nrows,
        figsize=(11.25* nrows, 25),
        tight_layout=True)
  
    def unnorm(frame):
        # min_v = frame.min()
        # max_v = frame.max()
        # frame = (frame - min_v) / (max_v - min_v)
        return frame

    ims = []
    answer_idx = 0

    def init():
        # k = 0
        for k in range(nrows):
            ims.append(ax[k].imshow(
                unnorm(frames[0]), 'gray'
            ))
            ax[k].grid(False)
            ax[k].set_xticks([])
            ax[k].set_yticks([])
        return ims

    ims = init()

    def update(i):
        for j, jumps in enumerate(jumps_list):
            if i in jumps:
                ims[j].set_data(unnorm(frames[i]))
            else:
                ims[j].set_data(unnorm(gray_scale(frames[i])))

        ax[0].set_title('GROUND TRUTH', fontsize = 36)
        for j, model_name in enumerate(model_list):
            ax[j+1].set_title(f'{model_name}', fontsize = 36)
        fig.suptitle('FRAME {}'.format(i), fontsize = 36, y=0.95)
        plt.tight_layout()
        return ims

    #Create animation
    print("Generating animation.....")
    anim = FuncAnimation(
        fig,
        update,
        frames=np.arange(len(raw_frames)),
        interval=100,
        blit=False)
    anim.save(OUTPUT_PATH, dpi=40)

    plt.close()

def main(_argv):
    answer = get_answer(model_list[0], FLAGS.video_name)
    prediction_list = []
    for model_name in model_list:
        prediction = get_prediction(model_name, FLAGS.video_name)
        prediction_list.append(prediction)
    raw_frames = get_frames(FLAGS.video_name)
    gen_result(raw_frames, answer, prediction_list)

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass