import cv2
import os 
from absl import flags
from absl import app
import glob

flags.DEFINE_string('action', 'flip', 'Path to videos.')
FLAGS = flags.FLAGS

def main(_argv):
    root_dir = "/home/lin10/projects/SkatingJumpClassifier/20221208"
    video_dir = os.path.join(root_dir, FLAGS.action)
    videos = sorted(os.listdir(video_dir))
    for video in videos:
        print(video)
        frames = sorted(glob.glob(os.path.join(root_dir, FLAGS.action, video, '*.jpg')))
        for frame in frames:
            img = cv2.imread(frame)
            flipped_img = cv2.flip(flipped_img, -1)
            cv2.imwrite(frame, flipped_img)
        print("done")
            
if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass