import os
from pathlib import Path
from absl import flags
from absl import app

flags.DEFINE_string('action', None, 'axel, flip, loop, lutz, old, salchow, toe')
FLAGS = flags.FLAGS

def main(_argv):
    data_root_dir = "/home/lin10/projects/SkatingJumpClassifier/data"
    video_root_dir = f'/home/lin10/projects/SkatingJumpClassifier/20220801/'
    video_num = 0
    total_frame_num = 0
    for split in ['test']:
        with open(os.path.join(data_root_dir, FLAGS.action, f"{split}_list.txt")) as f:
            videos = f.read().splitlines()
            for video in videos:
                split_video_name = video.split('_')
                if (len(split_video_name) == 3):
                    action = f"{split_video_name[0]}_{split_video_name[1]}"
                else:
                    action = split_video_name[0]
                frames = list(Path(os.path.join(video_root_dir, action, video)).glob("*.jpg"))
                frames = [_ for _ in os.listdir(os.path.join(video_root_dir, action, video)) if _.endswith(".jpg")]
                last_frame = sorted([int(os.path.split(frame)[1].replace('.jpg','')) for frame in frames])[-1]
                video_len = last_frame + 1
                
                total_frame_num += video_len
                video_num += 1
    
    print(f"Average # of frames for {FLAGS.action} dataset: {total_frame_num/video_num}")
       
    
if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass