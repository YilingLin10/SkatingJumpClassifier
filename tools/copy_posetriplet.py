import shutil, os
from absl import flags
from absl import app

flags.DEFINE_string('action', None, 'axel, flip, loop, lutz, old, salchow, toe')
FLAGS = flags.FLAGS

def main(_argv):
    folder = f'/home/lin10/projects/PoseTriplet/estimator_inference/wild_eval/pred3D_pose/{FLAGS.action}'
    video_folder = f'/home/lin10/projects/SkatingJumpClassifier/20220801/{FLAGS.action}'
    videos = sorted(os.listdir(video_folder))

    for video in videos:
        print(video)
        if video =="Flip_54":
            continue
        shutil.copy('{}/{}_pred3D.pkl'.format(folder, video), '{}/{}'.format(video_folder, video))

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass