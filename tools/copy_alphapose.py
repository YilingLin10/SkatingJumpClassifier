import shutil, os
from absl import flags
from absl import app

flags.DEFINE_string('action', None, 'axel, flip, loop, lutz, old, salchow, toe')
FLAGS = flags.FLAGS

def main(_argv):
    folder = '/home/lin10/projects/tcc/alpha_pose/outputs'
    videos = sorted(os.listdir(folder))
    videos = [video.replace('alpha_pose_','') for video in videos]
    dest_folder = f'/home/lin10/projects/SkatingJumpClassifier/20220801/{FLAGS.action}'

    for video in videos:
        print(video)
        shutil.copy('{}/alpha_pose_{}/alphapose-results.json'.format(folder, video), '{}/{}'.format(dest_folder, video))
    
if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass