import os
import pickle
from absl import flags
from absl import app

"""
    This file can be used to modify the tags for a specific video if there's an annotation mistake.
"""

flags.DEFINE_string('action', None, 'axel, flip, loop, lutz, old, salchow, toe')
flags.DEFINE_string('split', "test", 'train or test')
flags.DEFINE_string('feature_type', "poem_embeddings", 'embeddings or raw_skeletons or poem_embeddings')
FLAGS = flags.FLAGS

def get_data(action, feature_type, split):
    data_file = os.path.join(f'/home/lin10/projects/SkatingJumpClassifier/data/{action}', feature_type, f"{split}.pkl")
    with open(data_file, 'rb') as f:
        data_list = pickle.load(f)
    return data_list

def writePickle(samples, action, feature_type, split):
    pickle_file = os.path.join(f'/home/lin10/projects/SkatingJumpClassifier/data/{action}', feature_type, f"{split}.pkl")
    with open(pickle_file, "wb") as f:
        pickle.dump(samples, f)

"""
    video: the name of the video which has an annotation error
    start_frame: the start_frame of the missing flight phase
    end_frame: the end_frame of the missing flight phase
"""
def main(_argv):
    video = "Axel_19"
    start_frame = 98
    end_frame = 107
    
    video_data_list = get_data(FLAGS.action, FLAGS.feature_type, FLAGS.split)
    for video_data in video_data_list:
        if video in video_data["video_name"]:
            video_data["output"][start_frame] = 1
            video_data["output"][start_frame+1: end_frame] = 2
            video_data["output"][end_frame] = 3
    writePickle(video_data_list, FLAGS.action, FLAGS.feature_type, FLAGS.split)
    
if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass