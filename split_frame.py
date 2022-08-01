import cv2
import os 
from absl import flags
from absl import app

flags.DEFINE_string('type', 'flip', 'Path to videos.')
flags.DEFINE_string('filename', None, 'Path to videos.')
FLAGS = flags.FLAGS

def main(_argv):
    cap = cv2.VideoCapture("20220801_Jump_重新命名/{}/{}.MOV".format(FLAGS.type, FLAGS.filename))
    success,image = cap.read()
    i = 0

    os.makedirs("20220801/{}/{}/".format(FLAGS.type, FLAGS.filename), exist_ok=False)

    if not success:
        print("failed to read file: {}".format("20220801_Jump_重新命名/{}/{}.MOV".format(FLAGS.type, FLAGS.filename)))

    while(success):
        cv2.imwrite("20220801/{}/{}/".format(FLAGS.type, FLAGS.filename) + str(i) + '.jpg', image)
        success,image = cap.read()
        i+=1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
