import shutil, os

folder = '/home/lin10/projects/tcc/alpha_pose/outputs'
videos = sorted(os.listdir(folder))
videos = [video.replace('alpha_pose_','') for video in videos]
dest_folder = '/home/lin10/projects/SkatingJumpClassifier/20220801/Loop'

for video in videos:
    print(video)
    shutil.copy('{}/alpha_pose_{}/alphapose-results.json'.format(folder, video), '{}/{}'.format(dest_folder, video))