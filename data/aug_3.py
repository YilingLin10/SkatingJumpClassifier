import numpy as np
import os 
import glob
from PIL import Image
import pandas as pd
import json
import cv2
from pathlib import Path
import random 
########################################################################
## For each video, create subvideos with various lengths and start_frame
## {
##    "id": "a01-0",  // The id of the created subvideos
##    "video_name": "a01",  // The original video name
##    "start_frame": 0,   // The start frame annotated in the original video
##    "end_frame": 20,    // The end frame annotated in the original video
##    "middle_frames": [6, 8, 11, 13] // The randomly selected frames between start_jump & end_jump
##    "start_jump": 5,
##    "end_jump": 15
## }
########################################################################
csv_file='/home/lin10/projects/SkatingJumpClassifier/data/skatingloop_0801.csv'
root_dir='/home/lin10/projects/SkatingJumpClassifier/20220801/Loop'
videos = list(Path(root_dir).glob("*/"))
jump_frame = pd.read_csv(csv_file)
data = []

for video in videos:
    id = 0
    video_name = Path(video).parts[-1]
    frames = list(Path(f'{root_dir}/{video_name}').glob("*.jpg"))
    video_data = jump_frame.loc[jump_frame['Video'] == video_name]
    if video_data.empty:
        continue
    print("==================")
    print("video name:", video_name)
    # calculate the air_length
    start_jump = int(video_data['start_jump_1'])
    end_jump = int(video_data['end_jump_1'])
    air_length = end_jump - start_jump -1
    last_frame = sorted([int(os.path.split(frame)[1].replace('.jpg','')) for frame in frames])[-1]

    # Orignial videos
    middle_frames = [*range(start_jump + 1, end_jump, 1)]
    d = {
        "id": f'{video_name}-{id}',
        "video_name": video_name,
        "start_frame": 0,
        "end_frame": last_frame,
        "middle_frames": middle_frames,
        "start_jump": start_jump,
        "end_jump": end_jump
    }
    data.append(d)
    id += 1      
    # Augmentation_1: randomly select 4 lengths that are less than jump_length

    # random_lengths = sorted(random.sample(range(5, air_length+1), k=4))
    # for l in random_lengths:
    #     print("length = ", l)
    #     start_frames = sorted(random.sample(range(start_jump - l + 2, start_jump + 1), k= 4))
    #     for i, start_frame in enumerate(start_frames):
    #         # check if start frame is valid
    #         if not (Path(f'{root_dir}{video_name}/{start_frame}.jpg') in frames):
    #             print("start_frame", start_frame , "is not valid")
    #             break
    #         end_frame = end_jump - 1 + l - (start_jump - start_frame + 1)
    #         # check if end_frame is valid
    #         if not (Path(f'{root_dir}{video_name}/{end_frame}.jpg') in frames):
    #             print("end_frame", end_frame , "is not valid")
    #             break
    #         # print(i, "START:", start_frame, "|", "END:", end_frame)
    #         # sample l middle_frames
    #         middle_frames = sorted(random.sample(range(start_jump +1, end_jump), k=l))

    #         d = {
    #             "id": f'{video_name}-{i}',
    #             "video_name": video_name,
    #             "start_frame":start_frame,
    #             "end_frame": end_frame,
    #             "middle_frames": middle_frames,
    #             "start_jump": start_jump,
    #             "end_jump": end_jump
    #         }
    #         data.append(d)
        
    # Augmentation_2: keep all middle frames, drop other frames
    # middle_len * i | middle_len | middle_len * j
    k = int(start_jump / air_length)
    l = int((last_frame - end_jump) / air_length)
    middle_frames = [*range(start_jump + 1, end_jump, 1)]
    for i in range(1, k+1):
        start_frame = start_jump - air_length * i
        for j in range(1, l):
            end_frame = end_jump + air_length * j
            d = {
                "id": f'{video_name}-{id}',
                "video_name": video_name,
                "start_frame":start_frame,
                "end_frame": end_frame,
                "middle_frames": middle_frames,
                "start_jump": start_jump,
                "end_jump": end_jump
            }
            data.append(d)
            id += 1            

print(len(data))

json_file = '/home/lin10/projects/SkatingJumpClassifier/data/skating_data_3.jsonl'
with open(os.path.join(json_file), "w") as f:
    for d in data:
        json.dump(d, f)
        f.write('\n')