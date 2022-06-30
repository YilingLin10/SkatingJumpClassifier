import numpy as np
import os 
from torch.utils.data import Dataset, DataLoader
import glob
from PIL import Image
import pandas as pd
import json
import cv2
from pathlib import Path
from helper import *

class IceSkatingDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.jump_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

        for row in self.jump_frame.iterrows():
            videoName = row['Video']
            postive_frame = row["start_jump_1"]
            print(postive_frame)


    # def __len__(self):
    #     return len(self.frames)

    # def __getitem__(self, idx):
    #     if torch.is_tensor(idx):
    #         idx = idx.tolist()
        
    #     video_name, frame = Path(self.frames[idx]).parts[-2:]
    #     frameNumber = int(frame.replace(".jpg",""))
    #     video_data = self.jump_frame.loc[self.jump_frame['Video'] == video_name]

    #     output = 0 

    #     if(frameNumber > int(video_data['start_jump_1']) and int(frameNumber < video_data['end_jump_1'])):
    #         output = 1

    #     if(video_data['start_jump_2'].isnull().bool()):

    #         if(frameNumber >  int(video_data['start_jump_2']) and frameNumber <  int(video_data['end_jump_2'])):
    #             output = 1
        
    #     with open("{}{}/alphapose-results.json".format(self.root_dir, video_name), 'r') as f:
    #         alphaposeResults = json.load(f)
            
    #     pose2d = list(filter(lambda frame: frame["image_id"]=="{}.jpg".format(frameNumber), alphaposeResults))
    #     if len(pose2d) > 1:
    #         pose2d =max(pose2d, key=lambda val: val['score'])
    #         keypoints= pose2d['keypoints']

    #     if(type(pose2d) == list): 
    #         if(len(pose2d) == 0): 
    #             keypoints= torch.zeros(51)
    #         else: keypoints = pose2d[0]['keypoints']
        
    #     keypoints = torch.FloatTensor(keypoints)
    #     # result = {
    #     #     'imgname': "{}.jpg".format(frameNumber),
    #     #     'result': pose2d
    #     # }

    #     # img = cv2.imread("{}/{}/{}.jpg".format(self.root_dir,video_name, frameNumber))
    #     # vis_img = vis_frame(img, result)

    #     # cv2.imwrite("./test-1.png",vis_img)

    #     sample = {"keypoints": keypoints, "video_name": video_name, "frame": frame, "output": torch.tensor(output, dtype=torch.float32)}
    #     return sample


if __name__ == '__main__':
    ice_skating_dataset = IceSkatingDataset(csv_file='/home/calvin/github/skating_classifier/data/iceskatingjump.csv',
                                    root_dir='/home/calvin/github/skating_classifier/data/train/')

    dataloader = DataLoader(ice_skating_dataset,batch_size=32,
                        shuffle=True, num_workers=4)

    
    # for i_batch, sample_batched in enumerate(dataloader):
    #     print(sample_batched)
    #     break





