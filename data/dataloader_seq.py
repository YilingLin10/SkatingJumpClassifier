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
        self.videos = list(Path(self.root_dir).glob("*/"))
    
    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        video_name = Path(self.videos[idx]).parts[-1]
        frames = list(Path(f'{self.root_dir}/{video_name}').glob("*.jpg"))
        video_data = self.jump_frame.loc[self.jump_frame['Video'] == video_name]
        
        output = []
        if(video_data['start_jump_2'].isnull().bool()):
            for frame in frames:
                frameNumber = int(Path(frame).parts[-1].replace(".jpg",""))
                if(frameNumber > int(video_data['start_jump_1']) and frameNumber < int(video_data['end_jump_1'])):
                    output.append(1)
                elif (frameNumber >  int(video_data['start_jump_2']) and frameNumber < int(video_data['end_jump_2'])):
                    output.append(1)
                else:
                    output.append(0)
        else:
            for frame in frames:
                frameNumber = int(Path(frame).parts[-1].replace(".jpg",""))
                if (frameNumber > int(video_data['start_jump_1']) and frameNumber < int(video_data['end_jump_1'])):
                    output.append(1)
                else:
                    output.append(0)

        output = np.array(output)
        # print(video_name, output)
        
        with open("{}{}/alphapose-results.json".format(self.root_dir, video_name), 'r') as f:
            alphaposeResults = json.load(f)
        
        keypoints_list = []
        for frame in frames:

            pose2d = list(filter(lambda frame: frame["image_id"]=="{}.jpg".format(frameNumber), alphaposeResults))
            if len(pose2d) > 1:
                pose2d =max(pose2d, key=lambda val: val['score'])
                keypoints= pose2d['keypoints']

            if(type(pose2d) == list): 
                if(len(pose2d) == 0): 
                    keypoints= torch.zeros(51)
                else: keypoints = pose2d[0]['keypoints']
            
            keypoints_list.append(np.array(keypoints))
        sample = {"keypoints": keypoints_list, "video_name": video_name, "output": output}
        return sample
    
    def collate_fn(self, samples):
        to_len = max(len(sample['output']) for sample in samples)
        # pad samples['keypoints'] to to_len
        keypoints = np.zeros((len(samples), to_len, 51))
        # pad samples['output'] to to_len and create padding_mask
        output = (-1) * np.ones((len(samples), to_len))
        padding_mask = np.zeros((len(samples), to_len))

        for i in range(len(samples)):
            output_len = len(samples[i]['output'])
            output[i][:output_len] = samples[i]['output']
            keypoints[i][:output_len] = samples[i]['keypoints']
            padding_mask[i][:output_len] = [1] * output_len

        padded_output = torch.LongTensor(output)
        padded_keypoints = torch.FloatTensor(keypoints)
        padding_mask = torch.FloatTensor(padding_mask)
        return {'keypoints': padded_keypoints, 'padding_mask': padding_mask, 'output': padded_output}



if __name__ == '__main__':
    dataset = IceSkatingDataset(csv_file='/home/lin10/projects/SkatingJumpClassifier/data/iceskatingjump.csv',
                                    root_dir='/home/lin10/projects/SkatingJumpClassifier/data/train_balance/')

    dataloader = DataLoader(dataset,batch_size=2,
                        shuffle=True, num_workers=1, collate_fn=dataset.collate_fn)

    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch)
        # print(sample_batched['output'])
        # print(sample_batched['keypoints'].size())
        # print(sample_batched['padding_mask'].size())
        break