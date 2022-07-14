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

######### Dataset for whole videos ######
class IceSkatingDataset(Dataset):

    def __init__(self, csv_file, root_dir, tag_mapping_file, transform=None):
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
        self.tag_mapping = json.loads(Path(tag_mapping_file).read_text())
        self._idx2tag = {idx: tag for tag, idx in self.tag_mapping.items()}
    
    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        video_name = Path(self.videos[idx]).parts[-1]
        frames = list(Path(f'{self.root_dir}/{video_name}').glob("*.jpg"))
        frameNumber_list = sorted([int(Path(frame).parts[-1].replace(".jpg","")) for frame in frames])
        video_data = self.jump_frame.loc[self.jump_frame['Video'] == video_name]
        
        tags = []
        one = True
        start_frame = frameNumber_list[0]
        start_jump_1 = int(video_data['start_jump_1'])
        end_jump_1 = int(video_data['end_jump_1'])
        end_frame = frameNumber_list[-1]
        ####### TODO: Fix this bug
        if(video_data['start_jump_2'].isnull().bool()):
            ###### the video includes 2 jumps 
            one = False
            start_jump_2 = int(video_data['start_jump_2'])
            end_jump_2 = int(video_data['end_jump_2'])
            for i in range(start_frame, start_jump_1):
                tags.append(self.tag2idx('O'))
            tags.append(self.tag2idx('B'))
            for i in range(start_jump_1 + 1, end_jump_1):
                tags.append(self.tag2idx('I'))
            tags.append(self.tag2idx('E'))
            for i in range(end_jump_1+1, start_jump_2):
                tags.append(self.tag2idx('O'))
            tags.append(self.tag2idx('B'))
            for i in range(start_jump_2 + 1, end_jump_2):
                tags.append(self.tag2idx('I'))
            tags.append(self.tag2idx('E'))
            for i in range(end_jump_2+1, end_frame + 1):
                tags.append(self.tag2idx('O'))
            
        else:
            ###### the video only includes 1 jump
            for i in range(start_frame, start_jump_1):
                tags.append(self.tag2idx('O'))
            tags.append(self.tag2idx('B'))
            for i in range(start_jump_1 + 1, end_jump_1):
                tags.append(self.tag2idx('I'))
            tags.append(self.tag2idx('E'))
            for i in range(end_jump_1+1, end_frame + 1):
                tags.append(self.tag2idx('O'))
        # print(video_name, one, tags)
        tags = np.array(tags)
        
        with open("{}{}/alphapose-results.json".format(self.root_dir, video_name), 'r') as f:
            alphaposeResults = json.load(f)
        
        keypoints_list = []
        for frameNumber in frameNumber_list:
            pose2d = list(filter(lambda frame: frame["image_id"]=="{}.jpg".format(frameNumber), alphaposeResults))
            if len(pose2d) > 1:
                pose2d =max(pose2d, key=lambda val: val['score'])
                keypoints= pose2d['keypoints']

            if(type(pose2d) == list): 
                if(len(pose2d) == 0): 
                    keypoints= torch.zeros(51)
                else: keypoints = pose2d[0]['keypoints']
            
            keypoints_list.append(np.array(keypoints))
        sample = {"keypoints": keypoints_list, "video_name": video_name, "output": tags}
        return sample
    
    def collate_fn(self, samples):
        to_len = max(len(sample['output']) for sample in samples)
        # pad samples['keypoints'] to to_len
        keypoints = np.zeros((len(samples), to_len, 51))
        # pad samples['output'] to to_len
        output = (-1) * np.ones((len(samples), to_len))

        for i in range(len(samples)):
            output_len = len(samples[i]['output'])
            output[i][:output_len] = samples[i]['output']
            keypoints[i][:output_len] = samples[i]['keypoints']

        padded_output = torch.LongTensor(output)
        padded_keypoints = torch.FloatTensor(keypoints)
        return {'keypoints': padded_keypoints, 'output': padded_output}

    def tag2idx(self, tag: str):
        return self.tag_mapping[tag]

    def idx2tag(self, idx: int):
        return self._idx2tag[idx]


if __name__ == '__main__':
    dataset = IceSkatingDataset(csv_file='/home/lin10/projects/SkatingJumpClassifier/data/iceskatingjump.csv',
                                    root_dir='/home/lin10/projects/SkatingJumpClassifier/data/test/',
                                    tag_mapping_file='/home/lin10/projects/SkatingJumpClassifier/data/tag2idx.json')

    dataloader = DataLoader(dataset,batch_size=2,
                        shuffle=True, num_workers=1, collate_fn=dataset.collate_fn)

    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch)
        # print(sample_batched['output'])
        # print(sample_batched['keypoints'].size())
        # break