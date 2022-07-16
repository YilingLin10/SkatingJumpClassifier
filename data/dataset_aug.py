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

class IceSkatingAugDataset(Dataset):

    def __init__(self, json_file, root_dir, tag_mapping_file, use_crf, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.video_data_list = []
        with open(json_file, 'r') as f:
            for line in f:
                self.video_data_list.append(json.loads(line))
        self.root_dir = root_dir
        self.transform = transform
        self.use_crf = use_crf
        self.videos = list(Path(self.root_dir).glob("*/"))
        self.tag_mapping = json.loads(Path(tag_mapping_file).read_text())
        self._idx2tag = {idx: tag for tag, idx in self.tag_mapping.items()}
    
    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        video_name = self.video_data_list[idx]['id']
        original_video = self.video_data_list[idx]['video_name']
        frames = list(Path(f'{self.root_dir}/{original_video}').glob("*.jpg"))
        start_frame = self.video_data_list[idx]['start_frame']
        end_frame = self.video_data_list[idx]['end_frame']
        start_jump = self.video_data_list[idx]['start_jump']
        end_jump = self.video_data_list[idx]['end_jump']
        middle_frames = self.video_data_list[idx]['middle_frames']

        tags = []
        frameNumber_list = []
        # create tags tags and frameNumber_list
        for i in range(start_frame, start_jump):
            tags.append(self.tag2idx('O'))
            frameNumber_list.append(i)
        tags.append(self.tag2idx('B'))
        frameNumber_list.append(start_jump)
        for i in range(len(middle_frames)):
            tags.append(self.tag2idx('I'))
            frameNumber_list.append(middle_frames[i])
        tags.append(self.tag2idx('E'))
        frameNumber_list.append(end_jump)
        for i in range(end_jump+1, end_frame + 1):
            tags.append(self.tag2idx('O'))
            frameNumber_list.append(i)
        tags = np.array(tags)
        # print(video_name, output)
        
        with open("{}{}/alphapose-results.json".format(self.root_dir, original_video), 'r') as f:
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
        if self.use_crf:
            samples.sort(key=lambda x: len(x['output']), reverse=True)
            to_len = len(samples[0]['output'])
        else:
            to_len = max(len(sample['output']) for sample in samples)
        # pad samples['keypoints'] to to_len
        keypoints = np.zeros((len(samples), to_len, 51))
        # pad samples['output'] to to_len
        output = (-1) * np.ones((len(samples), to_len))
        mask = np.zeros((len(samples), to_len))

        for i in range(len(samples)):
            output_len = len(samples[i]['output'])
            output[i][:output_len] = samples[i]['output']
            keypoints[i][:output_len] = samples[i]['keypoints']
            mask[i][:output_len] = np.ones(output_len)

        padded_output = torch.LongTensor(output)
        padded_keypoints = torch.FloatTensor(keypoints)
        mask = torch.tensor(mask).bool()
        return {'keypoints': padded_keypoints, 'output': padded_output, 'mask': mask}

    def tag2idx(self, tag: str):
        return self.tag_mapping[tag]

    def idx2tag(self, idx: int):
        return self._idx2tag[idx]



if __name__ == '__main__':
    dataset = IceSkatingAugDataset(json_file='/home/lin10/projects/SkatingJumpClassifier/data/skating_data.jsonl',
                                    root_dir='/home/lin10/projects/SkatingJumpClassifier/data/train/',
                                    tag_mapping_file='/home/lin10/projects/SkatingJumpClassifier/data/tag2idx.json',
                                    use_crf=True)

    dataloader = DataLoader(dataset,batch_size=2,
                        shuffle=True, num_workers=1, collate_fn=dataset.collate_fn)

    for i_batch, sample_batched in enumerate(dataloader):
        # print(i_batch)
        print(sample_batched['output'])
        print(sample_batched['mask'])
        # print(sample_batched['keypoints'].size())
        break