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
import csv
from .read_skeleton import get_main_skeleton, subtract_features, get_posetriplet

class IceSkatingAugDataset(Dataset):

    def __init__(self, json_file, root_dir, tag_mapping_file, use_crf, add_noise, subtract_feature, pose3d, transform=None):
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
        self.add_noise = add_noise
        self.subtract_feature = subtract_feature
        self.pose3d = pose3d
        self.videos = list(Path(self.root_dir).glob("*/"))
        self.tag_mapping = json.loads(Path(tag_mapping_file).read_text())
        self._idx2tag = {idx: tag for tag, idx in self.tag_mapping.items()}
    
    def __len__(self):
        return len(self.video_data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        video_name = self.video_data_list[idx]['id']
        original_video = self.video_data_list[idx]['video_name']
        frames = list(Path(f'{self.root_dir}/{original_video}').glob("*.jpg"))
        start_frame = self.video_data_list[idx]['start_frame']
        end_frame = self.video_data_list[idx]['end_frame']
        start_jump_1 = self.video_data_list[idx]['start_jump_1']
        end_jump_1 = self.video_data_list[idx]['end_jump_2']
        start_jump_2 = self.video_data_list[idx]['start_jump_2']
        end_jump_2 = self.video_data_list[idx]['end_jump_2']
        tags = []
        frameNumber_list = []
        # create tags tags and frameNumber_list
        for i in range(start_frame, start_jump_1):
            tags.append(self.tag2idx('O'))
            frameNumber_list.append(i)
        tags.append(self.tag2idx('B'))
        frameNumber_list.append(start_jump_1)
        for i in range(start_jump_1+1, end_jump_1):
            tags.append(self.tag2idx('I'))
            frameNumber_list.append(i)
        tags.append(self.tag2idx('E'))
        frameNumber_list.append(end_jump_1)
        if start_jump_2 == -1:
            ## 1 Jump
            for i in range(end_jump_1+1, end_frame + 1):
                tags.append(self.tag2idx('O'))
                frameNumber_list.append(i)
        else:
            ## 2 jumps
            for i in range(end_jump_1 + 1, start_jump_2):
                tags.append(self.tag2idx('O'))
                frameNumber_list.append(i)
            tags.append(self.tag2idx('B'))
            frameNumber_list.append(start_jump_2)
            for i in range(start_jump_2 + 1, end_jump_2):
                tags.append(self.tag2idx('I'))
                frameNumber_list.append(i)
            tags.append(self.tag2idx('E'))
            frameNumber_list.append(end_jump_2)
            for i in range(end_jump_2 + 1, end_frame + 1):
                tags.append(self.tag2idx('O'))
                frameNumber_list.append(i)
        tags = np.array(tags)
        # print(video_name, output)
        if self.pose3d:
            posetripletResults = get_posetriplet("{}{}/{}_pred3D.pkl".format(self.root_dir, original_video, original_video))
            # print(video_name, len(posetripletResults))
            keypoints_list = [posetripletResults[frameNumber].reshape(-1) for frameNumber in frameNumber_list]
        else:
            alphaposeResults = get_main_skeleton("{}{}/alphapose-results.json".format(self.root_dir, original_video))
            assert len(frames) == len(alphaposeResults), "{} error".format(video_name)
            subtractions_list = [subtract_features(alphaposeResults[frameNumber][1]) for frameNumber in frameNumber_list]
            keypoints_list = [np.delete(alphaposeResults[frameNumber][1], 2, axis=1).reshape(-1) for frameNumber in frameNumber_list]
            if self.subtract_feature:
                features_list = np.append(keypoints_list, subtractions_list, axis=1)  
            else:
                features_list = keypoints_list
        sample = {"keypoints": features_list, "video_name": video_name, "output": tags}
        return sample
    
    def collate_fn(self, samples):
        d_model = samples[0]['keypoints'][0].shape[0]
        if self.use_crf:
            samples.sort(key=lambda x: len(x['output']), reverse=True)
            to_len = len(samples[0]['output'])
        else:
            to_len = max(len(sample['output']) for sample in samples)
        ids = []
        # pad samples['keypoints'] to to_len
        keypoints = np.zeros((len(samples), to_len, d_model))
        # pad samples['output'] to to_len
        output = (-1) * np.ones((len(samples), to_len))
        mask = np.zeros((len(samples), to_len))

        for i in range(len(samples)):
            ids.append(samples[i]['video_name'])
            output_len = len(samples[i]['output'])
            output[i][:output_len] = samples[i]['output']
            keypoints[i][:output_len] = samples[i]['keypoints']
            mask[i][:output_len] = np.ones(output_len)

        padded_output = torch.LongTensor(output)
        padded_keypoints = torch.FloatTensor(keypoints)
        ### Add noise to keypoints
        if self.add_noise:
            padded_keypoints = padded_keypoints + torch.randn(padded_keypoints.size(0), padded_keypoints.size(1), padded_keypoints.size(2))
        mask = torch.tensor(mask).bool()
        return {'ids': ids, 'keypoints': padded_keypoints, 'output': padded_output, 'mask': mask}
    
    def tag2idx(self, tag: str):
        return self.tag_mapping[tag]

    def idx2tag(self, idx: int):
        return self._idx2tag[idx]

class IceSkatingAugEmbDataset(Dataset):

    def __init__(self, json_file, root_dir, tag_mapping_file, use_crf, add_noise, transform=None):
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
        self.add_noise = add_noise
        self.videos = list(Path(self.root_dir).glob("*/"))
        self.tag_mapping = json.loads(Path(tag_mapping_file).read_text())
        self._idx2tag = {idx: tag for tag, idx in self.tag_mapping.items()}
    
    def __len__(self):
        return len(self.video_data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        video_name = self.video_data_list[idx]['id']
        original_video = self.video_data_list[idx]['video_name']
        frames = list(Path(f'{self.root_dir}/{original_video}').glob("*.jpg"))
        start_frame = self.video_data_list[idx]['start_frame']
        end_frame = self.video_data_list[idx]['end_frame']
        start_jump_1 = self.video_data_list[idx]['start_jump_1']
        end_jump_1 = self.video_data_list[idx]['end_jump_2']
        start_jump_2 = self.video_data_list[idx]['start_jump_2']
        end_jump_2 = self.video_data_list[idx]['end_jump_2']

        tags = []
        frameNumber_list = []
        # create tags tags and frameNumber_list
        for i in range(start_frame, start_jump_1):
            tags.append(self.tag2idx('O'))
            frameNumber_list.append(i)
        tags.append(self.tag2idx('B'))
        frameNumber_list.append(start_jump_1)
        for i in range(start_jump_1+1, end_jump_1):
            tags.append(self.tag2idx('I'))
            frameNumber_list.append(i)
        tags.append(self.tag2idx('E'))
        frameNumber_list.append(end_jump_1)
        if start_jump_2 == -1:
            ## 1 Jump
            for i in range(end_jump_1+1, end_frame + 1):
                tags.append(self.tag2idx('O'))
                frameNumber_list.append(i)
        else:
            ## 2 jumps
            for i in range(end_jump_1 + 1, start_jump_2):
                tags.append(self.tag2idx('O'))
                frameNumber_list.append(i)
            tags.append(self.tag2idx('B'))
            frameNumber_list.append(start_jump_2)
            for i in range(start_jump_2 + 1, end_jump_2):
                tags.append(self.tag2idx('I'))
                frameNumber_list.append(i)
            tags.append(self.tag2idx('E'))
            frameNumber_list.append(end_jump_2)
            for i in range(end_jump_2 + 1, end_frame + 1):
                tags.append(self.tag2idx('O'))
                frameNumber_list.append(i)
        tags = np.array(tags)
        
        with open("{}{}/unnormalized_embeddings.csv".format(self.root_dir, original_video)) as f:
            csv_reader = csv.reader(f, delimiter=',')
            all_embeddings = []
            for embeddings in csv_reader:
                all_embeddings.append(np.array(embeddings))

        embeddings_list = []
        for frameNumber in frameNumber_list:
            frame_embeddings = all_embeddings[frameNumber]
            embeddings_list.append(np.array(frame_embeddings))
        sample = {"keypoints": embeddings_list, "video_name": video_name, "output": tags}
        return sample
    
    def collate_fn(self, samples):
        if self.use_crf:
            samples.sort(key=lambda x: len(x['output']), reverse=True)
            to_len = len(samples[0]['output'])
        else:
            to_len = max(len(sample['output']) for sample in samples)
        ids = []
        # pad samples['keypoints'] to to_len
        keypoints = np.zeros((len(samples), to_len, 16))
        # pad samples['output'] to to_len
        output = (-1) * np.ones((len(samples), to_len))
        mask = np.zeros((len(samples), to_len))

        for i in range(len(samples)):
            ids.append(samples[i]['video_name'])
            output_len = len(samples[i]['output'])
            output[i][:output_len] = samples[i]['output']
            keypoints[i][:output_len] = samples[i]['keypoints']
            mask[i][:output_len] = np.ones(output_len)

        padded_output = torch.LongTensor(output)
        padded_keypoints = torch.FloatTensor(keypoints)
        ### Add noise to keypoints
        if self.add_noise:
            padded_keypoints = padded_keypoints + torch.randn(padded_keypoints.size(0), padded_keypoints.size(1), padded_keypoints.size(2))
        mask = torch.tensor(mask).bool()
        return {'ids': ids, 'keypoints': padded_keypoints, 'output': padded_output, 'mask': mask}

    def tag2idx(self, tag: str):
        return self.tag_mapping[tag]

    def idx2tag(self, idx: int):
        return self._idx2tag[idx]



if __name__ == '__main__':
    dataset = IceSkatingAugDataset(json_file='/home/lin10/projects/SkatingJumpClassifier/data/all_jump/train_aug3.jsonl',
                                    root_dir='/home/lin10/projects/SkatingJumpClassifier/data/all_jump/train/',
                                    tag_mapping_file='/home/lin10/projects/SkatingJumpClassifier/data/tag2idx.json',
                                    use_crf=True,
                                    add_noise=False,
                                    subtract_feature=True,
                                    pose3d=False)

    dataloader = DataLoader(dataset,batch_size=64,
                        shuffle=False, num_workers=1, collate_fn=dataset.collate_fn)

    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch)
        # print(sample_batched['ids'])
        # print(sample_batched['output'])
        # print(sample_batched['mask'])
        # print(sample_batched['keypoints'].size())
        

    ### AugEmbDataset
    # emb_dataset = IceSkatingAugEmbDataset(json_file='/home/lin10/projects/SkatingJumpClassifier/data/skating_data_4.jsonl',
    #                                 root_dir='/home/lin10/projects/SkatingJumpClassifier/20220801/Loop/',
    #                                 tag_mapping_file='/home/lin10/projects/SkatingJumpClassifier/data/tag2idx.json',
    #                                 use_crf=True,
    #                                 add_noise=True)

    # emb_dataloader = DataLoader(emb_dataset,batch_size=2,
    #                     shuffle=True, num_workers=1, collate_fn=emb_dataset.collate_fn)

    # for i_batch, sample_batched in enumerate(emb_dataloader):
    #     # print(i_batch)
    #     # print(sample_batched['output'])
    #     # print(sample_batched['mask'])
    #     print(sample_batched['keypoints'].size())
    #     break