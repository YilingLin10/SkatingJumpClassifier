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

######### Dataset for whole videos ######
class IceSkatingDataset(Dataset):

    def __init__(self, csv_file, root_dir, tag_mapping_file, use_crf, add_noise, subtract_feature, pose3d, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.jump_frame = pd.read_csv(csv_file, na_values=["None"])
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
        return len(self.videos)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        video_name = Path(self.videos[idx]).parts[-1]
        frames = list(Path(f'{self.root_dir}/{video_name}').glob("*.jpg"))
        frameNumber_list = sorted([int(Path(frame).parts[-1].replace(".jpg","")) for frame in frames])
        video_data = self.jump_frame.loc[self.jump_frame['Video'] == video_name]
        tags = []
        start_frame = frameNumber_list[0]
        start_jump_1 = int(video_data['start_jump_1'])
        end_jump_1 = int(video_data['end_jump_1'])
        end_frame = frameNumber_list[-1]

        if video_data['start_jump_2'].item() > 0:
            ###### the video includes 2 jumps 
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
        
        if self.pose3d:
            posetripletResults = get_posetriplet("{}{}/{}_pred3D.pkl".format(self.root_dir, video_name, video_name))
            keypoints_list = [posetripletResults[frameNumber].reshape(-1) for frameNumber in frameNumber_list]
        else:
            alphaposeResults = get_main_skeleton("{}{}/alphapose-results.json".format(self.root_dir, video_name))
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


class IceSkatingEmbDataset(Dataset):

    def __init__(self, csv_file, root_dir, tag_mapping_file, use_crf, add_noise, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.jump_frame = pd.read_csv(csv_file, na_values=["None"])
        self.root_dir = root_dir
        self.transform = transform
        self.use_crf = use_crf
        self.add_noise = add_noise
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
        start_frame = frameNumber_list[0]
        start_jump_1 = int(video_data['start_jump_1'])
        end_jump_1 = int(video_data['end_jump_1'])
        end_frame = frameNumber_list[-1]

        if video_data['start_jump_2'].item() > 0:
            ###### the video includes 2 jumps 
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
        
        with open("{}{}/unnormalized_embeddings.csv".format(self.root_dir, video_name)) as f:
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
    dataset = IceSkatingDataset(csv_file='/home/lin10/projects/SkatingJumpClassifier/data/all_jump/info.csv',
                                    root_dir='/home/lin10/projects/SkatingJumpClassifier/data/all_jump/test/',
                                    tag_mapping_file='/home/lin10/projects/SkatingJumpClassifier/data/tag2idx.json',
                                    use_crf=True,
                                    add_noise=False,
                                    subtract_feature=True,
                                    pose3d=False)

    dataloader = DataLoader(dataset,batch_size=64,
                        shuffle=False, num_workers=1, collate_fn=dataset.collate_fn)

    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch)
        # print(sample_batched['output'])
        print(sample_batched['keypoints'].size())
        # break
    
    ### EmbDataset test
    # emb_dataset = IceSkatingEmbDataset(csv_file='/home/lin10/projects/SkatingJumpClassifier/data/iceskatingjump.csv',
    #                                 root_dir='/home/lin10/projects/SkatingJumpClassifier/data/test/',
    #                                 tag_mapping_file='/home/lin10/projects/SkatingJumpClassifier/data/tag2idx.json',
    #                                 use_crf=True,
    #                                 add_noise=False)

    # emb_dataloader = DataLoader(emb_dataset,batch_size=2,
    #                     shuffle=True, num_workers=1, collate_fn=emb_dataset.collate_fn)

    # for i_batch, sample_batched in enumerate(emb_dataloader):
    #     # print(i_batch)
    #     # print(sample_batched['output'])
    #     print(sample_batched['keypoints'].size())
    #     break