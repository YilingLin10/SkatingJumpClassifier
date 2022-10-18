import numpy as np
import os 
import torch
from torch.utils.data import Dataset, DataLoader
import glob
import cv2
from pathlib import Path
import pickle
import json

class IceSkatingDataset(Dataset):

    def __init__(self, pkl_file, tag_mapping_file, subtract_feature, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        with open(pkl_file, 'rb') as f:
            self.video_data_list = pickle.load(f)
        self.transform = transform
        self.subtract_feature = subtract_feature
        self.bos_src_embedding = np.random.rand(42) if self.subtract_feature else np.random.rand(34) 
        self.eos_src_embedding = np.random.rand(42) if self.subtract_feature else np.random.rand(34) 
        self.bos_tgt_embedding = np.zeros(42) if self.subtract_feature else np.zeros(34)
        self.eos_tgt_embedding = np.zeros(42) if self.subtract_feature else np.zeros(34)
        self.tag_mapping = json.loads(Path(tag_mapping_file).read_text())
        self._idx2tag = {idx: tag for tag, idx in self.tag_mapping.items()}
    
    def __len__(self):
        return len(self.video_data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        video_name = self.video_data_list[idx]['video_name']
        tags = self.video_data_list[idx]['output']
        if self.subtract_feature:
            features_list = self.video_data_list[idx]['subtraction_features']
            tgt_embeddings_list = self.video_data_list[idx]['tgt_subtraction_embeddings']
        else:
            features_list = self.video_data_list[idx]['features']
            tgt_embeddings_list = self.video_data_list[idx]['tgt_embeddings']
        features_list = np.vstack((self.bos_src_embedding, features_list, self.eos_src_embedding))
        tgt_embeddings_list = np.vstack((self.bos_tgt_embedding, tgt_embeddings_list, self.eos_tgt_embedding))
        tags = np.insert(tags, 0, self.tag2idx("<BOS>"))
        tags = np.append(tags, self.tag2idx("<EOS>"))
        sample = {"keypoints": features_list, "video_name": video_name, "output": tags, "tgt_embeddings": tgt_embeddings_list}
        return sample
    
    def collate_fn(self, samples):
        d_model = samples[0]['keypoints'][0].shape[0]
        to_len = max(len(sample['output']) for sample in samples)
        ids = []
        # pad samples['keypoints'] to to_len
        keypoints = np.zeros((len(samples), to_len, d_model))
        tgt_embeddings = np.zeros((len(samples), to_len, d_model))
        # pad samples['output'] to to_len
        output = (self.tag2idx("<PAD>")) * np.ones((len(samples), to_len))
        mask = np.ones((len(samples), to_len))

        for i in range(len(samples)):
            ids.append(samples[i]['video_name'])
            output_len = len(samples[i]['output'])
            output[i][:output_len] = samples[i]['output']
            keypoints[i][:output_len] = samples[i]['keypoints']
            tgt_embeddings[i][:output_len] = samples[i]['tgt_embeddings']
            mask[i][:output_len] = np.zeros(output_len)

        padded_output = torch.LongTensor(output)
        padded_keypoints = torch.FloatTensor(keypoints)
        padded_tgt_embeddings = torch.FloatTensor(tgt_embeddings)
        mask = torch.tensor(mask).bool()
        return {'ids': ids, 'keypoints': padded_keypoints, 'output': padded_output, 'mask': mask, 'tgt_embeddings': padded_tgt_embeddings}
    
    def tag2idx(self, tag: str):
        return self.tag_mapping[tag]

    def idx2tag(self, idx: int):
        return self._idx2tag[idx]

if __name__ == '__main__':
    dataset = IceSkatingDataset(pkl_file='/home/lin10/projects/SkatingJumpClassifier/data/flipNLoop/alphapose/test.pkl',
                                tag_mapping_file='/home/lin10/projects/SkatingJumpClassifier/data/tag2idx_seq2seq.json',
                                subtract_feature=True)

    dataloader = DataLoader(dataset,batch_size=2,
                        shuffle=False, num_workers=1, collate_fn=dataset.collate_fn)

    for i_batch, sample_batched in enumerate(dataloader):
        # print(i_batch)
        print(sample_batched['ids'][0])
        print(sample_batched['output'][0])
        # print(sample_batched['mask'])
        print(sample_batched['keypoints'][0])
        print(sample_batched['tgt_embeddings'][0])
        print(sample_batched['tgt_embeddings'][1])
        break