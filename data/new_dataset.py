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

    def __init__(self, dataset, split, feature_type, model_type, use_crf, add_noise, transform=None):
        """
        Args:
            dataset (str): name of the dataset
            split (str): "train" or "test" split
            feature_type (str): "raw_skeletons" or "embeddings"
            use_crf (bool): if True, the inputs need to be sorted during collation
            add_noise (bool): add noise to the features or not
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        with open(os.path.join("/home/lin10/projects/SkatingJumpClassifier/data", dataset, feature_type, f'{split}.pkl'), 'rb') as f:
            self.video_data_list = pickle.load(f)
        self.transform = transform
        self.use_crf = use_crf
        self.add_noise = add_noise
        self.model_type = model_type
        self.feature_type = feature_type
        self.tag_mapping = json.loads(Path("/home/lin10/projects/SkatingJumpClassifier/data/tag2idx.json").read_text())
        self._idx2tag = {idx: tag for tag, idx in self.tag_mapping.items()}
    
    def __len__(self):
        return len(self.video_data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        video_name = self.video_data_list[idx]['video_name']
        tags = self.video_data_list[idx]['output']
        features = self.video_data_list[idx]['features']
        sample = {"keypoints": features, "video_name": video_name, "output": tags}
        return sample
    
    def collate_fn(self, samples):
        d_model = samples[0]['keypoints'][0].shape[0]
        if self.use_crf:
            samples.sort(key=lambda x: len(x['output']), reverse=True)
            if self.model_type == "posetransformer-encoder-crf":
                to_len = 300
            else:
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

if __name__ == '__main__':
    dataset = IceSkatingDataset(dataset="all_jump",
                                split="test",
                                feature_type="embeddings",
                                use_crf=True,
                                add_noise=False)
    dataloader = DataLoader(dataset,batch_size=128,
                        shuffle=False, num_workers=1, collate_fn=dataset.collate_fn)

    for i_batch, sample_batched in enumerate(dataloader):
        # print(i_batch)
        # print(sample_batched['ids'][0])
        print(sample_batched['output'].shape)
        print(sample_batched['mask'].shape)
        print(sample_batched['keypoints'].shape)
        break