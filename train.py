from __future__ import division
from __future__ import print_function

import time
from argparse import ArgumentParser, Namespace
import numpy as np
import datetime
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

from data.dataset_seq2seq import IceSkatingDataset
from model.seq2seq_model import Transformer
from model.stgcn_seq2seq import GCN_Transformer
from utils import eval_seq2seq
from config import CONFIG
from tqdm import trange, tqdm
from typing import Dict
import json
import os
import csv
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4, 5, 6"
def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    parser.add_argument(
        "--dataset", type=str, default="loop", help="old, loop, flip, all_jum"
    )
    parser.add_argument(
        "--subtract_feature", action="store_true", help="whether or not to use subtracted features"
    )
    parser.add_argument(
        "--estimator", type=str, default="alphapose", help="alphapose or posetriplet"
    )
    parser.add_argument(
        "--model_path", type=str, default='./experiments/model_1/', help="path to saved model checkpoints"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=CONFIG.NUM_EPOCHS, help='number of epochs'
    )
    parser.add_argument(
        "--stgcn", action="store_true", help="whether or not to use the stgcn model"
    )
    parser.add_argument(
        "--out_channel", type=int, default=CONFIG.OUT_CHANNEL, help="output channel of agcn or stgcn"
    )
    parser.add_argument(
        "--hidden_channel", type=int, default=CONFIG.HIDDEN_CHANNEL, help="output channel of agcn or stgcn"
    )
    args = parser.parse_args()
    return args

def nll_loss(predict, y):
    PAD_IDX = 6
    # convert y from (batch_size, max_len) to (batch_size * max_len)
    y = y.contiguous().view(-1)
    if_padded = (y < PAD_IDX).float()
    total_token = int(torch.sum(if_padded).item())
    # predict: (batch_size * max_len, num_class)
    predict = predict[range(predict.size(0)), y]* if_padded
    ce = -torch.sum(predict) / total_token
    return ce

def same_seed(seed): 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main(args):
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    same_seed(args.seed)

    print("================================")
    print("Dataset: {}".format(args.dataset))
    print("Estimator: {}".format(args.estimator))
    if (args.subtract_feature):
        print("Subtraction features are used")

    print("================================")
    ########### LOAD DATA ############
    train_file = "/home/lin10/projects/SkatingJumpClassifier/data/{}/{}/train.pkl".format(args.dataset, args.estimator)
    test_file = "/home/lin10/projects/SkatingJumpClassifier/data/{}/{}/test.pkl".format(args.dataset, args.estimator)
    tag2idx_file = "/home/lin10/projects/SkatingJumpClassifier/data/tag2idx_seq2seq.json"
    train_dataset = IceSkatingDataset(pkl_file=train_file, 
                                    tag_mapping_file=tag2idx_file, 
                                    subtract_feature=args.subtract_feature)
    test_dataset = IceSkatingDataset(pkl_file=test_file, 
                                    tag_mapping_file=tag2idx_file, 
                                    subtract_feature=args.subtract_feature)
    trainloader = DataLoader(train_dataset,batch_size=CONFIG.BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=train_dataset.collate_fn)
    testloader = DataLoader(test_dataset,batch_size=CONFIG.BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=test_dataset.collate_fn)
    ############ MODEL && OPTIMIZER && LOSS ############
    if args.subtract_feature:
        d_model = 42
        nhead = 3
    else: 
        d_model = 34
        nhead = 2
    if args.stgcn:
        model = GCN_Transformer(
                    hidden_channel = CONFIG.HIDDEN_CHANNEL,
                    out_channel = CONFIG.OUT_CHANNEL,
                    num_class=7,
                    nhead=nhead, 
                    num_encoder_layers=CONFIG.NUM_ENCODER_LAYERS, 
                    num_decoder_layers=2, 
                    dim_feedforward = CONFIG.DIM_FEEDFORWARD,
                    dropout=0.1
                ).to(args.device)
        model_path = f"./experiments/seq2seq/GCN_{args.dataset}_{args.out_channel}/"
    else:
        model = Transformer(
                    num_class=7, 
                    d_model=d_model, 
                    nhead=nhead, 
                    num_encoder_layers=CONFIG.NUM_ENCODER_LAYERS, 
                    num_decoder_layers=2, 
                    dim_feedforward = CONFIG.DIM_FEEDFORWARD,
                    dropout=0.1
                ).to(args.device)
        model_path = f"./experiments/seq2seq/{args.dataset}_{d_model}/"
    writer = SummaryWriter()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3, betas=(0.9, 0.999))
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size= 250, gamma=0.1)

    save_path = model_path + 'save/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    ############ START ITERATION ##############
    losses = []
    accur = []
    eval_record = []
    steps = 0
    best_eval = 0
    epochss = trange(args.num_epochs, desc="Epoch")
    for epochs in epochss:
        # writer.add_scalar('TRAIN/LR', scheduler.optimizer.param_groups[0]['lr'], epochs)
        ################# TRAINING ##############
        model.train()
        for batch_idx, sample in enumerate(trainloader):
            #calculate output
            keypoints, labels, labels_embeddings = sample['keypoints'].to(args.device), sample['output'].to(args.device), sample['tgt_embeddings'].to(args.device)
            
            # shift the tgt by one
            labels_input = labels[:, :-1]
            labels_expected = labels[:, 1:]
            
            # create masks
            src_padding_mask = sample['mask'].to(args.device)
            tgt_mask, tgt_padding_mask = model.create_mask(labels_input)
            tgt_mask, tgt_padding_mask = tgt_mask.to(args.device), tgt_padding_mask.to(args.device)
            
            # generate output
            output = model(keypoints, labels_input, tgt_mask, src_padding_mask, tgt_padding_mask)
            
            #calculate loss
            loss = nll_loss(output, labels_expected)

            #backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            steps += 1
            if steps % 100 == 0:
                writer.add_scalar('TRAIN/LOSS', loss.detach().item(), steps)
                print("STEP-{}\t | LOSS : {}\t".format(steps, loss.detach().item()))
        
            ############### EVALUATION ##############
            if steps % CONFIG.EVAL_STEPS == 0:
                print("========= STEP-{} EVALUATING TESTING DATA =========".format(steps))
                eval_results_test = eval_seq2seq(model, testloader, "test")
                writer.add_scalar('EVAL/ACCURACY', eval_results_test['accuracy'], steps)
                eval_record.append({"steps":steps, "test":{"accuracy":eval_results_test['accuracy']}})

                # Dump evaluation record
                with open(model_path+'eval_record.json', 'w') as file:
                    json.dump(eval_record, file)
                
                # Dump prediction
                with open(model_path+'prediction.csv', 'a') as f:
                    csv_writer = csv.writer(f)
                    for id, label, pred in zip(eval_results_test['ids'], eval_results_test['labels'], eval_results_test['predictions']):
                        csv_writer.writerow([steps, id, label, pred])

                ####### SAVE THE BEST MODEL #########
                if eval_results_test["accuracy"] > best_eval:
                    best_eval = eval_results_test["accuracy"]
                    print("SAVING THE BEST MODEL - ACCURACY {:.1%}".format(best_eval))
                    checkpoint = {
                        'epochs': epochs + 1,
                        'steps': steps,
                        'state_dict': model.state_dict(),
                    }
                    torch.save(checkpoint, save_path + "transformer_bin_class.pth")
        # scheduler.step()

if __name__ == "__main__":
    args = parse_args()
    main(args)