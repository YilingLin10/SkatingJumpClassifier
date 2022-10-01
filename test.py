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

from model.transformer_model import TransformerModel
from model.agcn_transformer import AGCN_Transformer
from model.stgcn_transformer import STGCN_Transformer
from data.new_dataset import IceSkatingDataset
from utils import eval_seq, eval_crf
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
        "--dataset", type=str, default="loop", help="old, loop, flip, single_jump"
    )
    parser.add_argument(
        "--split", type=str, default="test", help="train or test"
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
        "--num_layers", type=int, default=CONFIG.NUM_ENCODER_LAYERS, help='number of encoder layers'
    )
    parser.add_argument(
        "--agcn", action="store_true", help="whether or not to use the agcn model"
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
    if CONFIG.USE_CRF:
        print("CRF layer is used.")
    if args.agcn:
        print("Model: AGCN_Transformer")
    elif args.stgcn:
        print("Model: STGCN_Transformer")
    else:
        print("Model: Transformer")
    print("================================")
    ########### LOAD DATA ############
    test_file = "/home/lin10/projects/SkatingJumpClassifier/data/{}/{}/{}.pkl".format(args.dataset, args.estimator, args.split)
    tag2idx_file = "/home/lin10/projects/SkatingJumpClassifier/data/tag2idx.json"
    test_dataset = IceSkatingDataset(pkl_file=test_file, 
                                    tag_mapping_file=tag2idx_file, 
                                    use_crf=CONFIG.USE_CRF, 
                                    add_noise=CONFIG.ADD_NOISE,
                                    subtract_feature=args.subtract_feature)
    testloader = DataLoader(test_dataset,batch_size=CONFIG.BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=test_dataset.collate_fn)
    ############ MODEL && OPTIMIZER && LOSS ############
    if args.agcn:
        args.estimator = "alphapose"
        args.subtract_feature = False
        nhead = 4
        model = AGCN_Transformer(
                    hidden_channel = CONFIG.HIDDEN_CHANNEL,
                    out_channel = CONFIG.OUT_CHANNEL,
                    nhead = nhead, 
                    num_encoder_layers = CONFIG.NUM_ENCODER_LAYERS,
                    dim_feedforward = CONFIG.DIM_FEEDFORWARD,
                    dropout = 0.1,
                    batch_first = True,
                    num_class = CONFIG.NUM_CLASS,
                    use_crf = CONFIG.USE_CRF
                ).to(args.device)
    elif args.stgcn:
        args.estimator = "alphapose"
        args.subtract_feature = False
        nhead = 4
        model = STGCN_Transformer(
                    hidden_channel = CONFIG.HIDDEN_CHANNEL,
                    out_channel = CONFIG.OUT_CHANNEL,
                    nhead = nhead, 
                    num_encoder_layers = CONFIG.NUM_ENCODER_LAYERS,
                    dim_feedforward = CONFIG.DIM_FEEDFORWARD,
                    dropout = 0.1,
                    batch_first = True,
                    num_class = CONFIG.NUM_CLASS,
                    use_crf = CONFIG.USE_CRF
                ).to(args.device)
    else:    
        if args.estimator == "alphapose":
            if args.subtract_feature:
                d_model = 42
                nhead = 3
            else: 
                d_model = 34
                nhead = 2
        else:
            nhead = 2
            if args.subtract_feature:
                d_model = 38
            else:
                d_model = 32
        model = TransformerModel(
                        d_model = d_model,
                        nhead = nhead, 
                        num_encoder_layers = CONFIG.NUM_ENCODER_LAYERS,
                        dim_feedforward = CONFIG.DIM_FEEDFORWARD,
                        dropout = 0.1,
                        batch_first = True,
                        num_class = CONFIG.NUM_CLASS,
                        use_crf = CONFIG.USE_CRF
                ).to(args.device)
    model.eval()
    ckpt_path = args.model_path + "save/transformer_bin_class.pth"
    ckpt = torch.load(ckpt_path)
    # load weights into model
    model.load_state_dict(ckpt['state_dict'])

    print("========= GENERATING PREDICTION =========")
    eval_results_test = eval_crf(model, testloader, "test") if CONFIG.USE_CRF else eval_seq(model, testloader, "test")
    
    # Dump prediction
    with open(args.model_path+ f'{args.dataset}_{args.split}_pred.csv', 'a') as f:
        csv_writer = csv.writer(f)
        for id, label, pred in zip(eval_results_test['ids'], eval_results_test['labels'], eval_results_test['predictions']):
            csv_writer.writerow([id, label, pred])

if __name__ == "__main__":
    args = parse_args()
    main(args)