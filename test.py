from __future__ import division
from __future__ import print_function

import time
from argparse import ArgumentParser, Namespace
import numpy as np
import datetime
import torch
from torch.utils.data import Dataset, DataLoader

from model.encoder_crf import EncoderCRFModel
from model.agcn_transformer import AGCN_Transformer
from model.stgcn_transformer import STGCN_Transformer
from data.new_dataset import IceSkatingDataset
from utils import eval_seq, eval_crf
from tqdm import trange, tqdm
from typing import Dict
import json
import os
import csv
import yaml

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4, 5, 6"

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    parser.add_argument(
        "--dataset", type=str, default="all_jump", help="old, loop, flip, all_jum"
    )
    parser.add_argument(
        "--config_name", required=True, type=str, help="name of the config file"
    )
    parser.add_argument(
        "--model_path", required=True, type=str, help="path to saved model checkpoints"
    )
    parser.add_argument(
        "--split", type=str, default="test", help='inference on which split of the dataset'
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

def load_config(config_name):
    CONFIG_PATH = "./configs"
    with open(os.path.join(CONFIG_PATH, f"{config_name}.YAML")) as file:
        config = yaml.safe_load(file)
    return config

def get_model(config):
    if config["model_type"] == "encoder-crf":
        return EncoderCRFModel(
                    d_model = config["d_model"],
                    nhead = config["nhead"], 
                    num_encoder_layers = config["num_encoder_layers"],
                    dim_feedforward = config["dim_feedforward"],
                    dropout = 0.1,
                    batch_first = True,
                    num_class = config["num_class"],
                    use_crf = config["use_crf"],
                    fc_before_encoders = config["fc_before_encoders"]
                )
    elif config["model_type"] == "stgcn-encoder-crf":
        return STGCN_Transformer(
                    in_channel = config["in_channel"],
                    hidden_channel = config["hidden_channel"],
                    out_channel = config["out_channel"],
                    nhead = config["nhead"], 
                    num_encoder_layers = config["num_encoder_layers"],
                    dim_feedforward = config["dim_feedforward"],
                    dropout = 0.1,
                    batch_first = True,
                    num_class = config["num_class"],
                    use_crf = config["use_crf"]
                )
def main(args):
    config = load_config(args.config_name)
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    same_seed(args.seed)

    print("================================")
    print("Dataset: {}".format(args.dataset))
    print("Model type: {}".format(config["model_type"]))
    print("feature type: {}".format(config["feature_type"]))
    print("================================")
    ########### LOAD DATA ############
    test_dataset = IceSkatingDataset(dataset=args.dataset,
                                    split="test",
                                    feature_type=config["feature_type"],
                                    model_type=config["model_type"],
                                    use_crf=config["use_crf"],
                                    add_noise=config["add_noise"])
    testloader = DataLoader(test_dataset,batch_size=1, shuffle=False, num_workers=4, collate_fn=test_dataset.collate_fn)
    ############ MODEL && OPTIMIZER && LOSS ############
    model = get_model(config).to(args.device)
    model.eval()
    ckpt_path = os.path.join("/home/lin10/projects/SkatingJumpClassifier/experiments", args.model_path, "save", "transformer_bin_class.pth")
    ckpt = torch.load(ckpt_path)
    # load weights into model
    model.load_state_dict(ckpt['state_dict'])
    best_model_step = ckpt['steps']
    print(f"{best_model_step} steps")

    print("========= GENERATING PREDICTION =========")
    eval_results_test = eval_crf(model, testloader, "test") if config["use_crf"] else eval_seq(model, testloader, "test")
    
    ####### Dump prediction #######
    with open(os.path.join("/home/lin10/projects/SkatingJumpClassifier/experiments", args.model_path, f'{args.dataset}_{args.split}_pred.csv'), 'w') as f:
        csv_writer = csv.writer(f)
        for id, label, pred in zip(eval_results_test['ids'], eval_results_test['labels'], eval_results_test['predictions']):
            csv_writer.writerow([id, label, pred])

if __name__ == "__main__":
    args = parse_args()
    main(args)