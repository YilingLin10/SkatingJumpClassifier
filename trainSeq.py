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

from model.crf_model import TransformerModel
from model.linear_model import *
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
        "--num_layers", type=int, default=CONFIG.NUM_ENCODER_LAYERS, help='number of encoder layers'
    )
    args = parser.parse_args()
    return args

def nll_loss(predict, y):
    # convert y from (batch_size, max_len) to (batch_size * max_len)
    y = y.view(-1)
    if_padded = (y > -1).float()
    total_token = int(torch.sum(if_padded).item())
    # predict: (batch_size * max_len, num_class)
    predict = predict[range(predict.size(0)), y]* if_padded
    ce = -torch.sum(predict) / total_token
    return ce

def mse_loss(outputs, labels, mask):
    # outputs: [batch_size * max_len, d_model] (log_probabilities)
    # labels: [batch_size, max_len]
    # mask = [batch_size, max_len]
    labels = labels.view(-1).float().clone().detach().requires_grad_(True)
    mask = mask.view(-1)
    predictions = torch.argmax(outputs, dim=1).float().clone().detach().requires_grad_(True)
    loss = nn.MSELoss(reduction="none")
    loss_val = loss(predictions, labels)
    loss_val = (loss_val * mask.float()).sum()
    non_zero_elements = mask.sum()
    mse_loss_val = loss_val / non_zero_elements
    return Variable(mse_loss_val, requires_grad = True)

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
    print("Subtraction_features: {}".format(args.subtract_feature))
    print("USE_CRF: {}".format(CONFIG.USE_CRF))
    print("================================")
    ########### LOAD DATA ############
    train_file = "/home/lin10/projects/SkatingJumpClassifier/data/{}/cache/train.pkl".format(args.dataset)
    test_file = "/home/lin10/projects/SkatingJumpClassifier/data/{}/cache/test.pkl".format(args.dataset)
    tag2idx_file = "/home/lin10/projects/SkatingJumpClassifier/data/tag2idx.json"
    train_dataset = IceSkatingDataset(pkl_file=train_file, 
                                    tag_mapping_file=tag2idx_file, 
                                    use_crf=CONFIG.USE_CRF, 
                                    add_noise=CONFIG.ADD_NOISE,
                                    subtract_feature=args.subtract_feature)
    test_dataset = IceSkatingDataset(pkl_file=test_file, 
                                    tag_mapping_file=tag2idx_file, 
                                    use_crf=CONFIG.USE_CRF, 
                                    add_noise=CONFIG.ADD_NOISE,
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
    model = TransformerModel(
                    d_model = d_model,
                    nhead = nhead, 
                    num_encoder_layers = args.num_layers,
                    dim_feedforward = CONFIG.DIM_FEEDFORWARD,
                    dropout = 0.1,
                    batch_first = True,
                    num_class = CONFIG.NUM_CLASS,
                    use_crf = CONFIG.USE_CRF
            ).to(args.device)
    writer = SummaryWriter()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG.LR, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size= 1000, gamma=0.1)

    model_path = args.model_path
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
        writer.add_scalar('TRAIN/LR', scheduler.optimizer.param_groups[0]['lr'], epochs)
        ################# TRAINING ##############
        model.train()
        for batch_idx, sample in enumerate(trainloader):
            #calculate output
            keypoints, labels = sample['keypoints'].to(args.device), sample['output'].to(args.device)
            mask = sample['mask'].to(args.device)
            output = model(keypoints, mask)

            #calculate loss
            # loss = model.loss_fn(keypoints, labels, mask) if CONFIG.USE_CRF else nll_loss(output, labels)
            loss = model.loss_fn(keypoints, labels, mask) if CONFIG.USE_CRF else mse_loss(output, labels, mask)

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
                print("========= STEP-{} EVALUATING TRAINING DATA =========".format(steps))
                eval_results_train = eval_crf(model, trainloader, "train") if CONFIG.USE_CRF else eval_seq(model, trainloader, "train")
                
                print("========= STEP-{} EVALUATING TESTING DATA =========".format(steps))
                eval_results_test = eval_crf(model, testloader, "test") if CONFIG.USE_CRF else eval_seq(model, testloader, "test")
                writer.add_scalar('TRAIN/ACCURACY', eval_results_train['accuracy'], steps)
                writer.add_scalar('TRAIN/MSE', eval_results_train['mse'], steps)
                writer.add_scalar('EVAL/ACCURACY', eval_results_test['accuracy'], steps)
                writer.add_scalar('EVAL/MSE', eval_results_test['mse'], steps)
                eval_record.append({"steps":steps, "train":{"accuracy":eval_results_train['accuracy'], "mse":eval_results_train['mse']}, "test":{"accuracy":eval_results_test['accuracy'], "mse":eval_results_test['mse']}})

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
        scheduler.step()

if __name__ == "__main__":
    args = parse_args()
    main(args)