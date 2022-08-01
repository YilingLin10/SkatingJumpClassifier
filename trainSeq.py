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

from model.crf_model import TransformerModel
from model.linear_model import *
from data.dataset_aug import *
from data.dataset import *
from sklearn.metrics import accuracy_score, confusion_matrix
from utils import eval_seq, eval_crf
from config import CONFIG
from tqdm import trange, tqdm
from typing import Dict
import json

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda:0"
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

    ########### LOAD DATA ############
    train_dataset = IceSkatingAugDataset(json_file=CONFIG.JSON_FILE,
                                        root_dir=CONFIG.TRAIN_DIR, 
                                        tag_mapping_file=CONFIG.TAG2IDX_FILE, 
                                        use_crf=CONFIG.USE_CRF, 
                                        add_noise=CONFIG.ADD_NOISE)
    test_dataset = IceSkatingDataset(csv_file=CONFIG.CSV_FILE,
                                    root_dir=CONFIG.TEST_DIR,
                                    tag_mapping_file=CONFIG.TAG2IDX_FILE, 
                                    use_crf=CONFIG.USE_CRF,
                                    add_noise=CONFIG.ADD_NOISE)
    trainloader = DataLoader(train_dataset,batch_size=CONFIG.BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=train_dataset.collate_fn)
    testloader = DataLoader(test_dataset,batch_size=CONFIG.BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=test_dataset.collate_fn)
    ############ MODEL && OPTIMIZER && LOSS ############
    model = TransformerModel(
                    d_model = 51,
                    nhead = CONFIG.NUM_HEADS, 
                    num_encoder_layers = CONFIG.NUM_ENCODER_LAYERS,
                    dim_feedforward = CONFIG.DIM_FEEDFORWARD,
                    dropout = 0.1,
                    batch_first = True,
                    num_class = CONFIG.NUM_CLASS,
                    use_crf = CONFIG.USE_CRF
            ).to(args.device)
    writer = SummaryWriter()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG.LR, betas=(0.9, 0.999))

    now = datetime.datetime.now()
    tinfo = "%d-%d-%d"%(now.year, now.month, now.day)
    exp_path = "./experiments/"
    model_name = "transformer"
    model_path = exp_path + tinfo + "_" + model_name + "/"
    save_path = model_path + "save/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    ############ START ITERATION ##############
    losses = []
    accur = []
    eval_record = []
    steps = 0
    best_eval = 0
    epochss = trange(CONFIG.NUM_EPOCHS, desc="Epoch")
    for epochs in epochss:
        ################# TRAINING ##############
        model.train()
        for batch_idx, sample in enumerate(trainloader):
            #calculate output
            keypoints, labels = sample['keypoints'].to(args.device), sample['output'].to(args.device)
            mask = sample['mask'].to(args.device)
            output = model(keypoints, mask)

            #calculate loss
            loss = model.loss_fn(keypoints, labels, mask) if CONFIG.USE_CRF else nll_loss(output, labels)

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
                model.eval()
                print("========= STEP-{} EVALUATING TRAINING DATA =========".format(steps))
                eval_results_train = eval_crf(model, trainloader) if CONFIG.USE_CRF else eval_seq(model, trainloader)
                
                print("========= STEP-{} EVALUATING TESTING DATA =========".format(steps))
                eval_results_test = eval_crf(model, testloader) if CONFIG.USE_CRF else eval_seq(model, testloader)
                writer.add_scalar('TRAIN/ACCURACY', eval_results_train['accuracy'], steps)
                writer.add_scalar('EVAL/ACCURACY', eval_results_test['accuracy'], steps)
                eval_record.append({"steps":steps, "train":eval_results_train, "test":eval_results_test})

                # Dump evaluation record
                with open(model_path+'eval_record.json', 'w') as file:
                    json.dump(eval_record, file)

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

                model.train()

if __name__ == "__main__":
    args = parse_args()
    main(args)