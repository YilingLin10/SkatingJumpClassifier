from __future__ import division
from __future__ import print_function

from argparse import ArgumentParser, Namespace
import numpy as np
from datetime import datetime
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from tqdm import trange, tqdm
from typing import Dict
import json
import os
import csv
import yaml

from model.encoder_crf import EncoderCRFModel
from model.stgcn_transformer import STGCN_Transformer
from model.poseTransformer_encoder_crf import PoseTransformerEncoderCRF
from data.new_dataset import IceSkatingDataset
from utils import eval_seq, eval_crf

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

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
        "--model_name", required=True, type=str, help="path to saved model checkpoints"
    )
    parser.add_argument(
        "--experiment_name", required=True, type=str, help="path to experiments"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=100, help='number of epochs'
    )
    args = parser.parse_args()
    return args

def nll_loss(predict, y):
    # convert y from (batch_size, max_len) to (batch_size * max_len)
    y = y.view(-1)
    # print(predict)
    if_padded = (y > -1).float()
    total_token = int(torch.sum(if_padded).item())
    # predict: (batch_size * max_len, num_class)
    predict = predict[range(predict.size(0)), y]* if_padded
    ce = -torch.sum(predict) / total_token
    return ce

def mse_loss(outputs, labels, mask):
    """
        outputs: [batch_size * max_len, d_model] (log_probabilities)
        labels: [batch_size, max_len]
        mask = [batch_size, max_len]
    """
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
    elif config["model_type"] == "posetransformer-encoder-crf":
        return PoseTransformerEncoderCRF(
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
    train_dataset = IceSkatingDataset(dataset=args.dataset,
                                split="train",
                                feature_type=config["feature_type"],
                                model_type=config["model_type"],
                                use_crf=config["use_crf"],
                                add_noise=config["add_noise"])
    test_dataset = IceSkatingDataset(dataset=args.dataset,
                                split="test",
                                feature_type=config["feature_type"],
                                model_type=config["model_type"],
                                use_crf=config["use_crf"],
                                add_noise=config["add_noise"])
    trainloader = DataLoader(train_dataset,batch_size=config["batch_size"], shuffle=True, num_workers=4, collate_fn=train_dataset.collate_fn)
    testloader = DataLoader(test_dataset,batch_size=config["batch_size"], shuffle=False, num_workers=4, collate_fn=test_dataset.collate_fn)
    ############ MODEL && OPTIMIZER && LOSS ############
    model = get_model(config).to(args.device)
    model_path = os.path.join(f"./experiments/{args.experiment_name}", args.model_name)
    log_dir = os.path.join(f"./runs/{args.experiment_name}", args.model_name)
    writer = SummaryWriter(log_dir=log_dir)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size= 250, gamma=0.1)
    save_path = os.path.join(model_path, 'save')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    ############ START ITERATION ##############
    losses = []
    accur = []
    eval_record = []
    steps = 0
    best_eval = 0
    epochss = trange(max(args.num_epochs, config["num_epochs"]), desc="Epoch")
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
            loss = model.loss_fn(keypoints, labels, mask) if config["use_crf"] else nll_loss(output, labels)
            # loss = model.loss_fn(keypoints, labels, mask) if config["use_crf"] else mse_loss(output, labels, mask)

            #backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            steps += 1
            if steps % config["log_loss_steps"] == 0:
                writer.add_scalar('TRAIN/LOSS', loss.detach().item(), steps)
                print("STEP-{}\t | LOSS : {}\t".format(steps, loss.detach().item()))
        
            ############### EVALUATION ##############
            if steps % config["eval_steps"] == 0:
                # print("========= STEP-{} EVALUATING TRAINING DATA =========".format(steps))
                # eval_results_train = eval_crf(model, trainloader, "train") if config["use_crf"] else eval_seq(model, trainloader, "train")
                # writer.add_scalar('TRAIN/ACCURACY', eval_results_train['accuracy'], steps)
                # writer.add_scalar('TRAIN/MACRO\\AVG\\RECALL', eval_results_train['macro_avg_recall'], steps)
                # writer.add_scalar('TRAIN/MACRO\\AVG\\PRECISION', eval_results_train['macro_avg_precision'], steps)
                # writer.add_scalar('TRAIN/MACRO\\AVG\\F1_SCORE', eval_results_train['macro_avg_f1'], steps)
                print("========= STEP-{} EVALUATING TESTING DATA =========".format(steps))
                eval_results_test = eval_crf(model, testloader, "test") if config["use_crf"] else eval_seq(model, testloader, "test")
                writer.add_scalar('EVAL/LOSS', eval_results_test['loss'], steps)
                writer.add_scalar('EVAL/ACCURACY', eval_results_test['accuracy'], steps)
                writer.add_scalar('EVAL/MACRO\\AVG\\RECALL', eval_results_test['macro_avg_recall'], steps)
                writer.add_scalar('EVAL/MACRO\\AVG\\PRECISION', eval_results_test['macro_avg_precision'], steps)
                writer.add_scalar('EVAL/MACRO\\AVG\\F1_SCORE', eval_results_test['macro_avg_f1'], steps)
                
                ####### Dump evaluation record #########
                eval_record.append({"steps":steps, 
                                    # "train": {
                                    #     "accuracy":eval_results_train['accuracy'], 
                                    #     "macro_avg_recall":eval_results_train['macro_avg_recall'],
                                    #     "macro_avg_precision":eval_results_train['macro_avg_precision'],
                                    #     "macro_avg_f1":eval_results_train['macro_avg_f1']
                                    #     },
                                    "test": {
                                        "loss": eval_results_test["loss"],
                                        "accuracy":eval_results_test['accuracy'], 
                                        "macro_avg_recall":eval_results_test['macro_avg_recall'],
                                        "macro_avg_precision":eval_results_test['macro_avg_precision'],
                                        "macro_avg_f1":eval_results_test['macro_avg_f1']
                                        }
                                    })
                with open(os.path.join(model_path,'eval_record.json'), 'w') as file:
                    json.dump(eval_record, file)
                
                ####### Dump prediction #########
                # with open(os.path.join(model_path, 'eval_prediction.csv'), 'a') as f:
                #     csv_writer = csv.writer(f)
                #     for id, label, pred in zip(eval_results_test['ids'], eval_results_test['labels'], eval_results_test['predictions']):
                #         csv_writer.writerow([steps, id, label, pred])
                
                select_model_metric = config["select_model_metric"]
                if eval_results_test[select_model_metric] > best_eval:
                    best_eval = eval_results_test[select_model_metric]
                    print("SAVING THE BEST MODEL - {} {:.1%}".format(select_model_metric, best_eval))
                    checkpoint = {
                        'epochs': epochs + 1,
                        'steps': steps,
                        'state_dict': model.state_dict(),
                    }
                    torch.save(checkpoint, os.path.join(save_path, "transformer_bin_class.pth"))
        scheduler.step()

if __name__ == "__main__":
    args = parse_args()
    main(args)