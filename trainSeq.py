from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import datetime
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from model.transformer_model import TransformerModel
from model.linear_model import *
from data.dataloader_seq import *
from sklearn.metrics import accuracy_score, confusion_matrix
from utils import eval_seq
from config import CONFIG
from tqdm import trange, tqdm

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda:0"
    )
# parser.add_argument('--epochs', type=int, default=200,
#                     help='Number of epochs to train.')
# parser.add_argument('--eval_step', type=int, default=1000,
#                     help='Number of steps to test.')
# parser.add_argument('--lr', type=float, default=0.01,
#                     help='Initial learning rate.')

args = parser.parse_args()
args.cuda = torch.cuda.is_available()
args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

model = TransformerModel(
                 d_model = 51,
                 nhead = CONFIG.NUM_HEADS, 
                 num_encoder_layers = CONFIG.NUM_ENCODER_LAYERS,
                 dim_feedforward = CONFIG.DIM_FEEDFORWARD,
                 dropout = 0.1,
                 batch_first = True
        ).to(args.device)
optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG.LR, betas=(0.9, 0.999))
loss_fn = torch.nn.BCELoss()

train_dataset = IceSkatingDataset(csv_file='/home/lin10/projects/SkatingJumpClassifier/data/iceskatingjump.csv',
                                    root_dir='/home/lin10/projects/SkatingJumpClassifier/data/train_balance/')

test_dataset = IceSkatingDataset(csv_file='/home/lin10/projects/SkatingJumpClassifier/data/iceskatingjump.csv',
                                    root_dir='/home/lin10/projects/SkatingJumpClassifier/data/test/')

trainloader = DataLoader(train_dataset,batch_size=CONFIG.BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=train_dataset.collate_fn)

testloader = DataLoader(test_dataset,batch_size=CONFIG.BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=test_dataset.collate_fn)

losses = []
accur = []
eval_record = []
steps = 0
best_eval = 0

now = datetime.datetime.now()
tinfo = "%d-%d-%d"%(now.year, now.month, now.day)
exp_path = "./experiments/"
model_name = "transformer"
model_path = exp_path + tinfo + "_" + model_name + "/"

save_path = model_path + "save/"

if not os.path.exists(save_path):
    os.makedirs(save_path)

def nll_loss(predict, y):
    # convert y from (batch_size, max_len) to (batch_size * max_len)
    y = y.view(-1)
    if_padded = (y > -1).float()
    total_token = int(torch.sum(if_padded).item())
    # predict: (batch_size * max_len, num_class)
    predict = predict[range(predict.size(0)), y]* if_padded
    ce = -torch.sum(predict) / total_token
    
    return ce

def accuracy(tags, preds):
    total_token = 0
    correct_token = 0
    correct_sequence = 0
    for tag, pred in zip(tags, preds):
        total_token += len(pred)
        if tag == pred:
            correct_sequence += 1
        for t, p in zip(tag, pred):
            if t == p:
                correct_token += 1
    token_acc = correct_token / total_token
    join_acc = correct_sequence / len(tags)            
    print("Token Accuracy: {:.1%}".format(token_acc))
    print("Join Accuracy: {:.1%}".format(join_acc))
    return token_acc, join_acc

epochss = trange(CONFIG.NUM_EPOCHS, desc="Epoch")
for epochs in epochss:
    ########## TRAINING ##############
    model.train()
    for batch_idx, sample in enumerate(trainloader):
        #calculate output
        keypoints, labels = sample['keypoints'].to(args.device), sample['output'].to(args.device)
        output = model(keypoints)

        #calculate loss
        loss = nll_loss(output, labels)

        #accuracy
        # y_pred = np.rint(output.reshape(-1).detach().numpy())
        # acc = accuracy(y_pred, sample['output'])
        # cm = confusion_matrix(sample['output'], y_pred, labels=[0, 1])

        # tn, fp, fn, tp = cm.ravel()

        #backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        steps += 1
        if steps % 100 == 0:
            print("EPOCH {}\t | LOSS : {}\t".format(epochs+1, loss.detach().item()))
        #     # print("epoch {}\tloss : {}\t accuracy : {}\t tn: {}\t fp: {}\tfn: {}, tp: {}".format(epochs,loss,acc, tn, fp, fn, tp))
    
        ########## EVALUATION ##############
        if steps % CONFIG.EVAL_STEPS == 0:
            model.eval()
            print("STEP-{} Evaluate Training Data ...".format(steps))
            eval_results_train = eval_seq(model, trainloader)

            print("STEP-{} Evaluate Testing Data ...".format(steps))
            eval_results_test = eval_seq(model, testloader)
            eval_record.append({"steps":steps, "train":eval_results_train, "test":eval_results_test})

            # Dump evaluation record
            with open(model_path+'eval_record.json', 'w') as file:
                json.dump(eval_record, file)
        
            if eval_results_test["accuracy"] > best_eval:
                best_eval = eval_results_test["accuracy"]
                print("Saving the best model with accuracy {:.1%}".format(best_eval))
                checkpoint = {
                    'epochs': epochs + 1,
                    'steps': steps,
                    'state_dict': model.state_dict(),
                }
                torch.save(checkpoint, save_path + "transformer_bin_class.pth")

            model.train()
            ########## SAVE MODEL ###########
            if steps % CONFIG.SAVE_STEPS == 0:
                print("STEP-{} Save model".format(steps))
                checkpoint = {
                    'epochs': epochs + 1,
                    'steps': steps,
                    'state_dict': model.state_dict()
                }
                torch.save(checkpoint, save_path + "transformer_bin_class" + str(steps).zfill(4) + ".pth")