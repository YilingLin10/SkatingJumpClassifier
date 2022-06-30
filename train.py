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

import model.gcn_model as GCN
from model.linear_model import *
from data.dataloader import *
from sklearn.metrics import accuracy_score, confusion_matrix
from utils import *

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--eval_step', type=int, default=1000,
                    help='Number of steps to test.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')

args = parser.parse_args()
args.cuda = torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

model = Net(input_shape=34)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
loss_fn = torch.nn.BCELoss()

train_ice_skating_dataset = IceSkatingDataset(csv_file='/home/calvin/github/skating_classifier/data/iceskatingjump.csv',
                                    root_dir='/home/calvin/github/skating_classifier/data/train_balance/')

test_ice_skating_dataset = IceSkatingDataset(csv_file='/home/calvin/github/skating_classifier/data/iceskatingjump.csv',
                                    root_dir='/home/calvin/github/skating_classifier/data/test/')

trainloader = DataLoader(train_ice_skating_dataset,batch_size=32, shuffle=True, num_workers=4)

testloader = DataLoader(test_ice_skating_dataset,batch_size=32, shuffle=True, num_workers=4)

losses = []
accur = []
eval_record = []
steps = 0
best_eval = 0

now = datetime.datetime.now()
tinfo = "%d-%d-%d"%(now.year, now.month, now.day)
exp_path = "./experiments/"
model_name = "linear"
model_path = exp_path + tinfo + "_" + model_name + "/"

save_path = model_path + "save/"

if not os.path.exists(save_path):
    os.makedirs(save_path)

for epochs in range(args.epochs):
    for batch_idx, sample in enumerate(trainloader):
        #calculate output
        output = model(sample['keypoints'])

        #calculate loss
        loss = loss_fn(output, sample['output'].reshape(-1,1))

        #accuracy
        y_pred = np.rint(output.reshape(-1).detach().numpy())
        acc = accuracy_score(sample['output'], y_pred)
        cm = confusion_matrix(sample['output'], y_pred, labels=[0, 1])

        tn, fp, fn, tp = cm.ravel()

        #backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        steps += 1

        if steps % 100 == 0:
            print("epoch {}\tloss : {}\t accuracy : {}\t tn: {}\t fp: {}\tfn: {}, tp: {}".format(epochs,loss,acc, tn, fp, fn, tp))
        
        if steps % args.eval_step == 0:
            print("Evaluate Training Data ...")
            eval_results_train = eval(model, trainloader)

            print("Evaluate Testing Data ...")
            eval_results_test = eval(model, testloader)
            eval_record.append({"steps":steps, "train":eval_results_train, "test":eval_results_test})

            print("Dump evaluation record ...")
            with open(model_path+'eval_record.json', 'w') as file:
                json.dump(eval_record, file)
        
            if eval_results_test["accuracy"] > best_eval:
                best_eval = eval_results_test["accuracy"]
                print("Save best model ...")
                checkpoint = {
                    'epochs': epochs + 1,
                    'steps': steps,
                    'state_dict': model.state_dict(),
                }
                torch.save(checkpoint, save_path + "linear_bin_class.pth")
                print("Best Test Accuracy:", best_eval)
                print("------------------------------")

            # ------------ Save Model ------------
            if steps%10000 == 0:
                print("Save model ...")
                checkpoint = {
                    'epochs': epochs + 1,
                    'steps': steps,
                    'state_dict': model.state_dict()
                }
                torch.save(checkpoint, save_path + "linear_bin_class" + str(steps).zfill(4) + ".pth")
    




