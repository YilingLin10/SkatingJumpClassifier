import torch 
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

############ Evaluation ############
def eval(model, dataset):
    acc_list =[]
    tn_total = 0
    fp_total = 0
    fn_total = 0
    tp_total = 0
    for it, batch in enumerate(dataset):
        with torch.no_grad():
            output = model(batch['keypoints'])
            y_pred = np.rint(output.reshape(-1).detach().numpy())
            acc = accuracy_score(batch['output'], y_pred)
            cm = confusion_matrix(batch['output'], y_pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            acc_list.append(acc)
            tn_total += tn
            fp_total += fp
            fn_total += fn
            tp_total += tp

    return {"accuracy": float(np.array(acc_list).mean()),"tn": int(tn_total),"fp": int(fp_total) , "fn": int(fn_total), "tp": int(tp_total)}

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
    # print("Token Accuracy: {:.1%}".format(token_acc))
    # print("Join Accuracy: {:.1%}".format(join_acc))
    return token_acc, join_acc

def eval_seq(model, dataset):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    token_acc_list =[]
    join_acc_list = []
    val_preds, val_labels = [], []
    model.eval()
    for it, batch in enumerate(dataset):
        with torch.no_grad():
            batch_size = batch['keypoints'].size(0)
            keypoints, labels_list = batch['keypoints'].to(device), batch['output'].to(device)
            output = model(keypoints)
            loss = nll_loss(output, labels_list)
            
            batch_labels = []
            for labels in labels_list:
                single_labels = []
                for label in labels:
                    if label.item() < 0:
                        break
                    single_labels.append(label.item())
                batch_labels.append(single_labels)
            val_labels.extend(batch_labels)
            
            pred = torch.exp(output)
            pred = torch.max(output,dim=1)[1]
            pred = pred.view(batch_size, -1)
            batch_prediction = []
            for frames, labels in zip(keypoints, pred):
                single_prediction = []
                for i, frame in enumerate(frames):
                    if torch.any(frame):
                        single_prediction.append(labels[i].item())
            
                batch_prediction.append(single_prediction)
            
            val_preds.extend(batch_prediction)
            # print("==============")
            # print("PREDICTION")
            # print(batch_prediction)
            token_acc, join_acc = accuracy(val_labels, val_preds)
            token_acc_list.append(token_acc)
            join_acc_list.append(join_acc)
    print("TOKEN ACCURACY: {:.1%}".format(float(np.array(token_acc_list).mean())))
    print("JOIN ACCURACY: {:.1%}".format(float(np.array(join_acc_list).mean())))
    print("=============END OF EVALUATION=============")
    return {"accuracy": float(np.array(token_acc_list).mean())}