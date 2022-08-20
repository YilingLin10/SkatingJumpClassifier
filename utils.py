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
    preds_list, labels_list = [], []
    output_ids, output_preds, output_labels = [], [], []
    for it, batch in enumerate(dataset):
        with torch.no_grad():
            batch_size = batch['keypoints'].size(0)
            keypoints, batch_labels = batch['keypoints'].to(device), batch['output'].to(device)
            mask = batch['mask'].to(device)
            output = model(keypoints, mask)

            output_ids.extend(batch['ids'])          
            for labels, m in zip(batch_labels, mask):
                labels_list.append(labels[m==True].tolist())
                output_labels.append(labels[m==True].tolist())
            
            batch_preds = torch.argmax(output,dim=1)
            ## reshape: (batch * seq_len) ---> (batch, seq_len)
            batch_preds = batch_preds.view(batch_size, -1)

            for preds, m in zip(batch_preds, mask):
                preds_list.append(preds[m==True].tolist())
                output_preds.append(preds[m==True].tolist())

    token_acc, join_acc = accuracy(labels_list, preds_list)
    print("************")
    print("PREDICTION")
    print(labels_list[0])
    print(preds_list[0])
    print("************")
    print("TOKEN ACCURACY: {:.1%}".format(token_acc))
    print("JOIN ACCURACY: {:.1%}".format(join_acc))
    print("================= END OF EVALUATION ================")
    return {"accuracy": token_acc, "ids": output_ids, "predictions": output_preds, "labels": output_labels}

def eval_crf(model, dataset):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    preds_list, labels_list = [], []
    output_ids, output_preds, output_labels = [], [], []
    for it, batch in enumerate(dataset):
        with torch.no_grad():
            keypoints, batch_labels = batch['keypoints'].to(device), batch['output'].to(device)
            mask = batch['mask'].to(device)
            #### output: list of predictions
            batch_preds = model(keypoints, mask)
            
            output_ids.extend(batch['ids'])
            output_preds.extend(batch_preds)
            for preds in batch_preds:
                preds_list += preds
            for labels, m in zip(batch_labels, mask):
                labels_list += labels[m==True].tolist()
                output_labels.append(labels[m==True].tolist())
    
    preds_list_tensor = torch.tensor(preds_list)
    labels_list_tensor = torch.tensor(labels_list)
    token_acc = (preds_list_tensor == labels_list_tensor).sum()/len(preds_list_tensor)
    print("************")
    print(batch_labels[0].tolist())
    print(batch_preds[0])
    print("************")
    print("TOKEN ACCURACY: {:.1%}".format(token_acc))
    print("================= END OF EVALUATION ================")
    return {"accuracy": token_acc.detach().item(), "ids": output_ids, "predictions": output_preds, "labels": output_labels}