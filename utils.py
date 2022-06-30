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

            