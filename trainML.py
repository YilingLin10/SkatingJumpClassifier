from sklearn.neighbors import KNeighborsClassifier
from data.dataloader import *
from sklearn.metrics import accuracy_score, confusion_matrix

train_ice_skating_dataset = IceSkatingDataset(csv_file='/home/calvin/github/skating_classifier/data/iceskatingjump.csv',
                                    root_dir='/home/calvin/github/skating_classifier/data/train_balance/')

test_ice_skating_dataset = IceSkatingDataset(csv_file='/home/calvin/github/skating_classifier/data/iceskatingjump.csv',
                                    root_dir='/home/calvin/github/skating_classifier/data/test/')

model = KNeighborsClassifier(n_neighbors=2)

trainloader = DataLoader(train_ice_skating_dataset, batch_size=train_ice_skating_dataset.__len__(), shuffle=True, num_workers=4)

testloader = DataLoader(test_ice_skating_dataset, batch_size=test_ice_skating_dataset.__len__(), shuffle=True, num_workers=4)

for batch_idx, sample in enumerate(trainloader):
    model.fit(sample['keypoints'], sample['output'].reshape(-1,1))

for batch_idx, sample in enumerate(testloader):
    y_pred= model.predict(sample['keypoints'])
    acc = accuracy_score(sample['output'], y_pred)
    cm = confusion_matrix(sample['output'], y_pred, labels=[0, 1])

    tn, fp, fn, tp = cm.ravel()
    print("Accuracy: {}%\t TN: {}\t, FP: {}\t, FN: {}\t, TP: {}\t".format(acc, tn,fp,fn,tp))