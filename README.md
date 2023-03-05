# SkatingJumpClassifier

## Data Preprocessing
### generate PoseTransformer embeddings for each video
```
cd preprocess
python project_encoder.py --action ${action}
```
### generate Pr-VIPE embeddings for each video
```
cd /home/lin10/projects/poem
conda activate poem
bash alphapose2embs.sh
```
### Data augmentation & Generate .pkl files
```
cd preprocess
bash prepare_dataset.sh
```

## Training && Evaluation
```
bash run_experiment.sh
```
