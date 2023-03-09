import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn import manifold

def viz_tSNE(emb,
            output_path):
  
    X = np.empty((0, 544))
    frame_idx = []
    for j in range(len(emb)):
        X = np.append(X, np.array([emb[j]]), axis=0)
        frame_idx.append(j)
    frame_idx = np.array(frame_idx)

    #t-SNE
    X_tsne = manifold.TSNE(n_components=2, init='random', random_state=5, verbose=1).fit_transform(X)

    #Data Visualization
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  #Normalize
    plt.figure(figsize=(8, 8))
    for i in range(X_norm.shape[0]):
        plt.text(X_norm[i][0], X_norm[i][1], str(frame_idx[i]), 
                fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.savefig(output_path)

def get_posetransformer_embeddings(action, original_video):
    """
        T: # of frames of original_video
        return np.array (T, 544)
    """
    embedding_file = os.path.join(f'/home/lin10/projects/SkatingJumpClassifier/20220801/{action}', original_video, "skeleton_embedding.pkl")
    with open(embedding_file, 'rb') as f:
        embeddings = pickle.load(f)
    return embeddings

if __name__ == "__main__":
    action = "Loop"
    video = "Loop_187"
    output_path = f"./vis_{video}_emb.jpg"
    embedding = get_posetransformer_embeddings(action, video)
    viz_tSNE(embedding, output_path)
    