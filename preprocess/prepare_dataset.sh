for action in 'Axel' 'Loop' 'Lutz' 'Flip' 'all_jump'
do
    # echo "Generating skeleton embeddings for ${action} data..."
    # python project_encoder.py --action ${action}

    echo "Augmenting ${action} data..."
    python augmentation.py --action ${action}
    
    echo "Generating .pkl for ${action} data..."
    python preprocess.py --action ${action}
done