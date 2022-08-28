for mode in "train" "test"
do
    for action in "axel" "flip" "loop" "lutz" "salchow" "toe"
    do 
        echo "Copying ${action} ${mode} data..."
        cp -a /home/lin10/projects/SkatingJumpClassifier/data/${action}/${mode}/. /home/lin10/projects/SkatingJumpClassifier/data/all_jump/${mode}
    done
done