for action in "loop" "flip"; do
    test_list_file="/home/lin10/projects/SkatingJumpClassifier/data/${action}/alphapose/test_list.txt"
    while read line; do 
        echo "Generating visualization for $line"
        python3 vis_results.py --action ${action} --video_name ${line}
    done < "$test_list_file"
done