for type in 'Axel' 'Axel_combo' 'Flip' 'Flip_Combo' 'Loop' \
'Loop_combo' 'Lutz' 'Salchow' 'Salchow_combo' 'Toe-Loop'
do
    for file in 20220801_Jump_重新命名/${type}/*.MOV;
    do
        name=${file##*/}
        base=${name%.MOV}
        echo "Generating ${type} ${base} data..."
        # Image to tfrecord files
        python3 split_frame.py --type ${type} --filename ${base}
    done
done