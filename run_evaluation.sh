EXPERIMENT_NAME="raw_skeletons_0306"
CONFIG_NAME="raw_skeletons_0306"
for train_dataset in 'Axel' 'Loop' 'Flip' 'Lutz' 'all_jump'
do
MODEL_NAME=${train_dataset}
MODEL_PATH="${EXPERIMENT_NAME}/${MODEL_NAME}/"
OUTPUT_LOG_FILE="./experiments/${EXPERIMENT_NAME}/${MODEL_NAME}_evaluation.log"
    for action in 'Axel' 'Loop' 'Flip' 'Lutz' 'all_jump'
    do 
        echo "=================================================================" >> ${OUTPUT_LOG_FILE}
        echo "EVALUATING ${MODEL_PATH} ON ${action} DATASET" >> ${OUTPUT_LOG_FILE}
        echo "=================================================================" >> ${OUTPUT_LOG_FILE}
        python test.py --model_path ${MODEL_PATH} --config_name ${CONFIG_NAME} --dataset ${action}  >> ${OUTPUT_LOG_FILE}
        python eval.py --model_path ${MODEL_PATH} --action ${action} >> ${OUTPUT_LOG_FILE}
    done
done