EXPERIMENT_NAME="posetransformer_0308"
CONFIG_NAME="posetransformer_0308"

MODEL_NAME="all_jump"
MODEL_PATH="${EXPERIMENT_NAME}/${MODEL_NAME}/"
OUTPUT_LOG_FILE="./experiments/${EXPERIMENT_NAME}/${MODEL_NAME}_evaluation.log"
echo "TRAINING MODEL ON all_jump dataset"
python trainSeq.py --experiment_name ${EXPERIMENT_NAME} --model_name ${MODEL_NAME} --config_name ${CONFIG_NAME} --dataset all_jump
for action in 'Axel' 'Loop' 'Flip' 'Lutz' 'all_jump'
do
    echo "=================================================================" >> ${OUTPUT_LOG_FILE}
    echo "EVALUATING ${MODEL_PATH} ON ${action} DATASET" >> ${OUTPUT_LOG_FILE}
    echo "=================================================================" >> ${OUTPUT_LOG_FILE}
    python test.py --model_path ${MODEL_PATH} --config_name ${CONFIG_NAME} --dataset ${action}  >> ${OUTPUT_LOG_FILE}
    python eval.py --model_path ${MODEL_PATH} --action ${action} >> ${OUTPUT_LOG_FILE}
done

for train_dataset in 'Axel' 'Loop' 'Flip' 'Lutz'
do
MODEL_NAME=${train_dataset}
MODEL_PATH="${EXPERIMENT_NAME}/${MODEL_NAME}/"
OUTPUT_LOG_FILE="./experiments/${EXPERIMENT_NAME}/${MODEL_NAME}_evaluation.log"
echo "TRAINING MODEL ON ${train_dataset} dataset"
python trainSeq.py --experiment_name ${EXPERIMENT_NAME} --model_name ${MODEL_NAME} --config_name ${CONFIG_NAME} --dataset ${train_dataset} --num_epochs 300
    for action in 'Axel' 'Loop' 'Flip' 'Lutz' 'all_jump'
    do
        echo "=================================================================" >> ${OUTPUT_LOG_FILE}
        echo "EVALUATING ${MODEL_PATH} ON ${action} DATASET" >> ${OUTPUT_LOG_FILE}
        echo "=================================================================" >> ${OUTPUT_LOG_FILE}
        python test.py --model_path ${MODEL_PATH} --config_name ${CONFIG_NAME} --dataset ${action}  >> ${OUTPUT_LOG_FILE}
        python eval.py --model_path ${MODEL_PATH} --action ${action} >> ${OUTPUT_LOG_FILE}
    done
done