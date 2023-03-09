EXPERIMENT_NAME="raw_skeletons_0309"
CONFIG_NAME="raw_skeletons_0309"

MODEL_NAME="single_jump"
MODEL_PATH="${EXPERIMENT_NAME}/${MODEL_NAME}/"
OUTPUT_LOG_FILE="./experiments/${EXPERIMENT_NAME}/${MODEL_NAME}_evaluation.log"
echo "TRAINING MODEL ON ${MODEL_NAME} dataset"
python trainSeq.py --experiment_name ${EXPERIMENT_NAME} --model_name ${MODEL_NAME} --config_name ${CONFIG_NAME} --dataset ${MODEL_NAME}
for action in 'single_jump' 'multiple_jump'
do
    echo "=================================================================" >> ${OUTPUT_LOG_FILE}
    echo "EVALUATING ${MODEL_PATH} ON ${action} DATASET" >> ${OUTPUT_LOG_FILE}
    echo "=================================================================" >> ${OUTPUT_LOG_FILE}
    python test.py --model_path ${MODEL_PATH} --config_name ${CONFIG_NAME} --dataset ${action}  >> ${OUTPUT_LOG_FILE}
    python eval.py --model_path ${MODEL_PATH} --action ${action} >> ${OUTPUT_LOG_FILE}
done