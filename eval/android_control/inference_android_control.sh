

export DATASET="androidcontrol_high"
export MODEL_PATH=/mnt/82_store/lsq/gui-agent-store/ckpt/UI-AGILE  # Change this to your actual model path
export OUTPUT_DIR=output/${DATASET}
CUDA_VISIBLE_DEVICES=5 python inference_android_control.py \
    --model_path ${MODEL_PATH} \
    --prompt_template android_control_detailed \
    --output_path ${OUTPUT_DIR} \
    --data_path ${DATASET}.parquet


export DATASET="androidcontrol_low"
export MODEL_PATH="/path/to/your/model"  # Change this to your actual model path
export OUTPUT_DIR=output/${DATASET}
CUDA_VISIBLE_DEVICES=1 python inference_android_control.py \
    --model_path ${MODEL_PATH} \
    --prompt_template android_control_detailed \
    --output_path ${OUTPUT_DIR} \
    --data_path ${DATASET}.parquet



export DATASET="androidcontrol_high"
export MODEL_PATH=/mnt/82_store/lsq/gui-agent-store/ckpt/UI-AGILE 
export OUTPUT_DIR=output/${DATASET}
CUDA_VISIBLE_DEVICES=5 python inference_android_control.py \
    --model_path ${MODEL_PATH} \
    --prompt_template android_control_detailed \
    --output_path ${OUTPUT_DIR} \
    --data_path /mnt/82_store/lsq/gui-agent-store/android_control_test/androidcontrol_high_test_fixed_scroll_wo_direction.parquet