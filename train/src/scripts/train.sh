
# 9k examples 
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO
export TORCH_NCCL_TRACE_BUFFER_SIZE=2097152
export WANDB_MODE=offline
export DEBUG_MODE="false"
export DATA_BASE="/data/lsq/gui-agent-store"
export CKPT_PATH="/mnt/82_store/huggingface_cache/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5"
export SAVE_PATH=${DATA_BASE}/ckpt/UI-AGILE
export LOG_PATH=${SAVE_PATH}"/debug_log.txt"
WANDB_API_KEY=e4b9ff0ef1e6f6bb13f7d8582c81e1d5e76b0f1c \
ACCELERATE_TIMEOUT=60 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12347" \
    ../trl_train/src/open_r1/train.py \
    --resample_if_useless true \
    --thinking_strategy "simple_thinkingv2" \
    --grounding_only false \
    --binary_grounding_reward false \
    --pred_type "coord" \
    --output_dir ${SAVE_PATH}  \
    --model_name_or_path ${CKPT_PATH} \
    --data_file_paths ${DATA_BASE}/training_data/train.json \
    --image_folders ${DATA_BASE}/training_data/train_imgs \
    --dataset_name 9k \
    --deepspeed ../trl_train/local_scripts/zero3.json \
    --max_prompt_length 1024 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --logging_steps 1 \
    --bf16 \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --max_pixels 12845056 \
    --num_train_epochs 2 \
    --save_steps 500 \
    --save_only_model false \
    --num_generations 8 \
    --dataloader_drop_last
# dataloader_drop_last to avoid raise ValueError("Global batch size is not divisible by num_generations. Prompt ordering cannot be determined.")
