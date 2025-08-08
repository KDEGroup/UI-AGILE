# vllm local
export DATA_BASE="/data/lsq/gui-agent-store"
export VLLM_MODEL_NAME="UI-AGILE"
export MODEL_PATH=/mnt/82_store/lsq/gui-agent-store/ckpt/UI-AGILE
export PROMPT_TEMPLATE="nothink_point"
export DATASET="ScreenSpot-Pro"
CUDA_VISIBLE_DEVICES=0 python eval_grounding_vllm_no_ray.py \
    --model_path ${MODEL_PATH} \
    --data_path ${DATA_BASE}/grounding_bench/data/${DATASET}.parquet \
    --num_gpus 1 \
    --prompt_template ${PROMPT_TEMPLATE} \
    --log_path results/ckpt/${DATASET}/${VLLM_MODEL_NAME}_${PROMPT_TEMPLATE}.json

# ScreenSpot_v2
export DATA_BASE="/data/lsq/gui-agent-store"
export VLLM_MODEL_NAME="UI-AGILE-3B"
export MODEL_PATH=/mnt/82_store/lsq/gui-agent-store/ckpt/UI-AGILE-3B
export PROMPT_TEMPLATE="nothink_point"
export DATASET="ScreenSpot_v2"
CUDA_VISIBLE_DEVICES=1 python eval_grounding_vllm_no_ray.py \
    --model_path ${MODEL_PATH} \
    --data_path ${DATA_BASE}/grounding_bench/data/${DATASET}.parquet \
    --num_gpus 1 \
    --prompt_template ${PROMPT_TEMPLATE} \
    --log_path results/ckpt/${DATASET}/${VLLM_MODEL_NAME}_${PROMPT_TEMPLATE}.json



# baselines


# uitars1.5-7B
export DATA_BASE="/data/lsq/gui-agent-store"
export VLLM_MODEL_NAME="UI-TARS-1.5-7B"
export MODEL_PATH="/mnt/82_store/huggingface_cache/hub/models--ByteDance-Seed--UI-TARS-1.5-7B/snapshots/683d002dd99d8f95104d31e70391a39348857f4e"
export PROMPT_TEMPLATE="uitars"
export DATASET="ScreenSpot-Pro-cropped"
CUDA_VISIBLE_DEVICES=6 python eval_grounding_vllm_no_ray.py \
    --model_path ${MODEL_PATH} \
    --data_path ${DATA_BASE}/grounding_bench/data/${DATASET}.parquet \
    --num_gpus 1 \
    --prompt_template ${PROMPT_TEMPLATE} \
    --log_path results/ckpt/${DATASET}/${VLLM_MODEL_NAME}_${PROMPT_TEMPLATE}.json


# ui-r1-e
export DATA_BASE="/data/lsq/gui-agent-store"
export VLLM_MODEL_NAME="UI-R1-E"
export MODEL_PATH="/mnt/82_store/huggingface_cache/hub/models--LZXzju--Qwen2.5-VL-3B-UI-R1-E/snapshots/91c3e5f213ab3f42931e6398174f470c8500167f"
export PROMPT_TEMPLATE="nothink_point"
export DATASET="ScreenSpot_v2"
CUDA_VISIBLE_DEVICES=2 python eval_grounding_vllm_no_ray.py \
    --model_path ${MODEL_PATH} \
    --model_type qwen2.5vl \
    --data_path ${DATA_BASE}/grounding_bench/data/${DATASET}.parquet \
    --num_gpus 1 \
    --prompt_template ${PROMPT_TEMPLATE} \
    --log_path results/${DATASET}/ckpt/${VLLM_MODEL_NAME}_${PROMPT_TEMPLATE}.json



# qwen
export DATA_BASE="/data/lsq/gui-agent-store"
export VLLM_MODEL_NAME="Qwen2.5-VL-7B"
export MODEL_PATH="/mnt/82_store/huggingface_cache/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5"
export PROMPT_TEMPLATE="nothink_point"
export DATASET="ScreenSpot_v2"
CUDA_VISIBLE_DEVICES=1 python eval_grounding_vllm_no_ray.py \
    --model_path ${MODEL_PATH} \
    --model_type qwen2.5vl \
    --data_path ${DATA_BASE}/grounding_bench/data/${DATASET}.parquet \
    --num_gpus 1 \
    --prompt_template ${PROMPT_TEMPLATE} \
    --log_path results/${DATASET}/ckpt/${VLLM_MODEL_NAME}_${PROMPT_TEMPLATE}.json


# GUI-R1
export TOKENIZERS_PARALLELISM=false
export DATA_BASE="/data/lsq/gui-agent-store"
export VLLM_MODEL_NAME="GUI-R1-7B"
export MODEL_PATH="/mnt/82_store/huggingface_cache/hub/models--ritzzai--GUI-R1/snapshots/e74baccc4cfa77074e2d53e99a8244ab9fc2ca10/GUI-R1-7B"
export PROMPT_TEMPLATE="gui_r1"
export DATASET="ScreenSpot_v2"
CUDA_VISIBLE_DEVICES=7 python eval_grounding_vllm_no_ray.py \
    --model_path ${MODEL_PATH} \
    --model_type qwen2.5vl \
    --data_path ${DATA_BASE}/grounding_bench/data/${DATASET}.parquet \
    --num_gpus 1 \
    --prompt_template ${PROMPT_TEMPLATE} \
    --log_path results/${DATASET}/ckpt/${VLLM_MODEL_NAME}_${PROMPT_TEMPLATE}.json


# uground-v1-7b
export DATA_BASE="/data/lsq/gui-agent-store"
export VLLM_MODEL_NAME="UGround-V1-7B"
export MODEL_PATH="/mnt/82_store/huggingface_cache/hub/models--osunlp--UGround-V1-7B/snapshots/db91e9617650940e27f84e04957d5be0ff1f91a4"
export PROMPT_TEMPLATE="uground"
export DATASET="ScreenSpot-Pro-cropped"
CUDA_VISIBLE_DEVICES=7 python eval_grounding_vllm_no_ray.py \
    --model_path ${MODEL_PATH} \
    --model_type qwen2vl \
    --data_path ${DATA_BASE}/grounding_bench/data/${DATASET}.parquet \
    --num_gpus 1 \
    --prompt_template ${PROMPT_TEMPLATE} \
    --log_path results/${DATASET}/ckpt/${VLLM_MODEL_NAME}_${PROMPT_TEMPLATE}.json



# os-atlas-base-7b
export DATA_BASE="/data/lsq/gui-agent-store"
export VLLM_MODEL_NAME="OS-Atlas-Base-7B"
export MODEL_PATH="/mnt/82_store/huggingface_cache/hub/models--OS-Copilot--OS-Atlas-Base-7B/snapshots/7ed87a4f5904cb3cd0c7ce673ea62656256e7b07"
export PROMPT_TEMPLATE="os_atlas_base_7b"
export DATASET="ScreenSpot_v2"
CUDA_VISIBLE_DEVICES=2 python eval_grounding_vllm_no_ray.py \
    --model_path ${MODEL_PATH} \
    --model_type qwen2vl \
    --data_path ${DATA_BASE}/grounding_bench/data/${DATASET}.parquet \
    --num_gpus 1 \
    --prompt_template ${PROMPT_TEMPLATE} \
    --log_path results/${DATASET}/ckpt/${VLLM_MODEL_NAME}_${PROMPT_TEMPLATE}.json


# aguvis
export DATA_BASE="/data/lsq/gui-agent-store"
export VLLM_MODEL_NAME="Aguvis-7B-720P"
export MODEL_PATH="/mnt/82_store/huggingface_cache/hub/models--xlangai--Aguvis-7B-720P/snapshots/6dd54127b5b84b9ee89172a5065ab6be576f0db9/"
export PROMPT_TEMPLATE="aguvis_7b_720p"
export DATASET="ScreenSpot_v2"
CUDA_VISIBLE_DEVICES=3 python eval_grounding_vllm_no_ray.py \
    --model_path ${MODEL_PATH} \
    --model_type aguvis_7b_720p \
    --data_path ${DATA_BASE}/grounding_bench/data/${DATASET}.parquet \
    --num_gpus 1 \
    --prompt_template ${PROMPT_TEMPLATE} \
    --log_path results/${DATASET}/ckpt/${VLLM_MODEL_NAME}_${PROMPT_TEMPLATE}.json

# showui range is [0,1] same as aguvis
export DATA_BASE="/data/lsq/gui-agent-store"
export VLLM_MODEL_NAME="ShowUI-2B"
export MODEL_PATH="/mnt/82_store/huggingface_cache/hub/models--showlab--ShowUI-2B/snapshots/cabec4fcc48d15ffd3efe0b33ea9bc7d41509d60"
export PROMPT_TEMPLATE="showui"
export DATASET="ScreenSpot_v2"
CUDA_VISIBLE_DEVICES=6 python eval_grounding_vllm_no_ray.py \
    --model_path ${MODEL_PATH} \
    --model_type aguvis_7b_720p \
    --data_path ${DATA_BASE}/grounding_bench/data/${DATASET}.parquet \
    --num_gpus 1 \
    --prompt_template ${PROMPT_TEMPLATE} \
    --log_path results/${DATASET}/ckpt/${VLLM_MODEL_NAME}_${PROMPT_TEMPLATE}.json

# showui debug
export DATA_BASE="/data/lsq/gui-agent-store"
export VLLM_MODEL_NAME="ShowUI-2B"
export MODEL_PATH="/mnt/82_store/huggingface_cache/hub/models--showlab--ShowUI-2B/snapshots/cabec4fcc48d15ffd3efe0b33ea9bc7d41509d60"
export PROMPT_TEMPLATE="showui"
export DATASET="ScreenSpot_v2"
CUDA_VISIBLE_DEVICES=6 python eval_showui.py \
    --model_path ${MODEL_PATH} \
    --data_path ${DATA_BASE}/grounding_bench/data/${DATASET}.parquet 


# vllm local   crop select
export TOKENIZERS_PARALLELISM=false
export DATA_BASE="/data/lsq/gui-agent-store"
export VLLM_MODEL_NAME="UI-AGILE"
export MODEL_PATH=/mnt/82_store/lsq/gui-agent-store/ckpt/UI-AGILE
export PROMPT_TEMPLATE="nothink_point"
export SCORE_METHOD="lmm_logits"
export DATASET="ScreenSpot-Pro"
CUDA_VISIBLE_DEVICES=2,3 python eval_grounding_vllm_no_ray.py \
    --model_path ${MODEL_PATH} \
    --data_path ${DATA_BASE}/grounding_bench/data/${DATASET}.parquet \
    --prompt_template ${PROMPT_TEMPLATE} \
    --crop_select \
    --step_ratio 0.5 \
    --tile_ratio 0.6 \
    --score_method ${SCORE_METHOD} \
    --score_model_path /mnt/82_store/huggingface_cache/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5 \
    --num_gpus_for_generate 1 \
    --score_eval_image_save_dir /data/crop_select/output/score_eval/score_eval_${VLLM_MODEL_NAME}_${SCORE_METHOD}_tile0_6/images \
    --score_eval_out_path /data/crop_select/output/score_eval/score_eval_${VLLM_MODEL_NAME}_${SCORE_METHOD}_tile0_6/score_eval.json \
    --log_path results/${DATASET}/ckpt/${VLLM_MODEL_NAME}_${PROMPT_TEMPLATE}_${SCORE_METHOD}.json


# ui-r1-e
export TOKENIZERS_PARALLELISM=false
export DATA_BASE="/data/lsq/gui-agent-store"
export VLLM_MODEL_NAME="UI-R1-E"
export MODEL_PATH="/mnt/82_store/huggingface_cache/hub/models--LZXzju--Qwen2.5-VL-3B-UI-R1-E/snapshots/91c3e5f213ab3f42931e6398174f470c8500167f"
export PROMPT_TEMPLATE="nothink_point"
export SCORE_METHOD="lmm_logits"
export DATASET="ScreenSpot-Pro"
CUDA_VISIBLE_DEVICES=4 python eval_grounding_vllm_no_ray.py \
    --model_path ${MODEL_PATH} \
    --data_path ${DATA_BASE}/grounding_bench/data/${DATASET}.parquet \
    --prompt_template ${PROMPT_TEMPLATE} \
    --crop_select \
    --step_ratio 0.5 \
    --tile_ratio 0.6 \
    --score_method ${SCORE_METHOD} \
    --score_model_path /mnt/82_store/huggingface_cache/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5/ \
    --num_gpus_for_generate 1 \
    --score_eval_image_save_dir /data/crop_select/output/score_eval/score_eval_${VLLM_MODEL_NAME}_${SCORE_METHOD}_tile0_6/images \
    --score_eval_out_path /data/crop_select/output/score_eval/score_eval_${VLLM_MODEL_NAME}_${SCORE_METHOD}_tile0_6/score_eval.json \
    --log_path results/${DATASET}/ckpt/${VLLM_MODEL_NAME}_${PROMPT_TEMPLATE}_${SCORE_METHOD}.json


# gui-r1
export TOKENIZERS_PARALLELISM=false
export DATA_BASE="/data/lsq/gui-agent-store"
export VLLM_MODEL_NAME="GUI-R1-7B"
export MODEL_PATH="/mnt/82_store/huggingface_cache/hub/models--ritzzai--GUI-R1/snapshots/e74baccc4cfa77074e2d53e99a8244ab9fc2ca10/GUI-R1-7B"
export PROMPT_TEMPLATE="gui_r1"
export SCORE_METHOD="lmm_logits"
export DATASET="ScreenSpot-Pro"
CUDA_VISIBLE_DEVICES=2,3 python eval_grounding_vllm_no_ray.py \
    --model_path ${MODEL_PATH} \
    --data_path ${DATA_BASE}/grounding_bench/data/${DATASET}.parquet \
    --prompt_template ${PROMPT_TEMPLATE} \
    --crop_select \
    --step_ratio 0.5 \
    --tile_ratio 0.6 \
    --score_method ${SCORE_METHOD} \
    --score_model_path /mnt/82_store/huggingface_cache/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5/ \
    --num_gpus_for_generate 1 \
    --score_eval_image_save_dir /data/crop_select/output/score_eval/score_eval_${VLLM_MODEL_NAME}_${SCORE_METHOD}_tile0_6/images \
    --score_eval_out_path /data/crop_select/output/score_eval/score_eval_${VLLM_MODEL_NAME}_${SCORE_METHOD}_tile0_6/score_eval.json \
    --log_path results/${DATASET}/ckpt/${VLLM_MODEL_NAME}_${PROMPT_TEMPLATE}_${SCORE_METHOD}.json

# qwen
export TOKENIZERS_PARALLELISM=false
export DATA_BASE="/data/lsq/gui-agent-store"
export VLLM_MODEL_NAME="Qwen2.5-VL-7B-Instruct"
export MODEL_PATH="/mnt/82_store/huggingface_cache/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5"
export PROMPT_TEMPLATE="nothink_point"
export SCORE_METHOD="lmm_logits"
export DATASET="ScreenSpot-Pro"
CUDA_VISIBLE_DEVICES=4,5 python eval_grounding_vllm_no_ray.py \
    --model_path ${MODEL_PATH} \
    --data_path ${DATA_BASE}/grounding_bench/data/${DATASET}.parquet \
    --prompt_template ${PROMPT_TEMPLATE} \
    --crop_select \
    --step_ratio 0.5 \
    --tile_ratio 0.6 \
    --score_method ${SCORE_METHOD} \
    --score_model_path /mnt/82_store/huggingface_cache/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5/ \
    --num_gpus_for_generate 1 \
    --score_eval_image_save_dir /data/crop_select/output/score_eval/score_eval_${VLLM_MODEL_NAME}_${SCORE_METHOD}_tile0_6/images \
    --score_eval_out_path /data/crop_select/output/score_eval/score_eval_${VLLM_MODEL_NAME}_${SCORE_METHOD}_tile0_6/score_eval.json \
    --log_path results/${DATASET}/ckpt/${VLLM_MODEL_NAME}_${PROMPT_TEMPLATE}_${SCORE_METHOD}.json


# qwen2vl based

# uground-v1-7b
export TOKENIZERS_PARALLELISM=false
export DATA_BASE="/data/lsq/gui-agent-store"
export VLLM_MODEL_NAME="UGround-V1-7B"
export MODEL_PATH="/mnt/82_store/huggingface_cache/hub/models--osunlp--UGround-V1-7B/snapshots/db91e9617650940e27f84e04957d5be0ff1f91a4"
export PROMPT_TEMPLATE="uground"
export SCORE_METHOD="lmm_logits"
export DATASET="ScreenSpot-Pro"
CUDA_VISIBLE_DEVICES=2,3 python eval_grounding_vllm_no_ray.py \
    --model_path ${MODEL_PATH} \
    --data_path ${DATA_BASE}/grounding_bench/data/${DATASET}.parquet \
    --prompt_template ${PROMPT_TEMPLATE} \
    --model_type qwen2vl \
    --crop_select \
    --step_ratio 0.5 \
    --tile_ratio 0.6 \
    --score_method ${SCORE_METHOD} \
    --score_model_path /mnt/82_store/huggingface_cache/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5/ \
    --num_gpus_for_generate 1 \
    --score_eval_image_save_dir /data/crop_select/output/score_eval/score_eval_${VLLM_MODEL_NAME}_${SCORE_METHOD}_tile0_6/images \
    --score_eval_out_path /data/crop_select/output/score_eval/score_eval_${VLLM_MODEL_NAME}_${SCORE_METHOD}_tile0_6/score_eval.json \
    --log_path results/${DATASET}/ckpt/${VLLM_MODEL_NAME}_${PROMPT_TEMPLATE}_${SCORE_METHOD}.json


# os-atlas-base-7b
export TOKENIZERS_PARALLELISM=false
export DATA_BASE="/data/lsq/gui-agent-store"
export VLLM_MODEL_NAME="OS-Atlas-Base-7B"
export MODEL_PATH="/mnt/82_store/huggingface_cache/hub/models--OS-Copilot--OS-Atlas-Base-7B/snapshots/7ed87a4f5904cb3cd0c7ce673ea62656256e7b07"
export PROMPT_TEMPLATE="os_atlas_base_7b"
export SCORE_METHOD="lmm_logits"
export DATASET="ScreenSpot-Pro"
CUDA_VISIBLE_DEVICES=0 python eval_grounding_vllm_no_ray.py \
    --model_path ${MODEL_PATH} \
    --data_path ${DATA_BASE}/grounding_bench/data/${DATASET}.parquet \
    --prompt_template ${PROMPT_TEMPLATE} \
    --model_type qwen2vl \
    --crop_select \
    --step_ratio 0.5 \
    --tile_ratio 0.6 \
    --score_method ${SCORE_METHOD} \
    --score_model_path /mnt/82_store/huggingface_cache/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5/ \
    --num_gpus_for_generate 1 \
    --score_eval_image_save_dir /data/crop_select/output/score_eval/score_eval_${VLLM_MODEL_NAME}_${SCORE_METHOD}_tile0_6/images \
    --score_eval_out_path /data/crop_select/output/score_eval/score_eval_${VLLM_MODEL_NAME}_${SCORE_METHOD}_tile0_6/score_eval.json \
    --log_path results/${DATASET}/ckpt/${VLLM_MODEL_NAME}_${PROMPT_TEMPLATE}_${SCORE_METHOD}.json


# aguvis
export TOKENIZERS_PARALLELISM=false
export DATA_BASE="/data/lsq/gui-agent-store"
export VLLM_MODEL_NAME="Aguvis-7B-720P"
export MODEL_PATH="/mnt/82_store/huggingface_cache/hub/models--xlangai--Aguvis-7B-720P/snapshots/6dd54127b5b84b9ee89172a5065ab6be576f0db9/"
export PROMPT_TEMPLATE="aguvis_7b_720p"
export SCORE_METHOD="lmm_logits"
export DATASET="ScreenSpot-Pro"
CUDA_VISIBLE_DEVICES=0 python eval_grounding_vllm_no_ray.py \
    --model_path ${MODEL_PATH} \
    --data_path ${DATA_BASE}/grounding_bench/data/${DATASET}.parquet \
    --prompt_template ${PROMPT_TEMPLATE} \
    --model_type aguvis_7b_720p \
    --crop_select \
    --step_ratio 0.5 \
    --tile_ratio 0.6 \
    --score_method ${SCORE_METHOD} \
    --score_model_path /mnt/82_store/huggingface_cache/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5/ \
    --num_gpus_for_generate 1 \
    --score_eval_image_save_dir /data/crop_select/output/score_eval/score_eval_${VLLM_MODEL_NAME}_${SCORE_METHOD}_tile0_6_debug/images \
    --score_eval_out_path /data/crop_select/output/score_eval/score_eval_${VLLM_MODEL_NAME}_${SCORE_METHOD}_tile0_6_debug/score_eval.json \
    --log_path results/${DATASET}/ckpt/${VLLM_MODEL_NAME}_${PROMPT_TEMPLATE}_${SCORE_METHOD}.json
