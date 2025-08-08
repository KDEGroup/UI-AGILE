cd src/trl_train
python -m pip install -e ".[dev]"

# Addtional modules
python -m pip install wandb==0.18.3
python -m pip install tensorboardx loguru wandb
python -m pip install qwen_vl_utils torchvision
python -m pip install torch==2.6.0
python -m pip install vllm==0.8.3
python -m pip install transformers==4.52.4
python -m pip install flash-attn --no-build-isolation


# python -m  pip install vllm==0.8.3


# python -m pip install transformers==4.51.3
