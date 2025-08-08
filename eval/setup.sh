


conda create -n xxx python==3.12.9
conda activate xxx
# vllm 0.9.1 or later is quite fast, but requires Python 3.12 or later
python -m pip install vllm loguru qwen-vl-utils pillow datasets python-doctr ultralytics
python -m pip install -r requirements.txt