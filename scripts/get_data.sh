#!/bin/bash

# 1) Go to repo root
cd /Users/patrickbarker/ghq/github.com/Omni-Avatar/OmniAvatar

# 2) Make a folder for the raw .mov files
mkdir -p raw_mov

# 3) Download the Google Drive files (uses gdown)
pip install -q gdown
gdown --id 1lzfh1mLStdVACT3vuOyh4-lxww7vGQjv -O raw_mov/clip_001.mov
gdown --id 1OfDM88fDGCC2-CVDWAYY5kZE0Hpm-pGj -O raw_mov/clip_002.mov
gdown --id 1FtFUwwlQxA5w5jnneg-TWzIchFuDptSr -O raw_mov/clip_003.mov
gdown --id 1vOoaytNCS-kEF9W6om0HZsugKkNWxHtK -O raw_mov/clip_004.mov
gdown --id 1nuiUFpJyXWg20oKhyx2-TN87XRB6a1L0 -O raw_mov/clip_005.mov
gdown --id 1lSKuIV1iL4AHxm4SEMhus-xEZRJRjtO4 -O raw_mov/clip_006.mov
gdown --id 13bhY1E8mxENpjGAuzsDy6VW1_SvwKqXV -O raw_mov/clip_007.mov
gdown --id 1PdHKbMJl4hdFHOtJ3_Ma07Jq0duD8Oai -O raw_mov/clip_008.mov
gdown --id 1LUOKosNvqpZA9t4if9WaCjPxIeOCmlQd -O raw_mov/clip_009.mov
gdown --id 1CPZSXMNXraUnBX9cco_icQcrwSj-Ycub -O raw_mov/clip_010.mov

# 4) Prepare the dataset (defaults to 5s clips; trims the end if longer)
python scripts/prepare_mov_dataset.py raw_mov data

# 5) Set the training manifest path in config
# (configs/train_lora.yaml already defaults to data/train_manifest.txt; verify if needed)

# 6) (Optional) Inspect the first lines
head -n 5 data/train_manifest.txt

# 7) Launch Modal H100 training
modal run scripts/modal_train.py::train_h100 --config-path configs/train_lora.yaml --nproc 1