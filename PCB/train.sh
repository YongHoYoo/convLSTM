#!/bin/bash

CUDA_VISIBLE_DEVICES=2 python3 main.py --lr 1e-4 --dropout 0.1 --h_dropout 0.1 --train_folder "train_none" --valid_folder "valid_none"&
CUDA_VISIBLE_DEVICES=3 python3 main.py --lr 1e-4 --dropout 0.1 --h_dropout 0.1 --gate --train_folder "train_attn" --valid_folder "valid_attn"&

#CUDA_VISIBLE_DEVICES=1 python3 main.py --dropout 0.2 --h_dropout 0.0 --save_folder "01"&
#CUDA_VISIBLE_DEVICES=2 python3 main.py --dropout 0.0 --h_dropout 0.2 --save_folder "10"&
#CUDA_VISIBLE_DEVICES=3 python3 main.py --dropout 0.2 --h_dropout 0.2 --save_folder "11"&
