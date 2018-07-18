#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python3 main.py --dropout 0.0 --h_dropout 0.0 --save_folder "none"&
CUDA_VISIBLE_DEVICES=1 python3 main.py --dropout 0.0 --h_dropout 0.0 --gate --save_folder "attn"&

#CUDA_VISIBLE_DEVICES=1 python3 main.py --dropout 0.2 --h_dropout 0.0 --save_folder "01"&
#CUDA_VISIBLE_DEVICES=2 python3 main.py --dropout 0.0 --h_dropout 0.2 --save_folder "10"&
#CUDA_VISIBLE_DEVICES=3 python3 main.py --dropout 0.2 --h_dropout 0.2 --save_folder "11"&
