#!/bin/sh

python3 codebook_tf2np.py \
    --ckpt_dir roxford5k_codebook_65536/k65536_codebook_tfckpt/ \
    --out oxford5k_65536.txt

python3 codebook_tf2np.py \
    --ckpt_dir rparis6k_codebook_65536/k65536_codebook_tfckpt/ \
    --out paris6k_65536.txt

python3 codebook_tf2np.py \
    --ckpt_dir rparis6k_codebook_1024/k1024_codebook_tfckpt/ \
    --out paris6k_1024.txt

python3 codebook_tf2np.py \
    --ckpt_dir roxford5k_codebook_1024/k1024_codebook_tfckpt/ \
    --out oxford5k_1024.txt

