# wasabi
Image-Based Place Recognition on Bucolic Environment Across Seasons From Semantic Edge Description

Experiments code (coming soon, soon being in the order of days)

# Setup

    git clone --recursive 
    pip3 install -r requirements.txt

# Datasets
## CMU-Seasons
Script to download slice [6,8] and [22,25]:

    ./get_cmu.sh

## Symphony
Script to Symphony images:

    ./get_symphony.sh


# WASABI Results

    ./scripts/wasabi.sh <trial>
    python -m plots.tab2latex --method wasabi --trial <trial> --data cmu_park
    pdflatex res/wasabi/1/1.tex

# NetVLAD Results
    ./scripts/netvlad.sh <trial>
    python -m plots.tab2latex --method netvlad --trial <trial> --data cmu_park


# VLAD/BOW Results
With original codebook trained on Flickr100

    ./scripts/vlad_bow.sh <trial> <agg_mode> flickr100k

      # example (See scripts/vlad_bow.sh to run it)
      python3 -m methods.vlad.retrieve \
        --trial 0 \
        --dist_pos 5 \
        --top_k 20 \
        --lf_mode sift \
        --max_num_feat 1000 \
        --agg_mode bow \
        --centroids flickr100k \
        --vlad_norm ssr \
        --data cmu \
        --slice_id 22 \
        --cam_id 0 \
        --survey_id 0 \
        --img_dir "$img_dir" \
        --meta_dir "$meta_dir" \
        --n_words 64 

## Generate codebook on CMU-Seasons/Lake 
With codebook finetuned on cmu park

With codebook finetuned on lake


# DELF results

## Convert DELF codebooks from tf to np format

    cd meta/words/delf
    ./get_words.sh
    ./convert_words.sh

    cd "$WASABI_DIR"
    python3 methods/delf/codebook_tf2np.py \
        --ckpt_dir meta/words/delf/roxford5k_codebook_65536/k65536_codebook_tfckpt/ \
        --out meta/words/delf/oxford5k_65536.txt

# Paper plots
