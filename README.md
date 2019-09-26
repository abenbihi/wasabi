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
   
# Paper plots
