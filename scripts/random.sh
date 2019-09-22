#!/bin/sh

if [ "$#" -eq 0 ]; then 
  echo "1. trial"
  exit 0
fi

if [ "$#" -ne 1 ]; then 
  echo "Error: bad number of arguments"
  echo "1. trial"
  exit 0
fi

trial="$1"

res_dir=res/random/"$trial"
#if [ -d "$res_dir" ]; then
#    while true; do
#        read -p ""$res_dir" already exists. Do you want to overwrite it (y/n) ?" yn
#        case $yn in
#            [Yy]* ) rm -rf "$res_dir"; mkdir -p "$res_dir"; break;;
#            [Nn]* ) exit;;
#            * ) * echo "Please answer yes or no.";;
#        esac
#    done
#else
#    mkdir -p "$res_dir"
#fi

# source useful path
. ./scripts/export_path.sh

img_dir="$WS_DIR"datasets/Extended-CMU-Seasons/
meta_dir=meta/cmu/surveys/
seg_dir="$WS_DIR"tf/cross-season-segmentation/res/ext_cmu/

python3 -m methods.random.cmu \
  --trial "$trial" \
  --img_dir "$img_dir" \
  --seg_dir "$seg_dir" \
  --meta_dir "$meta_dir" \
  --dist_pos 5. \
  --repeat 10 \
  --top_k 20
