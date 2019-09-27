#!/bin/sh

MACHINE=1
if [ "$MACHINE" -eq 0 ]; then
  ws_dir=/home/abenbihi/ws/
elif [ "$MACHINE" -eq 1 ]; then
  ws_dir=/home/gpu_user/assia/ws/
else
  echo "Get your MTF MACHINE macro correct. Bye !"
  exit 1
fi


if [ $# -eq 0 ]; then
  echo "Usage"
  echo "1. trial"
  echo "2. retrieval instance {cmu_park, cmu_urban, lake}"
  echo "3. centroids"
  exit 0
fi

if [ $# -ne 3 ]; then
  echo "Bad number of arguments"
  echo "1. trial"
  echo "2. retrieval instance {cmu_park, cmu_urban, lake}"
  echo "3. centroids"
  exit 1
fi

# source useful path
. ./scripts/export_path.sh


trial="$1"
instance="$2"
centroids="$3"

out_dir=res/delf/"$trial"
if [ -d "$out_dir" ]; then
    while true; do
        read -p ""$out_dir" already exists. Do you want to overwrite it (y/n) ?" yn
        case $yn in
            [Yy]* ) rm -rf "$out_dir"; 
              mkdir -p "$out_dir"/retrieval; 
              mkdir -p "$out_dir"/perf; 
              break;;
            [Nn]* ) exit;;
            * ) * echo "Please answer yes or no.";;
        esac
    done
else
    mkdir -p "$out_dir"/retrieval
    mkdir -p "$out_dir"/perf
fi

# CMU
meta_dir=meta/cmu/surveys/
img_dir="$ws_dir"/datasets/Extended-CMU-Seasons/

python3 -m methods.delf.retrieve \
  --trial "$trial" \
  --dist_pos 5 \
  --top_k 20 \
  --data cmu \
  --instance meta/cmu/cmu_park_debug.txt \
  --centroids "$centroids" \
  --img_dir "$img_dir" \
  --meta_dir "$meta_dir" \
  --config_path meta/delf_config_example.pbtxt

