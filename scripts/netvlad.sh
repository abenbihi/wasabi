#!/bin/sh


MACHINE=1
if [ "$MACHINE" -eq 0 ]; then
  ws_dir=/home/abenbihi/ws/
elif [ "$MACHINE" -eq 1 ]; then
  ws_dir=/home/gpu_user/assia/ws/
elif [ "$MACHINE" -eq 2 ]; then
  ws_dir=/opt/BenbihiAssia/ws/
else
  echo "Error in train.sh: Get your MTF MACHINE macro correct"
  exit 1
fi

if [ "$#" -eq 0 ]; then
  echo "1. trial"
  exit 0
fi

if [ "$#" -ne 1 ]; then
  echo "Error: bad number of arguments"
  echo "1. trial"
  exit 1
fi

trial="$1"
slice_id="$2"

#if ! [ -d res/"$trial"/log/val/ ]; then
#  mkdir -p res/"$trial"/log/val/
#fi

#log_dir=res/netvlad/"$trial"/
#if [ -d "$log_dir" ]; then
#    while true; do
#        read -p ""$log_dir" already exists. Do you want to overwrite it (y/n) ?" yn
#        case $yn in
#            [Yy]* ) 
#              #rm -rf "$log_dir"; 
#              mkdir -p "$log_dir"/perf;
#              mkdir -p "$log_dir"/retrieval;
#              break;;
#            [Nn]* ) exit;;
#            * ) * echo "Please answer yes or no.";;
#        esac
#    done
#else
#  mkdir -p "$log_dir"/perf;
#  mkdir -p "$log_dir"/retrieval;
#fi


netvlad_dir=third_party/tf_land/netvlad/
pittsburg_weight="$netvlad_dir"meta/weights/netvlad_tf_open/vd16_pitts30k_conv5_3_vlad_preL2_intra_white


###############################################################################
# CMU
meta_dir=meta/cmu/surveys/
img_dir="$ws_dir"/datasets/Extended-CMU-Seasons/

# evaluate netvlad trained on pittsburg on everyone
python3 -m methods.netvlad.retrieve \
  --instance meta/cmu/cmu_park_debug.txt \
  --data cmu \
  --netvlad_dir "$netvlad_dir" \
  --pittsburg_weight "$pittsburg_weight" \
  --trial "$trial" \
  --dist_pos 5 \
  --top_k 20 \
  --mean_fn "$netvlad_dir"meta/mean_std.txt \
  --img_dir "$img_dir" \
  --meta_dir "$meta_dir" \
  --batch_size 1 \
  --resize 1 \
  --h 384 \
  --w 512 \
  --no_finetuning 1 \
  --moving_average_decay 0.9999 \



# evaluate a finetuned netvlad slice by slice
if [ 0 -eq 1 ]; then
    echo "Evaluate a finetuned netvlad on slice "$slice_id""
    #echo "\n\n\n** slice id: "$slice_id""
    for cam_id in 0 1 
    do
      echo "cam_id: "$cam_id""
      for survey_id in 0 1 2 3 4 5 6 7 8 9 10 
      do
        echo "survey_id: "$survey_id""
        if [ "$slice_id" -eq 24 ] || [ "$slice_id" -eq 25 ]; then
          if [ "$cam_id" -eq 1 ] && [ "$survey_id" -eq 8 ]; then
            echo "FAIL: this survey is empty, skip it."
          fi
        fi
        python3 bench.py \
          --trial "$trial" \
          --slice_id "$slice_id" \
          --cam_id "$cam_id" \
          --survey_id "$survey_id" \
          --mean_fn meta/mean_std.txt \
          --split_dir "$split_dir" \
          --img_dir "$img_dir" \
          --batch_size 1 \
          --resize 1 \
          --h 384 \
          --w 512 \
          --no_finetuning 0 \
          --moving_average_decay 0.9999 \
          --top_k 20

      done
    done
fi

###############################################################################
# lake
split_dir="$ws_dir"/datasets/lake/lake/meta/retrieval/
img_dir=/mnt/lake/VBags/

echo "Evaluate netvlad trained on pittsburg on everyone"
if [ 0 -eq 1 ]; then
    for survey_id in 0 1 2 3 4 5 6 7 8 9
    do
        echo "survey_id: "$survey_id""
        python3 bench.py \
            --trial "$trial" \
            --slice_id -1 \
            --cam_id -1 \
            --survey_id "$survey_id" \
            --mean_fn meta/mean_std.txt \
            --split_dir "$split_dir" \
            --img_dir "$img_dir" \
            --batch_size 1 \
            --resize 0 \
            --h 384 \
            --w 512 \
            --no_finetuning 0 \
            --moving_average_decay 0.9999 \
            --top_k 20 \
            --dist_pos 2

        if [ "$?" -ne 0 ]; then
            echo "Error in survey_id "$survey_id""
            exit 1
        fi
    done
fi


