#!/bin/sh
MACHINE=0
if [ "$MACHINE" -eq 0 ]; then
  export WS_DIR=/home/abenbihi/ws/
elif [ "$MACHINE" -eq 1 ]; then
  export WS_DIR=/home/gpu_user/assia/ws/
else
  echo "Get your MTF MACHINE macro correct. Bye !"
  exit 1
fi

echo "I am exporting this variable to you environment: "$WS_DIR""
