#!/bin/bash

pip install mmcv-full==1.5.0
pip install opencv-python==4.5.5.64

DIR=ViTPose/models
FILE=vitpose-l.pth 

if [ ! -d $DIR ]; then
   mkdir $DIR
fi 

if [ -f $DIR/$FILE ]; then
  echo "$FILE already exists."
  exit 0
fi

echo "Connecting..."

gdown 1rC9vy8YLvI3Avk_8axz1AP9KgHZ-VMTj

mv $FILE $DIR/

echo "Done."