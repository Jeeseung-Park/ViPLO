#!/bin/bash

DIR=upt/checkpoints
FILE=detr-r101-dc5-hicodet.pth
ID=1kkyVeoUGb8rT9b5J5Q3f51OFmm4Z73UD

if [ ! -d $DIR ]; then
   mkdir $DIR
fi 

if [ -f $DIR/$FILE ]; then
  echo "$FILE already exists."
  exit 0
fi

echo "Connecting..."

gdown $ID

mv $FILE $DIR/

echo "Done."