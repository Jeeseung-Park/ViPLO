#!/bin/bash

DIR=checkpoints
FILE=best_hicodet.pt
ID=1NRP05WDdjLW1MBMRnBU5LgcVmyOXp-wK

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
