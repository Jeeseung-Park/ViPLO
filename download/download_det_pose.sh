#!/bin/bash

DET_DIR=hicodet/detections

DET_FILE1=test2015_upt.zip 
DET_EXTR1=test2015_upt
ID1=1Is4lTFMi4YO4QaSVe4An5bYRsX2f75V_


if [ ! -d $DET_DIR ]; then
   mkdir $DET_DIR
fi 

if [ -f $DET_DIR/$DET_FILE1 ]; then
  echo "$DET_FILE1 already exists."
  exit 0
fi

echo "$DET_FILE1 Connecting..."

gdown $ID1

unzip -qq $DET_FILE1
rm $DET_FILE1

mv $DET_EXTR1 $DET_DIR/


echo "$DET_EXTR1 Done."



DET_FILE2=train2015_vitpose.zip 
DET_EXTR2=train2015_vitpose
ID2=19PZ3mg7O5TkytnyUU6QjYhn7k_ba-FAD



if [ -f $DET_DIR/$DET_FILE2 ]; then
  echo "$DET_FILE2 already exists."
  exit 0
fi

echo "$DET_FILE2 Connecting..."

gdown $ID2

unzip -qq $DET_FILE2
rm $DET_FILE2

mv $DET_EXTR2 $DET_DIR/


echo "$DET_EXTR2 Done."


DET_FILE3=test2015_vitpose.zip 
DET_EXTR3=test2015_vitpose
ID3=1bNC3cD4uj_mIDNK2Z-eMUOHXP7clbC6o


if [ -f $DET_DIR/$DET_FILE3 ]; then
  echo "$DET_FILE3 already exists."
  exit 0
fi

echo "$DET_FILE3 Connecting..."

gdown $ID3

unzip -qq $DET_FILE3
rm $DET_FILE3

mv $DET_EXTR3 $DET_DIR/


echo "$DET_EXTR3 Done."


DET_FILE4=test2015_upt_vitpose.zip 
DET_EXTR4=test2015_upt_vitpose
ID4=1xTFQqRTASUissFPgSk-gNN0KhWJf8PkB



if [ -f $DET_DIR/$DET_FILE4 ]; then
  echo "$DET_FILE4 already exists."
  exit 0
fi

echo "$DET_FILE4 Connecting..."

gdown $ID4

unzip -qq $DET_FILE4
rm $DET_FILE4

mv $DET_EXTR4 $DET_DIR/


echo "$DET_EXTR4 Done."


DET_FILE5=test2015_gt_vitpose.zip 
DET_EXTR5=test2015_gt_vitpose
ID5=1La4hf5xyTWA_4r1vWiJJOvyqeCBTbe2c


if [ -f $DET_DIR/$DET_FILE5 ]; then
  echo "$DET_FILE5 already exists."
  exit 0
fi

echo "$DET_FILE5 Connecting..."

gdown $ID5

unzip -qq $DET_FILE5
rm $DET_FILE5

mv $DET_EXTR5 $DET_DIR/


echo "$DET_EXTR5 Done."



INST_DIR=hicodet

INST_FILE1=instances_train2015_vitpose.json
ID1=1Rk9LJRQb2_7znp14514mevG68eeHmlu2


if [ ! -d $INST_DIR ]; then
   mkdir $INST_DIR
fi 

if [ -f $INST_DIR/$INST_FILE1 ]; then
  echo "$INST_FILE1 already exists."
  exit 0
fi

echo "$INST_FILE1 Connecting..."

gdown $ID1

mv $INST_FILE1 $INST_DIR/


echo "$INST_FILE1 Done."


INST_FILE2=instances_test2015_vitpose.json
ID2=1zM7RXfZ_k8MUlDMZ0Ql6_XLzhtUZ8IJd



if [ -f $INST_DIR/$INST_FILE2 ]; then
  echo "$INST_FILE2 already exists."
  exit 0
fi

echo "$INST_FILE2 Connecting..."

gdown $ID2

mv $INST_FILE2 $INST_DIR/


echo "$INST_FILE2 Done."

