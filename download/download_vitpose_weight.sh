#!/bin/bash

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

wget -O vitpose-l.pth https://public.sn.files.1drv.com/y4md7QiA4K71OxfxELiJnL35uho_TOEnM-gF2U4anmexXoubekW1Yd57rV-eXDY9U0S1kD08YQtCdzQ-moZd-k7trXECddAU6ec46qAHKEz42VKf8ieZpA8sv09NVaRhJ1hLjh6rUfNFgiAf9QYhyzn5NJaYENr0xCwrmue7Ij_ZrXMJ-yHvIzFy7nuZZmsO3RsibbdWlaM3BdDZKowoeTTvJNOniUwtHIJWZeQaEX_hB0


mv $FILE $DIR/

echo "Done."