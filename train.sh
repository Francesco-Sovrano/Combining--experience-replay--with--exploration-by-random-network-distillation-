#!/bin/bash

cd /public/francesco_sovrano
. .env/bin/activate

DIRECTORY=$(hostname)
if [ ! -d "$DIRECTORY" ]; then
  mkdir $DIRECTORY
fi
cd $DIRECTORY

if [ ! -d "log" ]; then
  mkdir log
fi

#export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
#export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
#export LIBRARY_PATH=/usr/local/cuda/lib64${LIBRARY_PATH:+:${LIBRARY_PATH}}

MY_DIR="`dirname \"$0\"`"
python3 $MY_DIR/framework/train.py