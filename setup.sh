#!/bin/bash

MY_DIR="`dirname \"$0\"`"

cd /public
if [ ! -d "francesco_sovrano" ]; then
	mkdir francesco_sovrano
	chmod 700 francesco_sovrano
fi
cd ./francesco_sovrano

if [ ! -d ".env" ]; then
	virtualenv -p python3 .env
fi
. .env/bin/activate

# upgrade pip
# install tensorflow with support for: FMA, AVX, AVX2, SSE4.1, SSE4.2
pip install pip==9.0.3 # pip 10.0.1 has issues with pybind11 -> required by fastText
# install common libraries
pip install psutil
pip install tensorflow==1.13.1 # tensorflow includes numpy
pip install scipy sklearn
pip install matplotlib seaborn imageio
pip install sortedcontainers
# install gym
pip install gym[atari]