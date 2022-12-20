#!/bin/bash
arr=($(cat config.yml | grep "name:"))
name=${arr[1]}
name="${name//\"}"
echo $name


mkdir experiments/$name

cp config.yml  experiments/$name/${name}_config.yml
accelerate launch train.py --config_path experiments/$name/${name}_config.yml > experiments/$name/${name}_train.txt
python -u test.py --config_path experiments/$name/${name}_config.yml > experiments/$name/${name}_test.txt
