#!/bin/bash

## first argument is experiment, then rank, then iterations and alpha
seed=(0 100 200 300 400 500 600 700 800 900)
j=$1

for i in "${seed[@]}"
do
	echo "$(python predictSample.py "data/cms-tensor-{0}.dat" $j $2 $3 $4 -t 0.5 -g 1e-4 1e-2 1e-2 -s $i)"
	j=$(($j+1))
done