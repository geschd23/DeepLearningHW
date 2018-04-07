#!/usr/bin/env bash
dirID=$(ls | grep ^output | sed 's/output//g' | sort -n | tail -n 1)
dirID=$((dirID +1))
id=1
for a in '--nodes 100,100';
do
for b in '--prediction_length 5'  '--prediction_length 3'  '--prediction_length 1';
do
for c in '--learning_rate 0.00001';
do
for d in '--l2_regularizer 0.0';
do
	sbatch run_py.sh main.py $a $b $c $d --glove /home/choueiry/dgeschwe/496/data/glove.6B.50d.nopunctuation.txt --save_dir output${dirID}/run${id}
	id=$((id +1))
done
done
done
done
