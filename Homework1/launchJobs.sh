#!/usr/bin/env bash

for a in '' '--l2_regularizer';
do
	for b in '--keep_probability 1.0' '--keep_probability 0.7';
	do
		for c in '--learning_rate 0.01' '--learning_rate 0.001'  '--learning_rate 0.0001';
		do
			for d in '--architecture "50"' '--architecture "256 128"' '--architecture "256 128 256"';
			do
				echo "sbatch run_py.sh main.py $a $b $c $d"
			done
		done
	done
done
