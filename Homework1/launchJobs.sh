#!/usr/bin/env bash

id=1
for a in '' '--l2_regularizer';
do
	for b in '--keep_probability 1.0' '--keep_probability 0.7';
	do
		for c in '--learning_rate 0.01' '--learning_rate 0.001'  '--learning_rate 0.0001';
		do
			for d in '--architecture 50' '--architecture 256,128' '--architecture 256,128,256';
			do
				sbatch run_py.sh main.py $a $b $c $d
				id=$((id +1))
			done
		done
	done
done
