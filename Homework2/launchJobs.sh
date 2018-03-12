#!/usr/bin/env bash

id=1
for a in '' '--l2_regularizer';
do
	for b in '--dropout_rate 0.0' '--dropout_rate 0.2';
	do
		for c in '--learning_rate 0.001' '--learning_rate 0.0001';
		do
			for d in '--filters 2,0,4,0' '--filters 16,0,32,0' '--filters 2,0,4,0,8,0' '--filters 16,0,32,0,64,0' '--filters 2,0,4,0,8,0,16,0' '--filters 16,0,32,0,64,0,128,0';
			do
				sbatch run_py.sh main.py $a $b $c $d
				id=$((id +1))
			done
		done
	done
done
