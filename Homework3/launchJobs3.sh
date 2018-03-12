#!/usr/bin/env bash

id=1
for a in '--l2_regularizer 0.0' '--l2_regularizer 0.1' '--l2_regularizer 0.01' '--l2_regularizer 0.001' '--l2_regularizer 0.0001';
do
	for b in '--dropout_rate 0.0';
	do
		for c in '--learning_rate 0.0001';
		do
			for d in '--filters 2,0,4,0' '--filters 4,8,0,8,16,0';
			do
				sbatch run_py.sh main.py $a $b $c $d
				id=$((id +1))
			done
		done
	done
done
