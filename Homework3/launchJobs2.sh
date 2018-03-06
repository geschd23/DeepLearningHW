#!/usr/bin/env bash

id=1
for a in '' '--l2_regularizer';
do
	for b in '--dropout_rate 0.0' '--dropout_rate 0.2';
	do
		for c in '--learning_rate 0.001' '--learning_rate 0.0001';
		do
			for d in '' '--linear_nodes 4' '--linear_nodes 4,4' '--linear_nodes 8' '--linear_nodes 8,8' '--linear_nodes 16' '--linear_nodes 16,16';
			do
				sbatch run_py.sh main.py $a $b $c $d --dataset SAVEE-British --model_transfer model4/emodb_homework_2-0
				id=$((id +1))
			done
		done
	done
done
