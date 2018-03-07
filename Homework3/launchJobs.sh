#!/usr/bin/env bash

id=1
for a in '--l2_regularizer 0.0' '--l2_regularizer 0.0001';
do
	for b in '--dropout_rate 0.0' '--dropout_rate 0.2';
	do
		for c in '--code_size 100' '--code_size 200';
		do
			for d in '--filters 2,0,4,0' '--filters 16,0,32,0' '--filters 2,2,0,4,4,0' '--filters 16,16,0,32,32,0';
			do
				sbatch run_py.sh main.py $a $b $c $d --output_model --save_dir output/run$id
				id=$((id +1))
			done
		done
	done
done
