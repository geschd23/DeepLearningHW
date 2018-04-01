#!/usr/bin/env bash
dirID=$(ls | grep ^output | sed 's/output//g' | sort -n | tail -n 1)
dirID=$((dirID +1))
id=1
for a in '--float16';
do
	for b in '--l2_regularizer 0.0' '--l2_regularizer 0.001';
	do
		for c in '--code_size 200';
		do
			for d in '--filters 8,16,32' '--filters 8,16,32,0,8,16,32' '--filters 8,16,32,0,8,16,32,0,8,16,32' ;
			do
				for e in '--normalize';
				do
					sbatch run_py.sh main.py $a $b $c $d $e --output_model --save_dir output${dirID}/run${id}
					id=$((id +1))
				done
			done
		done
	done
done
