#!/usr/bin/env bash
dirID=$(ls | grep ^output | sed 's/output//g' | sort -n | tail -n 1)
dirID=$((dirID +1))
id=1
for a in '--nodes 50' '--nodes 50,50' '--nodes 100' '--nodes 100,100';
do
	echo $a
	sbatch run_py.sh main.py $a --glove /home/choueiry/dgeschwe/496/data/glove.6B.50d.txt --save_dir output${dirID}/run${id}
	id=$((id +1))
done
