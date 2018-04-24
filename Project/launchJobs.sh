#!/usr/bin/env bash
id=1
for a in '--learning_rate 1.0' '--learning_rate 0.1' '--learning_rate 0.01'; #'--learning_rate 0.01' '--learning_rate 1' '--learning_rate 0.0001';
do
for b in '--beta 1.0' '--beta 10.0' '--beta 0.1';
do
for c in '--eta 0.01' '--eta 0.1'; # '--eta 0.0';
do
for d in '--use_advantage';
do
for e in '--parallel 1';
do
for f in '--t_max 40';
do
	echo $a $b $c $d $e $f
	sbatch run_py_cpu.sh $a $b $c $d $e $f --output 4
	id=$((id +1))
done
done
done
done
done
done
