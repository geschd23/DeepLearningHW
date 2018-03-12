#!/usr/bin/env bash

for f in $(ls $1/*.out);
do
	echo -n "$f, "
	for l in $(cat $f | grep "regularizer\|dropout_rate\|learning_rate\|filters\|linear_nodes\|Final VALIDATION ACCURACY\|Average accuracy across k folds" | sed 's/ //g');
	do
		echo -n '"'
		echo -n $l | sed 's/.*://g';
		echo -n '",'
	done
	echo ""
done
