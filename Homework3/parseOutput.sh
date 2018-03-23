#!/usr/bin/env bash

for f in $(find $1 -name *.txt);
do
	echo -n "$f, "
	for l in $(cat $f | grep "normalize\|float16\|regularizer\|dropout_rate\|learning_rate\|filters\|code_size\|Final VALIDATION PSNR\|Average psnr across k folds" | sed 's/ //g');
	do
		echo -n '"'
		echo -n $l | sed 's/.*://g';
		echo -n '",'
	done
	echo ""
done
