#!/usr/bin/env bash

for f in $(find $1 -name *.txt);
do
	echo -n "$f	"
	for l in $(cat $f | grep "prediction_length\|regularizer\|learning_rate\|nodes\|Final VALIDATION DISTANCE" | sed 's/ //g');
	do
		echo -n '"'
		echo -n $l | sed -e "s/b'\(.*\)'/\1/g" -e 's/.*://g';
		echo -n '"	'
	done
	echo ""
done
