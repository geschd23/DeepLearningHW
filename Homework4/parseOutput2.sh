#!/usr/bin/env bash

for f in $(ls $1);
do
	echo "$f, "
	for l in $(cat $f | grep "VALIDATION DISTANCE\|TRAIN DISTANCE" | sed 's/ //g');
	do
		echo -n '"'
		echo -n $l | sed 's/.*://g';
		echo -n '",'
		if echo $l | grep -q "VALIDATION"
		then
			echo ""
		fi
	done
	echo ""
done
