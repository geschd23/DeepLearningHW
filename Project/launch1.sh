#!/usr/bin/env bash
export SC2PATH=/home/choueiry/dgeschwe/496/project/StarCraftII
export PYTHONPATH=$PYTHONPATH:$PWD/pysc2-master
srun --qos=short --nodes=1 --ntasks-per-node=1 --mem-per-cpu=4096 --pty $SHELL
