#!/usr/bin/env bash

python -m pysc2.bin.agent --map MoveToBeacon --agent pysc2.agents.rl_agent.RlAgent --screen_resolution 64  --parallel 1 --t_max 40 --output 2
