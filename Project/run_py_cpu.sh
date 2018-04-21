#!/bin/sh
#SBATCH --time=02:00:00          # Run time in hh:mm:ss
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=12000              # Maximum memory required (in megabytes)
#SBATCH --job-name=SC2LE
#SBATCH --error=/work/cse496dl/dgeschwe/output/job.%J.err
#SBATCH --output=/work/cse496dl/dgeschwe/output/job.%J.out

cd /home/choueiry/dgeschwe/496/DeepLearningHW/Project
export SC2PATH=/home/choueiry/dgeschwe/496/project/StarCraftII
export PYTHONPATH=$PYTHONPATH:$PWD/pysc2-master
echo "$@"

module load singularity
singularity exec docker://unlhcc/tensorflow-gpu python -m pysc2.bin.agent --map MoveToBeacon --agent pysc2.agents.rl_agent.RlAgent --screen_resolution 64 --max_agent_steps 100000 $@
