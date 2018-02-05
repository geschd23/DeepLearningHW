#!/bin/sh
#SBATCH --time=02:00:00          # Run time in hh:mm:ss
#SBATCH --mem=32000              # Maximum memory required (in megabytes)
#SBATCH --job-name=deep-singularity
#SBATCH --partition=gpu
#SBATCH --gres=gpu
#SBATCH --constraint=[gpu_k20|gpu_k40|gpu_p100]
#SBATCH --error=/work/cse496dl/dgeschwe/output/job.err
#SBATCH --output=/work/cse496dl/dgeschwe/output/job.out
#SBATCH --qos=short		 # 6 hour job run time max

module load singularity
singularity exec docker://unlhcc/sonnet-gpu python3 -u $@
