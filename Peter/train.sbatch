#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=2:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --account=pr_101_general
#SBATCH --reservation=cds-hackathon
#SBATCH --job-name=hackathon
#SBATCH --output=output.out


module = purge

singularity exec --nv \
		--overlay /scratch/$USER/my_env/overlay-15GB-500K.ext3:ro \
		/scratch/work/public/singularity/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif \
		/bin/bash -c "source /ext3/env.sh; bash train.sh"