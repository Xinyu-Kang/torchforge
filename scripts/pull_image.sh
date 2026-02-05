#!/bin/bash
#SBATCH --job-name=forge_image
#SBATCH --nodes=3              # <-- set this
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --partition=amd-rccl   # <-- set this
#SBATCH --qos=normal           # <-- set this 
#SBATCH --nodelist=useocpm2m-097-[089,094,112]
#SBATCH --output=slurm_pull_%j.out
#SBATCH --error=slurm_pull_%j.err

set -euo pipefail

IMAGE="rocm/pytorch-private:torchforge-deps-rocm7.1-20260205-v1"

echo "Pulling image on all nodes..."
srun --ntasks-per-node=1 --nodes=${SLURM_JOB_NUM_NODES} \
  bash -lc "hostname; docker pull ${IMAGE}"
