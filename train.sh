#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=82G
#SBATCH --cpus-per-task=4
#SBATCH --time=8:00:00
#SBATCH --job-name=demucs_eval

# Email notifications
#SBATCH --mail-user=cyyeung3@sheffield.ac.uk
#SBATCH --mail-type=ALL

# Load the required modules
export LMOD_DISABLE_SAME_NAME_AUTOSWAP="no"
module load Python/3.11.3-GCCcore-12.3.0
module load FFmpeg/6.0-GCCcore-12.3.0
module load libsndfile/1.0.28-GCCcore-10.2.0
module load NCCL/2.16.2-GCCcore-12.2.0-CUDA-12.0.0

# Add PyTorch distributed environment variables
# export MASTER_ADDR=$(hostname -s)
# export MASTER_PORT=29500
# export WORLD_SIZE=2
# export NCCL_DEBUG=INFO


# Print job info for debugging
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Allocated GPU: $CUDA_VISIBLE_DEVICES"
nvidia-smi

# to see locally installed soundstretch 2.0.0
export PATH="/users/aca22cyy/.local/bin:$PATH"
cd /users/aca22cyy/demucs
source venv/bin/activate

# train the default model
# batch_size is for hydra
# +gpu, +mem_per_gpu is for dora
# dora run -d -f 81de367c batch_size=64 +gpu=4 +mem_per_gpu=78

# evaluate the default model
RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 python -m tools.test_pretrained -n htdemucs test.distorted=True
