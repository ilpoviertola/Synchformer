#!/bin/bash

# Define the following from CLI
#SBATCH --job-name=ts
#SBATCH --account=project_2004994
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1,nvme:500
#SBATCH --cpus-per-task=32
#SBATCH --partition=gpusmall
#SBATCH --time=36:00:00

#SBATCH --output=./sbatch_logs/%J.log
#SBATCH --error=./sbatch_logs/%J.log
#SBATCH --verbose

# exit when any command fails
set -e

## The following will assign a master port (picked randomly to avoid collision) and an address for ddp.
# We want names of master and slave nodes. Make sure this node (MASTER_ADDR) comes first
MASTER_ADDR=`/bin/hostname -s`
if (( $SLURM_JOB_NUM_NODES > 1 )); then
    WORKERS=`scontrol show hostnames $SLURM_JOB_NODELIST | grep -v $MASTER_ADDR`
fi
# Get a random unused port on this host(MASTER_ADDR)
MASTER_PORT=`comm -23 <(seq 49152 65535 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1`
export MASTER_PORT=$MASTER_PORT
export MASTER_ADDR=$MASTER_ADDR
echo "MASTER_ADDR" $MASTER_ADDR "MASTER_PORT" $MASTER_PORT "WORKERS" $WORKERS

# for AMD GPUs at LUMI
export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-cache-${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
export MIOPEN_DEBUG_CONV_IMPLICIT_GEMM=0
echo "MIOPEN_DEBUG_CONV_IMPLICIT_GEMM:" $MIOPEN_DEBUG_CONV_IMPLICIT_GEMM

# loading conda environment
export PATH="/projappl/project_2004994/viertoli/Synchformer/conda_env/bin:$PATH"

srun python main.py \
    start_time="24-03-08T18-50-21" \
    config="./configs/gh_sync.yaml" \
    logging.logdir="/projappl/project_2004994/viertoli/logs/sync/sync_models/" \
    data.vids_path="/scratch/project_2004994/viertoli/datasets/greatesthit/vis-data-256_h264_video_25fps_256side_16000hz_aac_len_5_splitby_random" \
    data.dataset.params.meta_path="/scratch/project_2004994/viertoli/datasets/greatesthit/vis-data-256_h264_video_25fps_256side_16000hz_aac_len_5_splitby_random/metadata.csv" \
    training.base_batch_size=32 \
    training.num_workers=32 \
    training.patience=40 \
    training.resume=True \
    ckpt_path="/projappl/project_2004994/viertoli/logs/sync/sync_models/24-03-08T18-50-21/24-03-08T18-50-21_latest.pt"
