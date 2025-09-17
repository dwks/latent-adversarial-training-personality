#!/bin/bash
PROJECT_NAME=$1
shift
OTHER_ARGS="$@"

CACHE_DIR=/network/scratch/l/let/projects/latent-adversarial-training-personality/cache

sbatch <<EOF
#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=$PROJECT_NAME
#SBATCH --time=24:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:l40s:1
#SBATCH -o job/$PROJECT_NAME.out
#SBATCH -e job/$PROJECT_NAME.err

cd /network/scratch/l/let/projects/latent-adversarial-training-personality

module load cuda/12.6.0/cudnn/9.3

export PYTHONBREAKPOINT=0
export WANDB_MODE=disabled

export PYTHONPATH=.

#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

module load anaconda
conda activate /network/scratch/l/let/envs/lat

python lat_personality.py $OTHER_ARGS
EOF
