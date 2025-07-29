#!/bin/bash
PROJECT_NAME=$1
shift
DATASET=$1
shift
OTHER_ARGS="$@"

CACHE_DIR=/network/scratch/d/d.williams-king/latent-adversarial-training-personality/cache

sbatch <<EOF
#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=lat-$PROJECT_NAME
#SBATCH --time=48:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:l40s
#SBATCH -o job/lat-$PROJECT_NAME.out
#SBATCH -e job/lat-$PROJECT_NAME.err

cd /network/scratch/d/d.williams-king/latent-adversarial-training-personality

module load cuda/12.6.0/cudnn/9.3

export PYTHONBREAKPOINT=0
export WANDB_MODE=disabled

export PYTHONPATH=.
. /network/scratch/d/d.williams-king/ccc.sh
conda activate lat

python lat_personality.py --cache-dir $CACHE_DIR --data-folder $DATASET --project-name $PROJECT_NAME $OTHER_ARGS
EOF
