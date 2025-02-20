#!/bin/bash

#SBATCH --time=96:00:00
#SBATCH --mem=360G
#SBATCH --nodes=1
#SBATCH --tasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --output=../log/slurm-%j.out
#SBATCH --gres=gpu:a100:2

module load Anaconda3

python3 -c "import sys; print('\n'.join(sys.path))"

export PATH='/scratch/user/bokun-wang/sogclr_sa/bin:$PATH'

source activate /scratch/user/bokun-wang/sogclr_sa

python3 -c "import sys; print('\n'.join(sys.path))"

export TRANSFORMERS_CACHE=/scratch/user/bokun-wang
export HUGGINGFACE_HUB_CACHE=/scratch/user/bokun-wang

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=9510 \
  --use_env ../clip.py --data_path /scratch/group/optmai/datasets \
  --ann_path /scratch/group/optmai/bokun/clip_train --train_file cc3m_train.json --train_image_root cc3m \
  --output_dir /scratch/user/bokun-wang/bimodal_outputs/infonce_cc3m_e30_dist_512-$SLURM_JOB_ID --zs_dataset=cifar100 \
  --zs_datafolder /scratch/group/optmai/datasets --init_model --ita_type infonce \
  --sched cosine --distributed --epochs 30 --batch_size_train 256 --batch_size_test 256 --temp 0.01 --seed 0
