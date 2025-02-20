#!/bin/bash

#SBATCH --time=10:00:00
#SBATCH --mem=360G
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --output=../log/slurm-%j.out
#SBATCH --gres=gpu:a100:1

module load Anaconda3

python3 -c "import sys; print('\n'.join(sys.path))"

export PATH='/scratch/user/bokun-wang/sogclr_sa/bin:$PATH'

source activate /scratch/user/bokun-wang/sogclr_sa

python3 -c "import sys; print('\n'.join(sys.path))"

export TRANSFORMERS_CACHE=/scratch/user/bokun-wang
export HUGGINGFACE_HUB_CACHE=/scratch/user/bokun-wang

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=9601 \
  --use_env ../clip.py --data_path /scratch/group/optmai/datasets \
  --ann_path /scratch/group/optmai/bokun/clip_train --train_file cc3m_train.json --train_image_root cc3m \
  --zs_dataset=cifar100 --zs_datafolder /scratch/group/optmai/datasets --ita_type infonce \
  --batch_size_test 256 --evaluate \
  --checkpoint /scratch/user/bokun-wang/bimodal_outputs/clip_cc3m_e30_dist_512-10686359/checkpoint_30.pth \
  --output_dir /scratch/group/optmai/bokun/bimodal_sa/cifar100-evaluate/clip_cc3m_seed0 \
