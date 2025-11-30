#!/bin/bash
#SBATCH --job-name=chfV6_levir
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40g
#SBATCH --time=36:00:00

# 1) módulos/conda
source ~/.bashrc
module load miniconda/3.0
conda activate changeformer

# 2) cd al repo
cd $SLURM_SUBMIT_DIR

# 3) lanzar entrenamiento
python main_cd.py \
  --img_size 256 \
  --checkpoint_root checkpoints_training_OG_2_256_batch_32_30_epochs \
  --vis_root vis \
  --lr_policy linear \
  --optimizer adamw \
  --pretrain ./pretrained_changeformer/segformer.b2.512x512.ade.160k.pth \
  --split train \
  --split_val val \
  --net_G ChangeFormerV6 \
  --multi_scale_train True \
  --multi_scale_infer False \
  --gpu_ids 0 \
  --max_epochs 30 \
  --project_name . \
  --batch_size 32 \
  --shuffle_AB False \
  --data_name LEVIR \
  --lr 0.0001 \
  --embed_dim 256 \
  --num_workers 8
