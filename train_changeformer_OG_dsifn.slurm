#!/bin/bash
#SBATCH --job-name=chfV6_OG_dsifn
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=24:00:00

# Información del entorno
echo "Running on node: $HOSTNAME"
nvidia-smi

# 1) módulos/conda
source ~/.bashrc
module load miniconda/3.0
conda activate changeformer

# 2) cd al repo
cd $SLURM_SUBMIT_DIR

python main_cd.py \
  --img_size 256 \
  --checkpoint_root checkpoints_training_OG_3_256_32_batch_30_epochs \
  --vis_root vis \
  --lr_policy linear \
  --optimizer adamw \
  --pretrain ./CD_ChangeFormerV6_LEVIR_b16_lr0.0001_adamw_train_test_200_linear_ce_multi_train_True_multi_infer_False_shuffle_AB_False_embed_dim_256/best_ckpt.pt \
  --split train \
  --split_val test \
  --net_G ChangeFormerV6 \
  --multi_scale_train True \
  --multi_scale_infer False \
  --gpu_ids 0 \
  --max_epochs 30 \
  --project_name . \
  --batch_size 32 \
  --shuffle_AB False \
  --data_name DSIFN \
  --lr 0.00006 \
  --embed_dim 256 \
  --num_workers 2
