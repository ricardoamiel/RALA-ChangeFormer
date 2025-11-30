# RALA-ChangeFormer

This repository contains the full implementation used for the experiments in **“Efficient Multi-Temporal Building Change Detection with Reduced-Rank Linear Attention”**.  
Our work extends the original **ChangeFormer** architecture by replacing its encoder Multi-Head Self-Attention (MHSA) modules with **Rank-Augmented Linear Attention (RALA)**, enabling more memory-efficient and scalable training for Very High Resolution (VHR) Building Change Detection (BCD).

The codebase includes:
- Training and evaluation scripts for **LEVIR-CD** and **DSIFN-CD**
- Our modified RALA modules
- Utilities for dataset preparation, tiling, evaluation, and visualization
- Tools to extract **attention maps** for both ChangeFormer and RALA-ChangeFormer
- Checkpoint loaders compatible with the original ChangeFormer structure

Get the article version on: [Paper](https://drive.google.com/file/d/1V3DqeU24k31jUTC7EOaSIX1WYncus33Y/view?usp=sharing)

---

## Overview

RALA-ChangeFormer preserves the hierarchical Siamese encoder–decoder structure of ChangeFormer, but replaces its quadratic MHSA with a **linear-time** rank-augmented alternative.  
This modification reduces attention-related GFLOPs and peak VRAM usage, while maintaining segmentation accuracy on LEVIR-CD and DSIFN-CD.

Key highlights:
- Near-identical accuracy to ChangeFormer  
- ~20–30% memory savings  
- Larger feasible batch sizes (up to 1.5×–2× depending on resolution)  
- Support for extracting attention maps for empirical comparison  

---

## Repository Structure

ChangeFormer/ \
│ \
├── checkpoints/ # Pretrained & intermediate checkpoints \
├── datasets/ # LEVIR-CD and DSIFN-CD (processed) \
├── images/ # Figures for reports/paper \
├── logs_training_* # Training logs from multiple experiments \
│ \
├── models/ # ChangeFormer + RALA modules (modified) \
├── scripts/ # SLURM + training/eval scripts \
├── outputs/ # Predictions, visualizations, metrics \
│
├── demo_LEVIR.py # Demo inference on LEVIR-CD samples \
├── demo_DSIFN.py # Demo inference on DSIFN-CD samples \
├── eval_cd.py # Evaluation pipeline \
├── main_cd.py # Main training entry point \
│
├── make_tiles_256.py # Patch extraction utility \
├── create_list_txt.py # Train/val/test split generator \
│ \
├── requirements.txt \
├── environment.yml \
└── README.md \


---

## Requirements

This version uses the same software environment as the original ChangeFormer to ensure fair comparison:

Install all dependencies:

```bash
pip install -r requirements.txt
```

Or recreate the conda environment:

```bash
conda env create -f environment.yml
conda activate ChangeFormer
```

## Dataset preparation

Each dataset should follow this structure:

```bash
A/          # t1 images
B/          # t2 images
label/      # binary change masks
list/
   train.txt
   val.txt
   test.txt
```

You may generate 256×256 tiles using:

```bash
python make_tiles_256.py
```

Train/val/test splits can be produced with:

```bash
python create_list_txt.py
```

## Training

### LEVIR-CD

```bash
sh scripts/train_changeformer_RALA_levir.slurm
```

### DSIFN-CD

```bash
sh scripts/train_changeformer_RALA_dsifn.slurm
```
Both SLURM scripts expose all hyperparameters (batch size, LR, multi-scale training, image size, etc.)
All RALA modules are automatically loaded when using the --net_G ChangeFormerV7 configuration.

## Evaluation

### LEVIR-CD

```bash
sh scripts/eval_changeformer_LEVIR.sh
```

### DSIFN-CD

```bash
sh scripts/eval_changeformer_DSIFN.sh
```

results_eval/
vis/


## Citation

If you use this repository or ideas from our work, please cite:

```bibtex
@article{rala_changeformer_2025,
  title={Efficient Multi-Temporal Building Change Detection with Reduced-Rank Linear Attention},
  author={Anonymous},
  journal={Under Review},
  year={2025}
}
```

For the original ChangeFormer:

```bibtex
@INPROCEEDINGS{Bandara2022ChangeFormer,
  author={Bandara, W. G. C. and Patel, V. M.},
  title={A Transformer-Based Siamese Network for Change Detection},
  booktitle={IGARSS 2022},
  year={2022},
  pages={207--210}
}
```

## License

This repository is released for research and academic purposes only.
For commercial use, please contact the authors of the original ChangeFormer.
