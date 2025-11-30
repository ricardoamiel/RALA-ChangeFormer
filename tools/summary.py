import torch
import torch.nn as nn
from models.ChangeFormer import ChangeFormerV6, ChangeFormerV7
import argparse
import csv
import os
from torchsummary import summary

#model = ChangeFormerV6() #ChangeFormerV7)()
model = ChangeFormerV7()

total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total: {total}, Entrenables: {trainable}")
