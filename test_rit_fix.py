#!/usr/bin/env python3
"""Test RiT model with backward pass to verify in-place operation fix"""
import os
import torch
import torch.nn as nn
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from related.triplet.rit import RiT

print("Testing RiT model with backward pass...")

# Create model
model = RiT(
    num_instruments=6,
    num_verbs=4,
    num_targets=14,
    num_triplets=56,
    clip_size=6,
    num_tam_layers=2,
    pretrained=False
).cuda()

# Test input (B, m, C, H, W)
frames = torch.randn(4, 6, 3, 224, 224).cuda()
print(f"Input shape: {frames.shape}")

# Forward pass
try:
    outputs = model(frames)
    print(f"✓ Forward pass successful")
    print(f"  Verb logits shape: {outputs['verb_logits'].shape}")
    print(f"  Target logits shape: {outputs['target_logits'].shape}")
except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test backward pass
try:
    criterion = nn.CrossEntropyLoss()
    verb_gt = torch.randint(0, 4, (4,)).cuda()
    target_gt = torch.randint(0, 14, (4,)).cuda()
    
    loss_verb = criterion(outputs['verb_logits'], verb_gt)
    loss_target = criterion(outputs['target_logits'], target_gt)
    loss = loss_verb + loss_target
    
    loss.backward()
    print(f"✓ Backward pass successful")
    print(f"  Loss: {loss.item():.4f}")
    print(f"✓✓ RiT model fixed successfully!")
except Exception as e:
    print(f"✗ Backward pass failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
