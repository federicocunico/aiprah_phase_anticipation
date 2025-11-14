#!/usr/bin/env python3
"""Quick test for SWAG model only"""
import os
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from related.phase.swag import SWAG

# Create model
model = SWAG(
    img_size=224,
    num_classes=6,
    seq_length=1440,
    window_size=20,
    num_context_tokens=24,
    decoder_type='single_pass',
    use_prior_knowledge=False,
    max_horizon=60
).cuda()

# Test forward pass
frames = torch.randn(4, 16, 3, 224, 224).cuda()
print(f"Input shape: {frames.shape}")

try:
    recognition_logits, anticipation_logits = model(frames)
    print(f"Recognition logits shape: {recognition_logits.shape}")
    print(f"Anticipation logits shape: {anticipation_logits.shape}")
    print("✅ SWAG forward pass successful!")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
