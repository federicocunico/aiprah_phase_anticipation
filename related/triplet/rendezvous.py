import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import time


class WSL(nn.Module):
    """Weakly Supervised Localization module for instrument detection"""
    def __init__(self, in_channels=512, num_instruments=6):
        super(WSL, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.cam_conv = nn.Conv2d(64, num_instruments, kernel_size=1)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        cam = self.cam_conv(x)  # Class Activation Maps
        logits = F.adaptive_max_pool2d(cam, 1).flatten(1)
        return cam, logits


class CAGAM(nn.Module):
    """Class Activation Guided Attention Mechanism"""
    def __init__(self, in_channels=512, num_instruments=6, num_classes=10, 
                 attention_type='channel'):
        super(CAGAM, self).__init__()
        self.attention_type = attention_type
        self.num_instruments = num_instruments
        
        # Context mapping
        self.context_conv = nn.Conv2d(in_channels, num_instruments, kernel_size=1)
        
        # Query, Key projections
        self.query_conv = nn.Conv2d(num_instruments, num_instruments, kernel_size=1)
        self.key_conv = nn.Conv2d(num_instruments, num_instruments, kernel_size=1)
        self.value_conv = nn.Conv2d(num_instruments, num_instruments, kernel_size=1)
        
        # For CAM
        self.cam_query = nn.Conv2d(num_instruments, num_instruments, kernel_size=1)
        self.cam_key = nn.Conv2d(num_instruments, num_instruments, kernel_size=1)
        
        # Output
        self.output_conv = nn.Conv2d(num_instruments, num_classes, kernel_size=1)
        
        self.beta = nn.Parameter(torch.zeros(1))
        
    def forward(self, x, cam):
        B, C, H, W = x.shape
        
        # Get context features
        context = F.relu(self.context_conv(x))  # [B, num_instruments, H, W]
        
        # Generate Q, K, V from context
        Q = self.query_conv(context)  # [B, C_I, H, W]
        K = self.key_conv(context)
        V = self.value_conv(context)
        
        # Generate Q_D, K_D from CAM (discriminative)
        cam_resized = F.interpolate(cam, size=(H, W), mode='bilinear', align_corners=False)
        Q_D = self.cam_query(cam_resized)
        K_D = self.cam_key(cam_resized)
        
        if self.attention_type == 'channel':
            # Channel attention for verbs
            Q_flat = Q.view(B, self.num_instruments, -1)  # [B, C_I, HW]
            K_flat = K.view(B, self.num_instruments, -1)
            
            # Non-discriminative affinity
            P_nd = torch.matmul(Q_flat.transpose(1, 2), K_flat)  # [B, HW, HW] -> need [B, C_I, C_I]
            P_nd = torch.matmul(Q_flat, K_flat.transpose(1, 2)) / (H * W)  # [B, C_I, C_I]
            
            # Discriminative affinity
            Q_D_flat = Q_D.view(B, self.num_instruments, -1)
            K_D_flat = K_D.view(B, self.num_instruments, -1)
            P_d = torch.matmul(Q_D_flat, K_D_flat.transpose(1, 2)) / (H * W)  # [B, C_I, C_I]
            
            # Attention
            scale = self.num_instruments ** 0.5
            attention = F.softmax((P_d * P_nd) / scale, dim=-1)  # [B, C_I, C_I]
            
            # Apply attention
            V_flat = V.view(B, self.num_instruments, -1)  # [B, C_I, HW]
            enhancement = torch.matmul(attention, V_flat)  # [B, C_I, HW]
            enhancement = enhancement.view(B, self.num_instruments, H, W)
            
        else:  # position attention for targets
            # Position attention
            Q_flat = Q.view(B, self.num_instruments, -1).transpose(1, 2)  # [B, HW, C_I]
            K_flat = K.view(B, self.num_instruments, -1).transpose(1, 2)  # [B, HW, C_I]
            
            # Non-discriminative affinity
            P_nd = torch.matmul(Q_flat, K_flat.transpose(1, 2))  # [B, HW, HW]
            
            # Discriminative affinity
            Q_D_flat = Q_D.view(B, self.num_instruments, -1).transpose(1, 2)
            K_D_flat = K_D.view(B, self.num_instruments, -1).transpose(1, 2)
            P_d = torch.matmul(Q_D_flat, K_D_flat.transpose(1, 2))  # [B, HW, HW]
            
            # Attention
            scale = self.num_instruments ** 0.5
            attention = F.softmax((P_d * P_nd) / scale, dim=-1)  # [B, HW, HW]
            
            # Apply attention
            V_flat = V.view(B, self.num_instruments, -1).transpose(1, 2)  # [B, HW, C_I]
            enhancement = torch.matmul(attention, V_flat)  # [B, HW, C_I]
            enhancement = enhancement.transpose(1, 2).view(B, self.num_instruments, H, W)
        
        # Add enhancement
        enhanced = context + self.beta * enhancement
        
        # Output activation maps
        output_maps = self.output_conv(enhanced)
        logits = F.adaptive_avg_pool2d(output_maps, 1).flatten(1)
        
        return output_maps, logits


class MultiHeadMixedAttention(nn.Module):
    """Multi-Head of Mixed Attention (self + cross attention)"""
    def __init__(self, num_triplets=100, num_instruments=6, num_verbs=10, 
                 num_targets=15, d_model=100):
        super(MultiHeadMixedAttention, self).__init__()
        
        self.d_model = d_model
        self.num_heads = 4
        
        # Projection functions for Q, K, V
        self.triplet_q_fc = nn.Linear(num_triplets, d_model)
        self.triplet_k_fc = nn.Linear(num_triplets, d_model)
        
        self.instrument_k_fc = nn.Linear(num_instruments, d_model)
        self.verb_k_fc = nn.Linear(num_verbs, d_model)
        self.target_k_fc = nn.Linear(num_targets, d_model)
        
        # Value convolutions
        self.triplet_v_conv = nn.Conv2d(num_triplets, d_model, kernel_size=1)
        self.instrument_v_conv = nn.Conv2d(num_instruments, d_model, kernel_size=1)
        self.verb_v_conv = nn.Conv2d(num_verbs, d_model, kernel_size=1)
        self.target_v_conv = nn.Conv2d(num_targets, d_model, kernel_size=1)
        
        # Output projection
        self.output_conv = nn.Conv2d(d_model * 4, num_triplets, kernel_size=1)
        
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, h_ivt, h_i, h_v, h_t):
        B, _, H, W = h_ivt.shape

        # Generate Q from triplet features (sink)
        q_triplet = F.adaptive_avg_pool2d(h_ivt, 1).flatten(1)  # [B, C]
        q_triplet = self.dropout(self.triplet_q_fc(q_triplet))  # [B, d_model]

        # Generate V from all features as spatial tensors
        v_triplet = self.triplet_v_conv(h_ivt).view(B, self.d_model, -1)  # [B, d_model, HW]
        v_instrument = self.instrument_v_conv(h_i).view(B, self.d_model, -1)
        v_verb = self.verb_v_conv(h_v).view(B, self.d_model, -1)
        v_target = self.target_v_conv(h_t).view(B, self.d_model, -1)

        # Use V as keys as well (simple and robust): K = V
        k_triplet = v_triplet  # [B, d_model, HW]
        k_instrument = v_instrument
        k_verb = v_verb
        k_target = v_target

        # Compute scaled dot-product attention across spatial locations
        scale = self.d_model ** 0.5

        # q: [B, d_model] -> [B, 1, d_model]
        q = q_triplet.unsqueeze(1)

        # For each source, compute attention weights over HW: [B,1,HW]
        attn_triplet = torch.matmul(q, k_triplet) / scale
        attn_triplet = F.softmax(attn_triplet, dim=-1)  # [B,1,HW]
        out_triplet = attn_triplet * v_triplet  # broadcast -> [B, d_model, HW]

        attn_instrument = torch.matmul(q, k_instrument) / scale
        attn_instrument = F.softmax(attn_instrument, dim=-1)
        out_instrument = attn_instrument * v_instrument

        attn_verb = torch.matmul(q, k_verb) / scale
        attn_verb = F.softmax(attn_verb, dim=-1)
        out_verb = attn_verb * v_verb

        attn_target = torch.matmul(q, k_target) / scale
        attn_target = F.softmax(attn_target, dim=-1)
        out_target = attn_target * v_target

        # Concatenate all attention outputs along channel (d_model) dim: each is [B, d_model, HW]
        combined = torch.cat([out_triplet, out_instrument, out_verb, out_target], dim=1)  # [B, 4*d_model, HW]
        combined = combined.view(B, -1, H, W)

        # Project back to triplet space
        output = self.output_conv(combined)

        return output


class TransformerDecoderLayer(nn.Module):
    """Single decoder layer with MHMA and feed-forward"""
    def __init__(self, num_triplets=100, num_instruments=6, num_verbs=10, num_targets=15):
        super(TransformerDecoderLayer, self).__init__()
        
        self.mhma = MultiHeadMixedAttention(num_triplets, num_instruments, num_verbs, num_targets)
        
        # Feed-forward network
        self.ff_conv1 = nn.Conv2d(num_triplets, num_triplets * 2, kernel_size=1)
        self.ff_conv2 = nn.Conv2d(num_triplets * 2, num_triplets, kernel_size=1)
        
        self.norm1 = nn.LayerNorm([num_triplets])
        self.norm2 = nn.LayerNorm([num_triplets])
        
    def forward(self, h_ivt, h_i, h_v, h_t):
        # MHMA with residual
        attn_out = self.mhma(h_ivt, h_i, h_v, h_t)
        h_ivt = h_ivt + attn_out
        
        # Layer norm
        B, C, H, W = h_ivt.shape
        h_ivt_norm = self.norm1(h_ivt.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        # Feed-forward with residual
        ff_out = F.relu(self.ff_conv1(h_ivt_norm))
        ff_out = self.ff_conv2(ff_out)
        h_ivt = h_ivt_norm + ff_out
        
        # Layer norm
        h_ivt = self.norm2(h_ivt.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        return h_ivt


class Rendezvous(nn.Module):
    """
    Rendezvous: Attention mechanisms for surgical action triplet recognition
    
    Uses CAGAM for component detection and MHMA decoder for triplet association.
    """
    def __init__(self, num_instruments=6, num_verbs=10, num_targets=15, 
                 num_triplets=100, num_decoder_layers=8, pretrained=True):
        super(Rendezvous, self).__init__()
        
        self.num_instruments = num_instruments
        self.num_verbs = num_verbs
        self.num_targets = num_targets
        self.num_triplets = num_triplets
        
        # Feature extraction backbone (ResNet-18 with modified strides)
        resnet = models.resnet18(pretrained=pretrained)
        
        # Modify strides in last two blocks for higher resolution
        resnet.layer3[0].conv1.stride = (1, 1)
        resnet.layer3[0].downsample[0].stride = (1, 1)
        resnet.layer4[0].conv1.stride = (1, 1)
        resnet.layer4[0].downsample[0].stride = (1, 1)
        
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        # Get low-level features from layer1
        self.layer1 = nn.Sequential(*list(resnet.children())[:5])
        
        # Bottleneck for global triplet features
        self.bottleneck_conv1 = nn.Conv2d(64, 256, kernel_size=3, padding=1)
        self.bottleneck_conv2 = nn.Conv2d(256, num_triplets, kernel_size=1)
        
        # Encoder modules
        self.wsl = WSL(in_channels=512, num_instruments=num_instruments)
        
        self.verb_cagam = CAGAM(in_channels=512, num_instruments=num_instruments, 
                                num_classes=num_verbs, attention_type='channel')
        
        self.target_cagam = CAGAM(in_channels=512, num_instruments=num_instruments,
                                  num_classes=num_targets, attention_type='position')
        
        # Decoder: Stack of transformer layers
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(num_triplets, num_instruments, num_verbs, num_targets)
            for _ in range(num_decoder_layers)
        ])
        
        # Final classifier
        self.classifier = nn.Linear(num_triplets, num_triplets)
        
    def forward(self, x):
        """
        Args:
            x: Input images [B, 3, H, W]
            
        Returns:
            dict containing logits and activation maps
        """
        B = x.size(0)
        
        # Feature extraction
        features = self.backbone(x)  # [B, 512, H/8, W/8] with modified strides
        
        # Low-level features for bottleneck
        low_features = self.layer1(x)  # [B, 64, H/4, W/4]
        
        # Bottleneck: global triplet features
        h_ivt = F.relu(self.bottleneck_conv1(low_features))
        h_ivt = self.bottleneck_conv2(h_ivt)  # [B, num_triplets, H/4, W/4]
        
        # Resize to match other features
        h_ivt = F.interpolate(h_ivt, size=features.shape[2:], mode='bilinear', align_corners=False)
        
        # Encoder: Component detection
        # 1. Instrument detection with WSL
        h_i, y_i = self.wsl(features)  # CAM and logits
        
        # 2. Verb detection with CAGAM (channel attention)
        h_v, y_v = self.verb_cagam(features, h_i)
        
        # 3. Target detection with CAGAM (position attention)
        h_t, y_t = self.target_cagam(features, h_i)
        
        # Decoder: Triplet association via transformer layers
        for layer in self.decoder_layers:
            h_ivt = layer(h_ivt, h_i, h_v, h_t)
        
        # Final classification
        h_ivt_pooled = F.adaptive_avg_pool2d(h_ivt, 1).flatten(1)  # [B, num_triplets]
        y_ivt = self.classifier(h_ivt_pooled)  # [B, num_triplets]
        
        return {
            'instrument_logits': y_i,
            'verb_logits': y_v,
            'target_logits': y_t,
            'triplet_logits': y_ivt,
            'instrument_cam': h_i,
            'verb_maps': h_v,
            'target_maps': h_t
        }


def test_rendezvous():
    """Test function for inference speed"""
    print("=" * 70)
    print("Testing Rendezvous Model - Inference Speed Test")
    print("=" * 70)
    
    # Model parameters from CholecT50 dataset
    num_instruments = 6
    num_verbs = 10
    num_targets = 15
    num_triplets = 100
    num_decoder_layers = 8
    
    print(f"\nModel Configuration:")
    print(f"  - Instruments: {num_instruments}")
    print(f"  - Verbs: {num_verbs}")
    print(f"  - Targets: {num_targets}")
    print(f"  - Triplets: {num_triplets}")
    print(f"  - Decoder Layers: {num_decoder_layers}")
    
    # Create model
    model = Rendezvous(
        num_instruments=num_instruments,
        num_verbs=num_verbs,
        num_targets=num_targets,
        num_triplets=num_triplets,
        num_decoder_layers=num_decoder_layers,
        pretrained=False
    )
    
    model.eval()
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"\nDevice: {device}")
    
    # Input size from paper (256x448)
    batch_size = 1
    input_tensor = torch.randn(batch_size, 3, 256, 448).to(device)
    
    print(f"Input shape: {input_tensor.shape}")
    
    # Warmup
    print("\nWarming up...")
    with torch.no_grad():
        for _ in range(5):
            _ = model(input_tensor)
    
    # Inference speed test
    print("\nRunning inference speed test...")
    num_iterations = 50
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_iterations):
            outputs = model(input_tensor)
            if device.type == 'cuda':
                torch.cuda.synchronize()
    
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_time = total_time / num_iterations
    fps = 1.0 / avg_time
    
    print("\n" + "=" * 70)
    print("Inference Speed Results:")
    print("=" * 70)
    print(f"Total time for {num_iterations} iterations: {total_time:.3f}s")
    print(f"Average time per frame: {avg_time*1000:.2f}ms")
    print(f"Frames per second (FPS): {fps:.2f}")
    
    # Single forward pass for output shapes
    print("\n" + "=" * 70)
    print("Output Shapes:")
    print("=" * 70)
    with torch.no_grad():
        outputs = model(input_tensor)
    
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"{key:25s}: {list(value.shape)}")
    
    # Model statistics
    print("\n" + "=" * 70)
    print("Model Statistics:")
    print("=" * 70)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Memory usage (if CUDA)
    if device.type == 'cuda':
        print(f"GPU memory allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
        print(f"GPU memory reserved: {torch.cuda.memory_reserved()/1024**2:.2f} MB")
    
    print("\n" + "=" * 70)
    print("✓ Rendezvous inference test completed successfully!")
    print("=" * 70)
    
    return model, outputs


if __name__ == "__main__":
    model, outputs = test_rendezvous()


def test_latency(warmup: int = 5, runs: int = 10, T: int = 16):
    """Latency helper for Rendezvous model."""
    import torch
    import time
    import numpy as np

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        num_instruments = 6
        num_verbs = 10
        num_targets = 15
        num_triplets = 100
        num_decoder_layers = 8

        model = Rendezvous(
            num_instruments=num_instruments,
            num_verbs=num_verbs,
            num_targets=num_targets,
            num_triplets=num_triplets,
            num_decoder_layers=num_decoder_layers,
            pretrained=False,
        ).to(device)
        model.eval()

        B = 1
        input_tensor = torch.randn(B, 3, 256, 448, device=device)

        with torch.no_grad():
            for _ in range(max(1, warmup)):
                _ = model(input_tensor)
            if device.type == 'cuda':
                torch.cuda.synchronize()

        times = []
        with torch.no_grad():
            for _ in range(max(1, runs)):
                start = time.time()
                _ = model(input_tensor)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                end = time.time()
                times.append((end - start) * 1000.0)

        times = np.array(times)
        return {'mean_ms': float(times.mean()), 'std_ms': float(times.std()), 'device': str(device), 'T': int(T)}
    except Exception as e:
        return {'mean_ms': None, 'std_ms': None, 'error': str(e), 'device': str(device), 'T': int(T)}