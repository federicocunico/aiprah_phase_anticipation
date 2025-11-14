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
        cam = self.cam_conv(x)
        logits = F.adaptive_max_pool2d(cam, 1).flatten(1)
        return cam, logits


class TemporalAttentionModule(nn.Module):
    """
    Temporal Attention Module (TAM) for aggregating verb features from past frames.
    Uses attention weights to fuse temporal information.
    """
    def __init__(self, num_verbs=10):
        super(TemporalAttentionModule, self).__init__()
        self.num_verbs = num_verbs
        
        # 1D convolution to generate attention weights across temporal dimension
        self.attention_conv = nn.Conv1d(num_verbs, num_verbs, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(num_verbs)
        
    def forward(self, verb_features):
        """
        Args:
            verb_features: [B, m, C, H, W] where m is clip size
        Returns:
            fused_features: [B, C, H, W] - temporally aggregated features
        """
        B, m, C, H, W = verb_features.shape
        
        # Global average pooling over spatial dimensions
        # [B, m, C, H, W] -> [B, m, C]
        pooled = F.adaptive_avg_pool2d(verb_features.view(B * m, C, H, W), 1)
        pooled = pooled.view(B, m, C)
        
        # Transpose for 1D conv: [B, m, C] -> [B, C, m]
        pooled = pooled.transpose(1, 2)
        
        # Generate attention weights using 1D conv + batch norm + sigmoid
        attention_weights = self.attention_conv(pooled)  # [B, C, m]
        attention_weights = self.bn(attention_weights)
        attention_weights = torch.sigmoid(attention_weights)  # [B, C, m]
        
        # Apply attention weights to verb features
        # Reshape weights: [B, C, m] -> [B, m, C, 1, 1]
        attention_weights = attention_weights.transpose(1, 2).unsqueeze(-1).unsqueeze(-1)
        
        # Weighted sum across temporal dimension
        # [B, m, C, H, W] * [B, m, C, 1, 1] -> sum over m -> [B, C, H, W]
        weighted_features = verb_features * attention_weights
        fused_features = weighted_features.sum(dim=1)  # [B, C, H, W]
        
        return fused_features


class CAGAM(nn.Module):
    """Class Activation Guided Attention Mechanism (position attention for targets)"""
    def __init__(self, in_channels=512, num_instruments=6, num_classes=15):
        super(CAGAM, self).__init__()
        self.num_instruments = num_instruments
        
        # Context mapping
        self.context_conv = nn.Conv2d(in_channels, num_instruments, kernel_size=1)
        
        # Q, K, V projections
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
        context = F.relu(self.context_conv(x))
        
        # Generate Q, K, V
        Q = self.query_conv(context)
        K = self.key_conv(context)
        V = self.value_conv(context)
        
        # CAM features
        cam_resized = F.interpolate(cam, size=(H, W), mode='bilinear', align_corners=False)
        Q_D = self.cam_query(cam_resized)
        K_D = self.cam_key(cam_resized)
        
        # Position attention
        Q_flat = Q.view(B, self.num_instruments, -1).transpose(1, 2)  # [B, HW, C_I]
        K_flat = K.view(B, self.num_instruments, -1).transpose(1, 2)
        P_nd = torch.matmul(Q_flat, K_flat.transpose(1, 2))  # [B, HW, HW]
        
        Q_D_flat = Q_D.view(B, self.num_instruments, -1).transpose(1, 2)
        K_D_flat = K_D.view(B, self.num_instruments, -1).transpose(1, 2)
        P_d = torch.matmul(Q_D_flat, K_D_flat.transpose(1, 2))  # [B, HW, HW]
        
        scale = self.num_instruments ** 0.5
        attention = F.softmax((P_d * P_nd) / scale, dim=-1)
        
        V_flat = V.view(B, self.num_instruments, -1).transpose(1, 2)
        enhancement = torch.matmul(attention, V_flat)
        enhancement = enhancement.transpose(1, 2).view(B, self.num_instruments, H, W)
        
        # Add enhancement
        enhanced = context + self.beta * enhancement
        
        # Output activation maps
        output_maps = self.output_conv(enhanced)
        logits = F.adaptive_avg_pool2d(output_maps, 1).flatten(1)
        
        return output_maps, logits


class CAGTAM(nn.Module):
    """
    Class Activation Guided Temporal Attention Mechanism
    Extends CAGAM with temporal modeling via TAM
    """
    def __init__(self, in_channels=512, num_instruments=6, num_verbs=10, num_tam_layers=2):
        super(CAGTAM, self).__init__()
        self.num_instruments = num_instruments
        self.num_verbs = num_verbs
        
        # Context mapping
        self.context_conv = nn.Conv2d(in_channels, num_instruments, kernel_size=1)
        
        # Q, K, V projections
        self.query_conv = nn.Conv2d(num_instruments, num_instruments, kernel_size=1)
        self.key_conv = nn.Conv2d(num_instruments, num_instruments, kernel_size=1)
        self.value_conv = nn.Conv2d(num_instruments, num_instruments, kernel_size=1)
        
        # For CAM
        self.cam_query = nn.Conv2d(num_instruments, num_instruments, kernel_size=1)
        self.cam_key = nn.Conv2d(num_instruments, num_instruments, kernel_size=1)
        
        # Intermediate output for verb features
        self.verb_conv = nn.Conv2d(num_instruments, num_verbs, kernel_size=1)
        
        self.beta = nn.Parameter(torch.zeros(1))
        
        # TAM layers - stacked for better temporal modeling
        self.tam_layers = nn.ModuleList([
            TemporalAttentionModule(num_verbs) for _ in range(num_tam_layers)
        ])
        self.tam_bns = nn.ModuleList([
            nn.BatchNorm2d(num_verbs) for _ in range(num_tam_layers)
        ])
        
    def forward(self, x_clip, cam_clip):
        """
        Args:
            x_clip: [B, m, C, H, W] - features for clip of m frames
            cam_clip: [B, m, num_instruments, H, W] - instrument CAMs for clip
        Returns:
            verb_maps: [B, num_verbs, H, W]
            verb_logits: [B, num_verbs]
        """
        B, m, C, H, W = x_clip.shape
        
        # Process each frame with CAGAM (channel attention for verbs)
        verb_features_list = []
        
        for t in range(m):
            x_t = x_clip[:, t]  # [B, C, H, W]
            cam_t = cam_clip[:, t]  # [B, num_instruments, H, W]
            
            # Context features
            context = F.relu(self.context_conv(x_t))
            
            # Q, K, V
            Q = self.query_conv(context)
            K = self.key_conv(context)
            V = self.value_conv(context)
            
            # CAM attention
            cam_resized = F.interpolate(cam_t, size=(H, W), mode='bilinear', align_corners=False)
            Q_D = self.cam_query(cam_resized)
            K_D = self.cam_key(cam_resized)
            
            # Channel attention for verbs
            Q_flat = Q.view(B, self.num_instruments, -1)
            K_flat = K.view(B, self.num_instruments, -1)
            P_nd = torch.matmul(Q_flat, K_flat.transpose(1, 2)) / (H * W)
            
            Q_D_flat = Q_D.view(B, self.num_instruments, -1)
            K_D_flat = K_D.view(B, self.num_instruments, -1)
            P_d = torch.matmul(Q_D_flat, K_D_flat.transpose(1, 2)) / (H * W)
            
            scale = self.num_instruments ** 0.5
            attention = F.softmax((P_d * P_nd) / scale, dim=-1)
            
            V_flat = V.view(B, self.num_instruments, -1)
            enhancement = torch.matmul(attention, V_flat)
            enhancement = enhancement.view(B, self.num_instruments, H, W)
            
            # Enhanced features
            enhanced = context + self.beta * enhancement
            
            # Convert to verb features
            verb_feat = self.verb_conv(enhanced)  # [B, num_verbs, H, W]
            verb_features_list.append(verb_feat)
        
        # Stack temporal features: [B, m, num_verbs, H, W]
        verb_features_temporal = torch.stack(verb_features_list, dim=1)
        
        # Apply TAM layers with batch norm and ReLU
        current_features = verb_features_temporal
        for tam, bn in zip(self.tam_layers, self.tam_bns):
            fused = tam(current_features)  # [B, num_verbs, H, W]
            fused = bn(fused)
            fused = F.relu(fused)
            # Update only the current frame (last frame in clip) - avoid in-place
            # Create new tensor instead of in-place assignment
            current_features = torch.cat([
                current_features[:, :-1],
                fused.unsqueeze(1)
            ], dim=1)
        
        # Output is the last frame's verb features
        verb_maps = current_features[:, -1]  # [B, num_verbs, H, W]
        verb_logits = F.adaptive_avg_pool2d(verb_maps, 1).flatten(1)
        
        return verb_maps, verb_logits


class SimpleDecoder(nn.Module):
    """Simplified decoder for triplet classification"""
    def __init__(self, num_instruments=6, num_verbs=10, num_targets=15, num_triplets=100):
        super(SimpleDecoder, self).__init__()
        
        # Bottleneck for global features
        self.bottleneck = nn.Sequential(
            nn.Conv2d(num_instruments + num_verbs + num_targets, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, num_triplets, kernel_size=1)
        )
        
        self.classifier = nn.Linear(num_triplets, num_triplets)
        
    def forward(self, h_i, h_v, h_t):
        # Concatenate all component features
        combined = torch.cat([h_i, h_v, h_t], dim=1)
        
        # Process
        h_ivt = self.bottleneck(combined)
        
        # Pool and classify
        h_ivt_pooled = F.adaptive_avg_pool2d(h_ivt, 1).flatten(1)
        logits = self.classifier(h_ivt_pooled)
        
        return logits


class RiT(nn.Module):
    """
    Rendezvous in Time (RiT)
    Extends Rendezvous with temporal modeling via TAM on verb features
    """
    def __init__(self, num_instruments=6, num_verbs=10, num_targets=15, 
                 num_triplets=100, clip_size=6, num_tam_layers=2, pretrained=True):
        super(RiT, self).__init__()
        
        self.num_instruments = num_instruments
        self.num_verbs = num_verbs
        self.num_targets = num_targets
        self.num_triplets = num_triplets
        self.clip_size = clip_size
        
        # Feature extraction backbone (ResNet-18 with modified strides)
        resnet = models.resnet18(pretrained=pretrained)
        
        # Modify strides for higher resolution
        resnet.layer3[0].conv1.stride = (1, 1)
        resnet.layer3[0].downsample[0].stride = (1, 1)
        resnet.layer4[0].conv1.stride = (1, 1)
        resnet.layer4[0].downsample[0].stride = (1, 1)
        
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        # Encoder modules
        self.wsl = WSL(in_channels=512, num_instruments=num_instruments)
        
        # CAGTAM for temporal verb modeling
        self.cagtam = CAGTAM(in_channels=512, num_instruments=num_instruments, 
                            num_verbs=num_verbs, num_tam_layers=num_tam_layers)
        
        # CAGAM for target (no temporal modeling)
        self.target_cagam = CAGAM(in_channels=512, num_instruments=num_instruments,
                                  num_classes=num_targets)
        
        # Simplified decoder
        self.decoder = SimpleDecoder(num_instruments, num_verbs, num_targets, num_triplets)
        
    def forward(self, x):
        """
        Args:
            x: Input video clip [B, m, 3, H, W] where m is clip_size
            
        Returns:
            dict containing logits for all components
        """
        B, m, C_in, H_in, W_in = x.shape
        
        # Process each frame through backbone
        features_list = []
        for t in range(m):
            feat = self.backbone(x[:, t])  # [B, 512, H, W]
            features_list.append(feat)
        
        # Stack: [B, m, 512, H, W]
        features_clip = torch.stack(features_list, dim=1)
        
        # Instrument detection for all frames
        cam_list = []
        logits_i_list = []
        for t in range(m):
            cam_t, logits_i_t = self.wsl(features_clip[:, t])
            cam_list.append(cam_t)
            logits_i_list.append(logits_i_t)
        
        # Stack CAMs: [B, m, num_instruments, H, W]
        cam_clip = torch.stack(cam_list, dim=1)
        
        # Use instrument from last frame (current frame)
        h_i = cam_list[-1]
        y_i = logits_i_list[-1]
        
        # Verb detection with temporal modeling (CAGTAM)
        h_v, y_v = self.cagtam(features_clip, cam_clip)
        
        # Target detection (no temporal, only current frame)
        h_t, y_t = self.target_cagam(features_clip[:, -1], cam_clip[:, -1])
        
        # Decoder for triplet classification
        y_ivt = self.decoder(h_i, h_v, h_t)
        
        return {
            'instrument_logits': y_i,
            'verb_logits': y_v,
            'target_logits': y_t,
            'triplet_logits': y_ivt,
            'instrument_cam': h_i,
            'verb_maps': h_v,
            'target_maps': h_t
        }


def test_rit():
    """Test function for inference speed"""
    print("=" * 70)
    print("Testing RiT (Rendezvous in Time) Model - Inference Speed Test")
    print("=" * 70)
    
    # Model parameters from CholecT45 dataset
    num_instruments = 6
    num_verbs = 10
    num_targets = 15
    num_triplets = 100
    clip_size = 6
    num_tam_layers = 2
    
    print(f"\nModel Configuration:")
    print(f"  - Instruments: {num_instruments}")
    print(f"  - Verbs: {num_verbs}")
    print(f"  - Targets: {num_targets}")
    print(f"  - Triplets: {num_triplets}")
    print(f"  - Clip Size: {clip_size} frames")
    print(f"  - TAM Layers: {num_tam_layers}")
    
    # Create model
    model = RiT(
        num_instruments=num_instruments,
        num_verbs=num_verbs,
        num_targets=num_targets,
        num_triplets=num_triplets,
        clip_size=clip_size,
        num_tam_layers=num_tam_layers,
        pretrained=False
    )
    
    model.eval()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"\nDevice: {device}")
    
    # Input: video clip (batch=1, clip_size=6, 3 channels, 256x448)
    batch_size = 1
    input_tensor = torch.randn(batch_size, clip_size, 3, 256, 448).to(device)
    
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
    print(f"Average time per clip ({clip_size} frames): {avg_time*1000:.2f}ms")
    print(f"Clips per second: {fps:.2f}")
    print(f"Effective FPS (single frame): {fps * clip_size:.2f}")
    
    # Output shapes
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
    
    # Memory usage
    if device.type == 'cuda':
        print(f"GPU memory allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
        print(f"GPU memory reserved: {torch.cuda.memory_reserved()/1024**2:.2f} MB")
    
    print("\n" + "=" * 70)
    print("✓ RiT inference test completed successfully!")
    print("=" * 70)
    
    return model, outputs


if __name__ == "__main__":
    model, outputs = test_rit()


def test_latency(warmup: int = 5, runs: int = 10, T: int = 16):
    """Latency helper for RiT model."""
    import torch
    import time
    import numpy as np

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        num_instruments = 6
        num_verbs = 10
        num_targets = 15
        num_triplets = 100
        clip_size = max(1, int(T))

        model = RiT(
            num_instruments=num_instruments,
            num_verbs=num_verbs,
            num_targets=num_targets,
            num_triplets=num_triplets,
            clip_size=clip_size,
            num_tam_layers=2,
            pretrained=False,
        ).to(device)
        model.eval()

        B = 1
        input_tensor = torch.randn(B, clip_size, 3, 256, 448, device=device)

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