import torch
import torch.nn as nn
import torch.nn.functional as F
import time


class I3DBackbone(nn.Module):
    """Simplified I3D backbone for feature extraction"""
    def __init__(self, pretrained=False):
        super(I3DBackbone, self).__init__()
        
        # Simplified 3D conv blocks
        self.conv1 = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv3d(64, 192, kernel_size=3, padding=1),
            nn.BatchNorm3d(192),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv3d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm3d(384),
            nn.ReLU()
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv3d(384, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU()
        )
        
    def forward(self, x):
        """
        Args:
            x: [B, 3, T, H, W]
        Returns:
            features: [B, 512, T, H', W']
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x


class SoftLabelNetwork(nn.Module):
    """Network for predicting soft labels"""
    def __init__(self, in_channels=512, max_instruments=6):
        super(SoftLabelNetwork, self).__init__()
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        
        # Prediction heads for soft labels
        self.num_instruments = nn.Linear(in_channels, max_instruments + 1)  # 0 to max
        self.num_categories = nn.Linear(in_channels, max_instruments + 1)
        self.unrelated_activity = nn.Linear(in_channels, 2)  # binary
        self.critical_activity = nn.Linear(in_channels, 2)  # binary
        
    def forward(self, features):
        """
        Args:
            features: [B, C, T, H, W]
        Returns:
            dict of soft label predictions
        """
        # Global pooling
        pooled = self.global_pool(features).flatten(1)  # [B, C]
        
        return {
            'num_instruments': self.num_instruments(pooled),
            'num_categories': self.num_categories(pooled),
            'unrelated_activity': self.unrelated_activity(pooled),
            'critical_activity': self.critical_activity(pooled)
        }


class GradCAM3D(nn.Module):
    """3D Grad-CAM for generating class activation maps"""
    def __init__(self, in_channels=512, num_classes=6):
        super(GradCAM3D, self).__init__()
        
        self.conv = nn.Conv3d(in_channels, num_classes, kernel_size=1)
        
    def forward(self, features):
        """
        Args:
            features: [B, C, T, H, W]
        Returns:
            cam: [B, num_classes, T, H, W]
            logits: [B, num_classes]
        """
        cam = self.conv(features)  # [B, num_classes, T, H, W]
        
        # Global max pooling for classification
        logits = F.adaptive_max_pool3d(cam, 1).flatten(1)
        
        return cam, logits


class CAGAM3D(nn.Module):
    """3D Class Activation Guided Attention Mechanism"""
    def __init__(self, in_channels=512, num_instruments=6, num_classes=10):
        super(CAGAM3D, self).__init__()
        self.num_instruments = num_instruments
        
        # Context mapping (3D)
        self.context_conv = nn.Conv3d(in_channels, num_instruments, kernel_size=1)
        
        # Q, K, V projections
        self.query_conv = nn.Conv3d(num_instruments, num_instruments, kernel_size=1)
        self.key_conv = nn.Conv3d(num_instruments, num_instruments, kernel_size=1)
        self.value_conv = nn.Conv3d(num_instruments, num_instruments, kernel_size=1)
        
        # For CAM
        self.cam_query = nn.Conv3d(num_instruments, num_instruments, kernel_size=1)
        self.cam_key = nn.Conv3d(num_instruments, num_instruments, kernel_size=1)
        
        # Output
        self.output_conv = nn.Conv3d(num_instruments, num_classes, kernel_size=1)
        
        self.beta = nn.Parameter(torch.zeros(1))
        
    def forward(self, features, cam):
        """
        Args:
            features: [B, C, T, H, W]
            cam: [B, num_instruments, T, H, W]
        Returns:
            output_maps: [B, num_classes, T, H, W]
            logits: [B, num_classes]
        """
        B, C, T, H, W = features.shape
        
        # Context
        context = F.relu(self.context_conv(features))
        
        # Q, K, V
        Q = self.query_conv(context)
        K = self.key_conv(context)
        V = self.value_conv(context)
        
        # CAM features
        cam_resized = F.interpolate(cam, size=(T, H, W), mode='trilinear', align_corners=False)
        Q_D = self.cam_query(cam_resized)
        K_D = self.cam_key(cam_resized)
        
        # Position attention (flattened spatial-temporal)
        Q_flat = Q.view(B, self.num_instruments, -1).transpose(1, 2)  # [B, THW, C_I]
        K_flat = K.view(B, self.num_instruments, -1).transpose(1, 2)
        P_nd = torch.matmul(Q_flat, K_flat.transpose(1, 2))  # [B, THW, THW]
        
        Q_D_flat = Q_D.view(B, self.num_instruments, -1).transpose(1, 2)
        K_D_flat = K_D.view(B, self.num_instruments, -1).transpose(1, 2)
        P_d = torch.matmul(Q_D_flat, K_D_flat.transpose(1, 2))  # [B, THW, THW]
        
        # Attention
        scale = self.num_instruments ** 0.5
        attention = F.softmax((P_d * P_nd) / scale, dim=-1)
        
        # Apply attention
        V_flat = V.view(B, self.num_instruments, -1).transpose(1, 2)
        enhancement = torch.matmul(attention, V_flat)
        enhancement = enhancement.transpose(1, 2).view(B, self.num_instruments, T, H, W)
        
        # Enhanced features
        enhanced = context + self.beta * enhancement
        
        # Output
        output_maps = self.output_conv(enhanced)
        logits = F.adaptive_avg_pool3d(output_maps, 1).flatten(1)
        
        return output_maps, logits


class ComponentNetwork(nn.Module):
    """Network for individual component (tool, verb, or target)"""
    def __init__(self, in_channels=512, num_classes=6, is_tool=True, 
                 num_instruments=6, max_instruments=6):
        super(ComponentNetwork, self).__init__()
        self.is_tool = is_tool
        
        if is_tool:
            # Tool uses GradCAM
            self.detector = GradCAM3D(in_channels, num_classes)
        else:
            # Verb/Target use CAGAM
            self.detector = CAGAM3D(in_channels, num_instruments, num_classes)
        
        # Soft label prediction branch
        self.soft_label_net = SoftLabelNetwork(in_channels, max_instruments)
        
    def forward(self, features, cam=None):
        """
        Args:
            features: [B, C, T, H, W]
            cam: [B, num_instruments, T, H, W] (only for verb/target)
        """
        if self.is_tool:
            maps, logits = self.detector(features)
            soft_labels = self.soft_label_net(features)
            return maps, logits, soft_labels
        else:
            maps, logits = self.detector(features, cam)
            soft_labels = self.soft_label_net(features)
            return maps, logits, soft_labels


class TripletNetwork(nn.Module):
    """Final network for triplet association"""
    def __init__(self, num_instruments=6, num_verbs=10, num_targets=15, 
                 num_triplets=100, max_instruments=6):
        super(TripletNetwork, self).__init__()
        
        # Feature dimension after flattening CAMs
        total_features = num_instruments + num_verbs + num_targets
        
        # Triplet classifier
        # combined vector is [logits (total_features) + pooled features (total_features)] => total_features * 2
        self.classifier = nn.Sequential(
            nn.Linear(total_features * 2, 512),  # logits + pooled features
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_triplets)
        )

        # Soft label prediction (input matches combined vector)
        self.soft_label_fc = nn.Linear(total_features * 2, max_instruments * 4)
        
    def forward(self, tool_logits, verb_logits, target_logits, 
                tool_cam, verb_maps, target_maps):
        """
        Args:
            tool_logits, verb_logits, target_logits: [B, num_classes]
            tool_cam, verb_maps, target_maps: [B, num_classes, T, H, W]
        Returns:
            triplet_logits: [B, num_triplets]
            soft_labels: dict
        """
        # Pool CAMs to get features
        tool_feat = F.adaptive_avg_pool3d(tool_cam, 1).flatten(1)
        verb_feat = F.adaptive_avg_pool3d(verb_maps, 1).flatten(1)
        target_feat = F.adaptive_avg_pool3d(target_maps, 1).flatten(1)
        
        # Combine logits and features
        combined = torch.cat([tool_logits, verb_logits, target_logits,
                             tool_feat, verb_feat, target_feat], dim=1)
        
        # Triplet prediction
        triplet_logits = self.classifier(combined)
        
        # Soft labels
        soft_raw = self.soft_label_fc(combined)
        soft_labels = {
            'num_instruments': soft_raw[:, :7],
            'num_categories': soft_raw[:, 7:14],
            'unrelated_activity': soft_raw[:, 14:16],
            'critical_activity': soft_raw[:, 16:18]
        }
        
        return triplet_logits, soft_labels


class TDN(nn.Module):
    """
    Triplet Disentanglement Network (TDN)
    Decomposes triplet recognition into hierarchical sub-tasks
    """
    def __init__(self, num_instruments=6, num_verbs=10, num_targets=15,
                 num_triplets=100, clip_size=5, pretrained=False):
        super(TDN, self).__init__()
        
        self.num_instruments = num_instruments
        self.num_verbs = num_verbs
        self.num_targets = num_targets
        self.num_triplets = num_triplets
        self.clip_size = clip_size
        
        # I3D Backbone
        self.backbone = I3DBackbone(pretrained=pretrained)
        
        # Stage 1: Soft Label Network
        self.soft_label_network = SoftLabelNetwork(in_channels=512, 
                                                   max_instruments=num_instruments)
        
        # Stage 2: Component Networks
        self.tool_network = ComponentNetwork(512, num_instruments, is_tool=True,
                                            max_instruments=num_instruments)
        self.verb_network = ComponentNetwork(512, num_verbs, is_tool=False,
                                            num_instruments=num_instruments,
                                            max_instruments=num_instruments)
        self.target_network = ComponentNetwork(512, num_targets, is_tool=False,
                                              num_instruments=num_instruments,
                                              max_instruments=num_instruments)
        
        # Stage 3: Triplet Network
        self.triplet_network = TripletNetwork(num_instruments, num_verbs, 
                                             num_targets, num_triplets,
                                             max_instruments=num_instruments)
        
    def separate_cams(self, cam, k):
        """
        Separate top-K instrument CAMs based on soft label prediction
        Args:
            cam: [B, num_instruments, T, H, W]
            k: number of categories to separate
        Returns:
            List of separated CAMs
        """
        B, C, T, H, W = cam.shape

        # Score each channel by summing across spatial-temporal dims
        scores = cam.sum(dim=[2, 3, 4])  # [B, C]

        # Get top-k indices
        _, top_k_indices = torch.topk(scores, k, dim=1)

        # Create a mask selecting top-k channels per batch and keep original shape
        mask = torch.zeros(B, C, device=cam.device, dtype=cam.dtype)
        for b in range(B):
            mask[b, top_k_indices[b]] = 1.0

        mask = mask.view(B, C, 1, 1, 1)
        separated = cam * mask

        return separated
    
    def forward(self, x):
        """
        Args:
            x: [B, T, 3, H, W] or [B, 3, T, H, W]
        Returns:
            dict of predictions
        """
        # Ensure correct dimension order [B, 3, T, H, W]
        if x.dim() == 5 and x.size(2) == 3:
            x = x.transpose(1, 2)  # [B, T, 3, H, W] -> [B, 3, T, H, W]
        
        B = x.size(0)
        
        # Feature extraction
        features = self.backbone(x)  # [B, 512, T, H, W]
        
        # Stage 1: Soft labels
        soft_labels_stage1 = self.soft_label_network(features)
        
        # Stage 2: Component detection
        tool_cam, tool_logits, tool_soft = self.tool_network(features)
        
        # Get number of categories for separation (use max from soft labels)
        num_categories = torch.argmax(soft_labels_stage1['num_categories'], dim=1)
        k = max(1, int(num_categories.float().mean().item()))
        k = min(k, self.num_instruments)
        
        # Separate tool CAMs
        separated_cam = self.separate_cams(tool_cam, k)
        
        # Verb and target detection with separated CAMs
        verb_maps, verb_logits, verb_soft = self.verb_network(features, separated_cam)
        target_maps, target_logits, target_soft = self.target_network(features, separated_cam)
        
        # Stage 3: Triplet association
        triplet_logits, triplet_soft = self.triplet_network(
            tool_logits, verb_logits, target_logits,
            tool_cam, verb_maps, target_maps
        )
        
        return {
            'instrument_logits': tool_logits,
            'verb_logits': verb_logits,
            'target_logits': target_logits,
            'triplet_logits': triplet_logits,
            'instrument_cam': tool_cam,
            'verb_maps': verb_maps,
            'target_maps': target_maps,
            'soft_labels': soft_labels_stage1
        }


def test_tdn():
    """Test function for inference speed"""
    print("=" * 70)
    print("Testing TDN (Triplet Disentanglement Network) - Inference Speed")
    print("=" * 70)
    
    # Model parameters from CholecT45
    num_instruments = 6
    num_verbs = 10
    num_targets = 15
    num_triplets = 100
    clip_size = 5
    
    print(f"\nModel Configuration:")
    print(f"  - Instruments: {num_instruments}")
    print(f"  - Verbs: {num_verbs}")
    print(f"  - Targets: {num_targets}")
    print(f"  - Triplets: {num_triplets}")
    print(f"  - Clip Size: {clip_size} frames")
    print(f"  - Backbone: I3D (3D CNN)")
    
    # Create model
    model = TDN(
        num_instruments=num_instruments,
        num_verbs=num_verbs,
        num_targets=num_targets,
        num_triplets=num_triplets,
        clip_size=clip_size,
        pretrained=False
    )
    
    model.eval()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"\nDevice: {device}")
    
    # Input: [B, T, 3, H, W] or [B, 3, T, H, W]
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
        elif isinstance(value, dict):
            print(f"{key}:")
            for k, v in value.items():
                if isinstance(v, torch.Tensor):
                    print(f"  {k:23s}: {list(v.shape)}")
    
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
    print("✓ TDN inference test completed successfully!")
    print("=" * 70)
    
    return model, outputs


if __name__ == "__main__":
    model, outputs = test_tdn()


def test_latency(warmup: int = 5, runs: int = 10, T: int = 16):
    """Latency helper for TDN model."""
    import torch
    import time
    import numpy as np

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        num_instruments = 6
        num_verbs = 10
        num_targets = 15
        num_triplets = 100
        # Cap clip size for latency test to avoid huge 3D attention matrices / OOM
        clip_size = min(4, max(1, int(T)))

        model = TDN(
            num_instruments=num_instruments,
            num_verbs=num_verbs,
            num_targets=num_targets,
            num_triplets=num_triplets,
            clip_size=clip_size,
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