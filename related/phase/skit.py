import torch
import torch.nn as nn
import time
import numpy as np


class SpatialFeatureExtractor(nn.Module):
    """
    Spatial Feature Extractor based on ViT-B/16.
    
    Uses a 12-head, 12-layer Transformer encoder.
    Input: 248×248 pixels
    Output: 768D representations
    """
    def __init__(self, img_size=248, patch_size=16, in_channels=3, embed_dim=768, 
                 depth=12, num_heads=12, mlp_ratio=4.0):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, 
                                     kernel_size=patch_size, stride=patch_size)
        
        # Class token and position embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # Layer norm
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        """
        Args:
            x: (batch, channels, height, width)
        Returns:
            features: (batch, embed_dim)
        """
        batch_size = x.shape[0]
        
        # Patch embedding: (B, C, H, W) -> (B, embed_dim, H/P, W/P)
        x = self.patch_embed(x)  # (B, 768, 15, 15)
        x = x.flatten(2).transpose(1, 2)  # (B, 225, 768)
        
        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (B, 1, 768)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, 226, 768)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Transformer
        x = self.transformer(x)
        
        # Use class token as output
        x = self.norm(x[:, 0])  # (B, 768)
        
        return x


class LocalTemporalAggregator(nn.Module):
    """
    L-aggregator: Transformer-based local temporal feature aggregator.
    
    Architecture:
    - m-layer self-attention encoder (first branch)
    - n-layer cascaded self-attention + cross-attention decoder (second branch)
    - Window size: λ
    """
    def __init__(self, input_dim=768, output_dim=512, window_size=100, 
                 num_encoder_layers=2, num_decoder_layers=2, num_heads=8):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.window_size = window_size
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, output_dim)
        
        # Encoder (self-attention only)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=output_dim,
            nhead=num_heads,
            dim_feedforward=output_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # Decoder (self-attention + cross-attention)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=output_dim,
            nhead=num_heads,
            dim_feedforward=output_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
    def forward(self, spatial_features):
        """
        Args:
            spatial_features: (batch, seq_len, input_dim) where seq_len = window_size
        Returns:
            local_features: (batch, seq_len, output_dim)
        """
        # Project to output dimension
        x = self.input_proj(spatial_features)  # (B, λ, 512)
        
        # Duplicate into two branches
        branch1 = x  # For decoder
        branch2 = x  # For encoder
        
        # Encoder branch
        memory = self.encoder(branch2)  # (B, λ, 512)
        
        # Decoder branch
        output = self.decoder(branch1, memory)  # (B, λ, 512)
        
        return output


class KeyRecorder(nn.Module):
    """
    Key-recorder: Records global appeared key information using key pooling.
    
    Key pooling operation:
    - g_i^j = max(g_{i-1}^j, k_i^j) for i > 1
    - g_i^j = k_i^j for i = 1
    
    Time complexity: O(1)
    """
    def __init__(self, input_dim=512, key_dim=64):
        super().__init__()
        
        self.input_dim = input_dim
        self.key_dim = key_dim
        
        # Linear layer to embed into key features
        self.key_proj = nn.Linear(input_dim, key_dim)
        
    def forward(self, local_feature, prev_global_key=None):
        """
        Args:
            local_feature: (batch, input_dim) - current local temporal feature
            prev_global_key: (batch, key_dim) - previous global key feature
        Returns:
            global_key: (batch, key_dim) - updated global key feature
        """
        # Embed into key feature
        key_feature = self.key_proj(local_feature)  # (B, key_dim)
        
        # Key pooling: element-wise max
        if prev_global_key is None:
            global_key = key_feature
        else:
            global_key = torch.max(prev_global_key, key_feature)
        
        return global_key
    
    def key_pooling_batch(self, local_features):
        """
        Apply key pooling to a batch of local features (for training).
        
        Args:
            local_features: (batch, seq_len, input_dim)
        Returns:
            global_keys: (batch, seq_len, key_dim)
        """
        batch_size, seq_len, _ = local_features.shape
        
        # Embed all features
        key_features = self.key_proj(local_features)  # (B, T, key_dim)
        
        # Apply cumulative max along sequence dimension
        global_keys = torch.cummax(key_features, dim=1)[0]  # (B, T, key_dim)
        
        return global_keys


class FusionHead(nn.Module):
    """
    Fusion head: Combines local and global features.
    
    Architecture:
    - Linear layer to match dimensions
    - Element-wise addition
    - Residual layer
    """
    def __init__(self, local_dim=512, key_dim=64, num_phases=7):
        super().__init__()
        
        # Project key feature to local dimension
        self.key_proj = nn.Linear(key_dim, local_dim)
        
        # Residual layer
        self.residual = nn.Sequential(
            nn.Linear(local_dim, local_dim),
            nn.ReLU(inplace=True),
            nn.Linear(local_dim, local_dim)
        )
        
        # Output layer for phase prediction
        self.phase_pred = nn.Linear(local_dim, num_phases)
        
    def forward(self, local_feature, global_key):
        """
        Args:
            local_feature: (batch, local_dim)
            global_key: (batch, key_dim)
        Returns:
            phase_pred: (batch, num_phases)
        """
        # Project global key to local dimension
        global_feature = self.key_proj(global_key)  # (B, local_dim)
        
        # Element-wise addition
        fused = local_feature + global_feature  # (B, local_dim)
        
        # Residual layer
        fused = fused + self.residual(fused)  # (B, local_dim)
        
        # Phase prediction
        phase_pred = self.phase_pred(fused)  # (B, num_phases)
        
        return phase_pred


class SKiT(nn.Module):
    """
    SKiT: Fast Key Information Video Transformer for Online Surgical Phase Recognition.
    
    Architecture:
    1. Spatial Feature Extractor (S^R): ViT-B/16
    2. Local temporal feature aggregator (L-aggregator)
    3. Key-recorder: Records global key information
    4. Fusion head: Combines local and global features
    
    Paper: "SKiT: a Fast Key Information Video Transformer for Online Surgical Phase Recognition"
    """
    def __init__(self, 
                 img_size=248,
                 num_phases=7,
                 window_size=100,
                 key_dim=64,
                 local_dim=512,
                 spatial_dim=768):
        super().__init__()
        
        self.img_size = img_size
        self.num_phases = num_phases
        self.window_size = window_size
        self.key_dim = key_dim
        self.local_dim = local_dim
        
        # 1. Spatial Feature Extractor
        self.spatial_extractor = SpatialFeatureExtractor(
            img_size=img_size,
            embed_dim=spatial_dim
        )
        
        # 2. Local temporal feature aggregator
        self.local_aggregator = LocalTemporalAggregator(
            input_dim=spatial_dim,
            output_dim=local_dim,
            window_size=window_size,
            num_encoder_layers=2,
            num_decoder_layers=2
        )
        
        # 3. Key-recorder
        self.key_recorder = KeyRecorder(
            input_dim=local_dim,
            key_dim=key_dim
        )
        
        # 4. Fusion head
        self.fusion_head = FusionHead(
            local_dim=local_dim,
            key_dim=key_dim,
            num_phases=num_phases
        )
        
        # Phase transition map prediction (auxiliary task)
        self.heatmap_pred = nn.Linear(local_dim, 1)
        
    def forward(self, frames, return_features=False):
        """
        Forward pass for training (batch mode).
        
        Args:
            frames: (batch, seq_len, channels, height, width)
        Returns:
            phase_preds: (batch, seq_len, num_phases)
            heatmap_preds: (batch, seq_len, 1)
        """
        batch_size, seq_len, c, h, w = frames.shape
        
        # 1. Extract spatial features for all frames
        frames_flat = frames.view(batch_size * seq_len, c, h, w)
        spatial_features = self.spatial_extractor(frames_flat)  # (B*T, 768)
        spatial_features = spatial_features.view(batch_size, seq_len, -1)  # (B, T, 768)
        
        # 2. Local temporal aggregation with sliding window
        # For training, we process with sliding windows
        local_features_list = []
        
        for i in range(seq_len):
            # Get window
            start_idx = max(0, i - self.window_size + 1)
            window_features = spatial_features[:, start_idx:i+1, :]  # (B, ≤λ, 768)
            
            # Pad if necessary
            if window_features.shape[1] < self.window_size:
                pad_len = self.window_size - window_features.shape[1]
                padding = torch.zeros(batch_size, pad_len, spatial_features.shape[2], 
                                    device=frames.device)
                window_features = torch.cat([padding, window_features], dim=1)
            
            # Apply L-aggregator
            local_feat = self.local_aggregator(window_features)  # (B, λ, 512)
            local_features_list.append(local_feat[:, -1, :])  # Take last timestep
        
        local_features = torch.stack(local_features_list, dim=1)  # (B, T, 512)
        
        # 3. Key recording (global key information)
        global_keys = self.key_recorder.key_pooling_batch(local_features)  # (B, T, key_dim)
        
        # 4. Fusion and prediction
        local_flat = local_features.view(batch_size * seq_len, -1)
        global_flat = global_keys.view(batch_size * seq_len, -1)
        
        phase_preds = self.fusion_head(local_flat, global_flat)  # (B*T, num_phases)
        phase_preds = phase_preds.view(batch_size, seq_len, -1)  # (B, T, num_phases)
        
        # Heatmap prediction
        heatmap_preds = self.heatmap_pred(local_flat)  # (B*T, 1)
        heatmap_preds = heatmap_preds.view(batch_size, seq_len, -1)  # (B, T, 1)
        
        if return_features:
            return phase_preds, heatmap_preds, local_features, global_keys
        
        return phase_preds, heatmap_preds
    
    def forward_online(self, frame, prev_local_window, prev_global_key):
        """
        Forward pass for online inference (frame-by-frame).
        
        Args:
            frame: (batch, channels, height, width) - current frame
            prev_local_window: (batch, window_size-1, spatial_dim) - previous window
            prev_global_key: (batch, key_dim) - previous global key
        Returns:
            phase_pred: (batch, num_phases)
            heatmap_pred: (batch, 1)
            updated_window: (batch, window_size-1, spatial_dim)
            updated_global_key: (batch, key_dim)
        """
        batch_size = frame.shape[0]
        
        # 1. Extract spatial feature
        spatial_feature = self.spatial_extractor(frame)  # (B, 768)
        spatial_feature = spatial_feature.unsqueeze(1)  # (B, 1, 768)
        
        # 2. Update window
        if prev_local_window is None:
            # Initialize window with zeros
            window = torch.zeros(batch_size, self.window_size, spatial_feature.shape[2],
                               device=frame.device)
            window[:, -1, :] = spatial_feature.squeeze(1)
        else:
            # Slide window: remove oldest, add newest
            window = torch.cat([prev_local_window, spatial_feature], dim=1)  # (B, λ, 768)
        
        # 3. Local temporal aggregation
        local_feature = self.local_aggregator(window)  # (B, λ, 512)
        current_local = local_feature[:, -1, :]  # (B, 512) - take last timestep
        
        # 4. Key recording
        current_global_key = self.key_recorder(current_local, prev_global_key)  # (B, key_dim)
        
        # 5. Fusion and prediction
        phase_pred = self.fusion_head(current_local, current_global_key)  # (B, num_phases)
        heatmap_pred = self.heatmap_pred(current_local)  # (B, 1)
        
        # 6. Update window for next frame (keep only last window_size-1 frames)
        updated_window = window[:, 1:, :]  # (B, λ-1, 768)
        
        return phase_pred, heatmap_pred, updated_window, current_global_key


def fps():
    """
    Measure execution time and FPS for the SKiT model.
    
    Performs:
    - 5 warmup rounds
    - 20 measurement runs
    - Returns timing statistics in milliseconds
    
    Returns:
        dict: Dictionary containing timing statistics
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model
    model = SKiT(
        img_size=248,
        num_phases=7,
        window_size=100,
        key_dim=64,
        local_dim=512,
        spatial_dim=768
    )
    model = model.to(device)
    model.eval()
    
    # For FPS testing use a batch-mode sequence of length 16 frames
    batch_size = 1
    seq_len = 16
    dummy_frames = torch.randn(batch_size, seq_len, 3, 248, 248).to(device)

    # Warmup rounds (batch forward)
    print("Performing 5 warmup rounds (batch)...")
    with torch.no_grad():
        for i in range(5):
            phase_pred, heatmap_pred = model(dummy_frames)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

    # Measurement runs
    print("Performing 20 measurement runs (batch)...")
    times = []
    num_runs = 20

    with torch.no_grad():
        for i in range(num_runs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            start = time.perf_counter()
            phase_pred, heatmap_pred = model(dummy_frames)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            end = time.perf_counter()
            elapsed_ms = (end - start) * 1000
            times.append(elapsed_ms)
    
    times = np.array(times)
    
    # Calculate statistics
    results = {
        'model_name': 'skit',
        'total_ms': float(np.mean(times)),
        'std_ms': float(np.std(times)),
        'min_ms': float(np.min(times)),
        'max_ms': float(np.max(times)),
        'median_ms': float(np.median(times)),
        'fps': 1000.0 / float(np.mean(times)),
        'num_runs': num_runs,
        'num_warmup': 5,
    'input_shape': list(dummy_frames.shape),
        'output_phase_shape': list(phase_pred.shape),
        'output_heatmap_shape': list(heatmap_pred.shape),
        'device': str(device),
        'num_parameters': sum(p.numel() for p in model.parameters()),
        'num_trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
    'inference_mode': 'batch',
        'time_complexity_key_pooling': 'O(1)'
    }
    
    return results


def main():
    """Test the model and run benchmarks."""
    print("=" * 70)
    print("SKiT - Fast Key Information Video Transformer")
    print("=" * 70)
    
    # Create model
    model = SKiT(
        img_size=248,
        num_phases=7,
        window_size=100,
        key_dim=64,
        local_dim=512,
        spatial_dim=768
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Test batch forward pass (training mode)
    print(f"\nTesting batch forward pass (training)...")
    dummy_batch = torch.randn(2, 10, 3, 248, 248)  # (batch=2, seq_len=10, C, H, W)
    
    model.eval()
    with torch.no_grad():
        phase_preds, heatmap_preds = model(dummy_batch)
    
    print(f"  Input shape: {list(dummy_batch.shape)}")
    print(f"  Phase predictions shape: {list(phase_preds.shape)}")
    print(f"  Heatmap predictions shape: {list(heatmap_preds.shape)}")
    
    # Test online forward pass (inference mode)
    print(f"\nTesting online forward pass (inference)...")
    dummy_frame = torch.randn(1, 3, 248, 248)
    
    prev_window = None
    prev_global_key = None
    
    with torch.no_grad():
        phase_pred, heatmap_pred, prev_window, prev_global_key = model.forward_online(
            dummy_frame, prev_window, prev_global_key
        )
    
    print(f"  Input shape: {list(dummy_frame.shape)}")
    print(f"  Phase prediction shape: {list(phase_pred.shape)}")
    print(f"  Heatmap prediction shape: {list(heatmap_pred.shape)}")
    print(f"  Updated window shape: {list(prev_window.shape)}")
    print(f"  Updated global key shape: {list(prev_global_key.shape)}")
    
    # Run FPS benchmark
    print(f"\n{'=' * 70}")
    print("Running Performance Benchmark")
    print(f"{'=' * 70}")
    
    results = fps()
    
    print(f"\nBenchmark Results:")
    print(f"  Model: {results['model_name']}")
    print(f"  Device: {results['device']}")
    print(f"  Input shape: {results['input_shape']}")
    print(f"  Inference mode: {results['inference_mode']}")
    print(f"\nTiming Statistics ({results['num_runs']} runs):")
    print(f"  Average: {results['total_ms']:.2f} ms ({results['fps']:.1f} FPS)")
    print(f"  Std Dev: {results['std_ms']:.2f} ms")
    print(f"  Median:  {results['median_ms']:.2f} ms")
    print(f"  Min:     {results['min_ms']:.2f} ms")
    print(f"  Max:     {results['max_ms']:.2f} ms")
    print(f"\nParameters:")
    print(f"  Total: {results['num_parameters']:,}")
    print(f"  Trainable: {results['num_trainable_parameters']:,}")
    print(f"\nKey Pooling Time Complexity: {results['time_complexity_key_pooling']}")
    print(f"  (Inference time is constant, independent of video length)")
    
    print(f"\n{'=' * 70}")


if __name__ == "__main__":
    main()


def test_latency(warmup: int = 5, runs: int = 10, T: int = 16):
    """
    Standardized test helper for SKiT: run warmup then timed runs with temporal length T.
    Returns dict with mean_ms and std_ms.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SKiT(img_size=248, num_phases=7, window_size=100, key_dim=64, local_dim=512, spatial_dim=768).to(device)
    model.eval()

    batch_size = 1
    seq_len = T
    dummy_frames = torch.randn(batch_size, seq_len, 3, 248, 248).to(device)

    with torch.no_grad():
        for _ in range(warmup):
            phase_pred, heatmap_pred = model(dummy_frames)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(runs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            phase_pred, heatmap_pred = model(dummy_frames)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000.0)

    arr = np.array(times)
    return {"mean_ms": float(np.mean(arr)), "std_ms": float(np.std(arr)), "device": str(device), "T": T}

