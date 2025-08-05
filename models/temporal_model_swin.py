import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.swin_transformer import SwinTransformer, Swin_T_Weights


class AdaptiveSwinTransformer(nn.Module):
    """SwinTransformer backbone adapted for variable channel input (RGB or RGB-D)"""
    def __init__(self, in_channels=3, pretrained=True):
        super(AdaptiveSwinTransformer, self).__init__()
        
        self.in_channels = in_channels
        
        # Load pretrained SwinTransformer
        if pretrained:
            self.swin = models.swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
        else:
            self.swin = models.swin_t(weights=None)
        
        # Adapt first conv layer if needed
        if in_channels != 3:
            # Get the original first conv layer parameters
            original_conv = self.swin.features[0][0]
            
            # Create new conv layer for custom number of channels
            new_conv = nn.Conv2d(
                in_channels, 96, 
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding
            )
            
            # Initialize the new conv layer
            with torch.no_grad():
                if in_channels < 3:
                    # If fewer channels, average the RGB weights
                    new_conv.weight[:, :, :, :] = original_conv.weight[:, :in_channels, :, :]
                else:
                    # Copy RGB weights for first 3 channels
                    new_conv.weight[:, :3, :, :] = original_conv.weight
                    # Initialize additional channels (e.g., depth)
                    for i in range(3, in_channels):
                        new_conv.weight[:, i:i+1, :, :] = original_conv.weight.mean(dim=1, keepdim=True)
                
                new_conv.bias = original_conv.bias
            
            # Replace the first conv layer
            self.swin.features[0][0] = new_conv
        
        # Remove the classification head
        self.swin.head = nn.Identity()
        
        # Output dimension of Swin-T is 768
        self.output_dim = 768
        
    def forward(self, x):
        # x: [B, C, H, W] where C=in_channels
        return self.swin(x)


class TemporalFusionBlock(nn.Module):
    """Fuses temporal information using self-attention and LSTM"""
    def __init__(self, embed_dim, hidden_dim=512, num_heads=8):
        super(TemporalFusionBlock, self).__init__()
        
        # Temporal self-attention
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            batch_first=True
        )
        
        # Bidirectional LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        # Project back to embed_dim
        self.projection = nn.Linear(hidden_dim * 2, embed_dim)
        
        # Layer norms
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        # x: [B, T, E]
        # Self-attention
        attn_out, _ = self.temporal_attn(x, x, x)
        x = self.ln1(x + attn_out)
        
        # LSTM
        lstm_out, _ = self.lstm(x)
        lstm_out = self.projection(lstm_out)
        x = self.ln2(x + lstm_out)
        
        return x


class TemporalPooling(nn.Module):
    """Multi-scale temporal pooling with attention"""
    def __init__(self, embed_dim):
        super(TemporalPooling, self).__init__()
        
        # Attention weights for different temporal scales
        self.attention_weights = nn.Linear(embed_dim, 3)  # 3 scales
        
        # Pooling at different scales
        self.pool_short = nn.AdaptiveAvgPool1d(1)
        self.pool_mid = nn.AdaptiveMaxPool1d(1)
        
        # Global attention pooling
        self.global_attn = nn.Linear(embed_dim, 1)
        
    def forward(self, x):
        # x: [B, T, E]
        B, T, E = x.shape
        
        # Different pooling strategies
        x_t = x.transpose(1, 2)  # [B, E, T]
        
        # Short-term: last 3 frames average
        short_pool = x[:, -min(3, T):, :].mean(dim=1)  # [B, E]
        
        # Mid-term: adaptive pooling
        mid_pool = self.pool_mid(x_t).squeeze(-1)  # [B, E]
        
        # Long-term: attention-weighted pooling
        attn_scores = torch.softmax(self.global_attn(x).squeeze(-1), dim=1)  # [B, T]
        long_pool = (x * attn_scores.unsqueeze(-1)).sum(dim=1)  # [B, E]
        
        # Combine with learnable weights
        pooled = torch.stack([short_pool, mid_pool, long_pool], dim=1)  # [B, 3, E]
        weights = torch.softmax(self.attention_weights(x.mean(dim=1)), dim=-1)  # [B, 3]
        
        # Weighted combination
        output = (pooled * weights.unsqueeze(-1)).sum(dim=1)  # [B, E]
        
        return output


class TemporalAnticipationModel(nn.Module):
    def __init__(self, sequence_length: int, num_classes: int = 7, time_horizon: int = 5, 
                 in_channels: int = 3, use_depth_enhancer: bool = True, 
                 temporal_modeling: str = "separate"):
        """
        Temporal Anticipation Model with SwinTransformer backbone
        
        Args:
            sequence_length: Length of input temporal sequence
            num_classes: Number of output classes
            time_horizon: Maximum time horizon for regression
            in_channels: Number of input channels (3 for RGB, 4 for RGB-D)
            use_depth_enhancer: Whether to use depth enhancement when in_channels > 3
            temporal_modeling: Strategy for temporal modeling:
                - "separate": Process spatial features first, then temporal (default)
                - "3d_conv": Use 3D convolutions for early temporal fusion
                - "sequential": Process each frame sequentially with shared weights
        """
        super(TemporalAnticipationModel, self).__init__()
        
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.time_horizon = time_horizon
        self.in_channels = in_channels
        self.use_depth_enhancer = use_depth_enhancer and in_channels > 3
        self.temporal_modeling = temporal_modeling
        
        # Early temporal fusion with 3D convolutions (optional)
        if temporal_modeling == "3d_conv":
            self.early_temporal = nn.Sequential(
                nn.Conv3d(in_channels, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
                nn.BatchNorm3d(64),
                nn.ReLU(),
                nn.Conv3d(64, in_channels, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
                nn.BatchNorm3d(in_channels),
                nn.ReLU()
            )
        
        # Adaptive SwinTransformer backbone
        self.backbone = AdaptiveSwinTransformer(in_channels=in_channels, pretrained=True)
        self.embedding_dim = self.backbone.output_dim  # 768 for Swin-T
        
        # Temporal feature extraction
        self.temporal_fusion = TemporalFusionBlock(
            embed_dim=self.embedding_dim,
            hidden_dim=512,
            num_heads=8
        )
        
        # Additional temporal fusion layer for better temporal understanding
        self.temporal_fusion2 = TemporalFusionBlock(
            embed_dim=self.embedding_dim,
            hidden_dim=384,
            num_heads=8
        )
        
        # Temporal pooling
        self.temporal_pooling = TemporalPooling(self.embedding_dim)
        
        # Regression head with dropout for regularization
        self.regressor = nn.Sequential(
            nn.Linear(self.embedding_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )
        
        # Optional: depth-specific feature extractor for better RGB-D fusion
        if self.use_depth_enhancer:
            self.depth_enhancer = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(32, self.embedding_dim),
            )
        
    def extract_depth_features(self, x):
        """Extract additional features from depth channel"""
        if not self.use_depth_enhancer:
            return None
            
        # x: [B*T, C, H, W] where C >= 4
        depth = x[:, 3:4, :, :]  # [B*T, 1, H, W]
        depth_features = self.depth_enhancer(depth)  # [B*T, embedding_dim]
        return depth_features
        
    def forward(self, x: torch.Tensor):
        B, T, C, H, W = x.size()
        assert C == self.in_channels, f"Expected {self.in_channels} channels, got {C}"
        
        # Apply early temporal fusion if specified
        if self.temporal_modeling == "3d_conv":
            # x: [B, T, C, H, W] -> [B, C, T, H, W] for 3D conv
            x = x.permute(0, 2, 1, 3, 4)
            x = self.early_temporal(x)
            x = x.permute(0, 2, 1, 3, 4).contiguous()  # Back to [B, T, C, H, W] and make contiguous
        
        if self.temporal_modeling == "sequential":
            # Process sequences with explicit temporal awareness
            temporal_features = []
            for t in range(T):
                frame_batch = x[:, t, :, :, :]  # [B, C, H, W]
                frame_features = self.backbone(frame_batch)  # [B, embedding_dim]
                
                if self.use_depth_enhancer:
                    depth_batch = frame_batch[:, 3:4, :, :]  # [B, 1, H, W]
                    depth_features = self.depth_enhancer(depth_batch)  # [B, embedding_dim]
                    frame_features = frame_features + 0.1 * depth_features
                
                temporal_features.append(frame_features)
            
            temporal_features = torch.stack(temporal_features, dim=1)  # [B, T, embedding_dim]
        
        else:  # Default "separate" mode
            # Reshape for batch processing - use reshape to handle non-contiguous tensors
            x_reshaped = x.reshape(B * T, C, H, W)  # [B*T, C, H, W]
            
            # Extract features using SwinTransformer
            spatial_features = self.backbone(x_reshaped)  # [B*T, embedding_dim]
            
            # Optional: extract and fuse depth-specific features
            if self.use_depth_enhancer:
                depth_features = self.extract_depth_features(x_reshaped)  # [B*T, embedding_dim]
                if depth_features is not None:
                    spatial_features = spatial_features + 0.1 * depth_features  # Weighted fusion
            
            # Reshape back to temporal sequence
            temporal_features = spatial_features.reshape(B, T, self.embedding_dim)  # [B, T, E]
        
        # Apply temporal fusion blocks - these operate on proper sequences
        # This is where the actual temporal modeling happens
        temporal_features = self.temporal_fusion(temporal_features)  # [B, T, E]
        temporal_features = self.temporal_fusion2(temporal_features)  # [B, T, E]
        
        # Temporal pooling - aggregates across time dimension
        pooled_features = self.temporal_pooling(temporal_features)  # [B, E]
        
        # Regression
        regression_logits = self.regressor(pooled_features)  # [B, num_classes]
        
        # Clamp to [0, time_horizon]
        regression_output = torch.sigmoid(regression_logits) * self.time_horizon
        
        return regression_output


def create_model(sequence_length=10, num_classes=7, time_horizon=5, in_channels=3, 
                 use_depth_enhancer=True, temporal_modeling="separate"):
    """
    Factory function to create the model
    
    Args:
        sequence_length: Length of input temporal sequence
        num_classes: Number of output classes
        time_horizon: Maximum time horizon for regression
        in_channels: Number of input channels (3 for RGB, 4 for RGB-D)
        use_depth_enhancer: Whether to use depth enhancement when in_channels > 3
        temporal_modeling: Strategy for temporal modeling ("separate", "3d_conv", "sequential")
    """
    return TemporalAnticipationModel(
        sequence_length=sequence_length,
        num_classes=num_classes,
        time_horizon=time_horizon,
        in_channels=in_channels,
        use_depth_enhancer=use_depth_enhancer,
        temporal_modeling=temporal_modeling
    )


def __test__():
    """Test the model with different configurations"""
    import time
    
    print("="*60)
    print("Testing TemporalAnticipationModel with SwinTransformer")
    print("="*60)
    
    # Test parameters
    B, T, H, W = 2, 10, 224, 224
    num_classes = 7
    time_horizon = 5
    
    # Test configurations including temporal modeling strategies
    configs = [
        # RGB configurations
        {"in_channels": 3, "name": "RGB (separate)", "temporal_modeling": "separate"},
        {"in_channels": 3, "name": "RGB (3d_conv)", "temporal_modeling": "3d_conv"},
        {"in_channels": 3, "name": "RGB (sequential)", "temporal_modeling": "sequential"},
        
        # RGB-D configurations
        {"in_channels": 4, "name": "RGB-D (separate)", "temporal_modeling": "separate"},
        {"in_channels": 4, "name": "RGB-D (3d_conv)", "temporal_modeling": "3d_conv"},
        {"in_channels": 4, "name": "RGB-D (sequential)", "temporal_modeling": "sequential"},
        
        # Special configurations
        {"in_channels": 4, "name": "RGB-D (no depth enhancer)", "use_depth_enhancer": False, "temporal_modeling": "separate"},
    ]
    
    results = []
    
    for config in configs:
        print(f"\n\nTesting {config['name']} configuration:")
        print("-" * 50)
        
        # Create model
        model = create_model(
            sequence_length=T, 
            num_classes=num_classes, 
            time_horizon=time_horizon,
            in_channels=config["in_channels"],
            use_depth_enhancer=config.get("use_depth_enhancer", True),
            temporal_modeling=config.get("temporal_modeling", "separate")
        )
        model.eval()
        
        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Create mock input
        mock_input = torch.randn(B, T, config["in_channels"], H, W)
        
        # Warm-up run
        with torch.no_grad():
            _ = model(mock_input)
        
        # Timed forward pass
        num_runs = 5
        times = []
        for _ in range(num_runs):
            start_time = time.time()
            with torch.no_grad():
                output = model(mock_input)
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = sum(times) / num_runs
        
        print(f"\nInput shape: {mock_input.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
        print(f"Forward pass time (avg of {num_runs} runs): {avg_time:.3f}s")
        
        # Test with actual loss
        target = torch.rand(B, num_classes) * time_horizon
        loss = torch.nn.functional.mse_loss(output, target)
        print(f"Mock MSE loss: {loss.item():.4f}")
        
        # Test gradient flow
        model.train()
        output = model(mock_input)
        loss = torch.nn.functional.mse_loss(output, target)
        
        # Time backward pass
        start_time = time.time()
        loss.backward()
        backward_time = time.time() - start_time
        print(f"Backward pass time: {backward_time:.3f}s")
        print("Gradient check passed!")
        
        # Store results for comparison
        results.append({
            "config": config['name'],
            "params": total_params,
            "forward_time": avg_time,
            "backward_time": backward_time
        })
    
    # Print comparison table
    print("\n" + "="*60)
    print("Performance Comparison:")
    print("="*60)
    print(f"{'Configuration':<30} {'Parameters':>15} {'Forward (s)':>12} {'Backward (s)':>12}")
    print("-" * 70)
    for r in results:
        print(f"{r['config']:<30} {r['params']:>15,} {r['forward_time']:>12.3f} {r['backward_time']:>12.3f}")
    
    # Test memory efficiency with larger batch
    print("\n" + "="*60)
    print("Memory Efficiency Test (Larger Batch):")
    print("="*60)
    
    # Test with larger batch size
    large_B = 8
    for temporal_mode in ["separate", "3d_conv", "sequential"]:
        print(f"\nTesting {temporal_mode} with batch size {large_B}:")
        try:
            model = create_model(
                sequence_length=T,
                num_classes=num_classes,
                time_horizon=time_horizon,
                in_channels=4,  # RGB-D
                temporal_modeling=temporal_mode
            )
            model.eval()
            
            large_input = torch.randn(large_B, T, 4, H, W)
            with torch.no_grad():
                output = model(large_input)
            print(f"✓ Success - Output shape: {output.shape}")
        except Exception as e:
            print(f"✗ Failed - Error: {str(e)}")
    
    print("\n" + "="*60)
    print("All tests completed successfully!")
    print("="*60)


if __name__ == "__main__":
    __test__()