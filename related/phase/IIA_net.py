import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
from torchvision.models import resnet50

class InstrumentInstrumentEncoder(nn.Module):
    """Encodes geometric relations between grasper and other instruments"""
    def __init__(self, embed_dim=64, max_instruments=10):
        super().__init__()
        self.max_instruments = max_instruments
        # 4 geometric features + 7 instrument categories
        self.input_dim = 4 + 7
        
        # Multiple linear layers for processing geometric features
        self.fc1 = nn.Linear(self.input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)  # For attention weights
        self.fc_out = nn.Linear(self.input_dim, embed_dim)
        
    def forward(self, bboxes):
        """
        Args:
            bboxes: (B, T, M, 4+7) where M is max number of interactions
                    First 4 dims: geometric features [log(|x_g-x_m|/w_g), log(|y_g-y_m|/h_g), log(w_m/w_g), log(h_m/h_g)]
                    Last 7 dims: one-hot encoded instrument category
        Returns:
            (B, T, embed_dim)
        """
        B, T, M, _ = bboxes.shape
        
        # Compute attention weights
        x = F.relu(self.fc1(bboxes))
        x = F.relu(self.fc2(x))
        attn = self.fc3(x)  # (B, T, M, 1)
        
        # Apply attention and sum over instruments
        weighted_features = attn * bboxes  # (B, T, M, input_dim)
        summed = weighted_features.sum(dim=2)  # (B, T, input_dim)
        
        # Final projection
        out = self.fc_out(summed)  # (B, T, embed_dim)
        return out

class InstrumentSurroundingEncoder(nn.Module):
    """Encodes instrument surroundings from semantic segmentation"""
    def __init__(self, num_classes=7, embed_dim=64):
        super().__init__()
        self.num_classes = num_classes
        
        # Convolutional layers to process semantic maps
        self.conv1 = nn.Conv2d(num_classes, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, embed_dim, kernel_size=3, stride=2, padding=1)
        
    def forward(self, semantic_maps):
        """
        Args:
            semantic_maps: (B, T, num_classes, H, W) - one-hot encoded semantic maps
        Returns:
            (B, T, embed_dim)
        """
        B, T, C, H, W = semantic_maps.shape
        
        # Reshape to process all timesteps at once
        x = semantic_maps.view(B * T, C, H, W)
        
        # Apply convolutions
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        # Global average pooling
        x = F.adaptive_avg_pool2d(x, 1)  # (B*T, embed_dim, 1, 1)
        x = x.view(B, T, -1)  # (B, T, embed_dim)
        
        return x

class InstrumentInteractionModule(nn.Module):
    """IIM: Combines instrument-instrument and instrument-surrounding interactions"""
    def __init__(self, embed_dim=64):
        super().__init__()
        self.instrument_encoder = InstrumentInstrumentEncoder(embed_dim=embed_dim)
        self.surrounding_encoder = InstrumentSurroundingEncoder(embed_dim=embed_dim)
        
    def forward(self, bboxes, semantic_maps):
        """
        Args:
            bboxes: (B, T, M, 11) - instrument bounding boxes with categories
            semantic_maps: (B, T, 7, H, W) - semantic segmentation maps
        Returns:
            (B, T, 2*embed_dim) - concatenated interaction features
        """
        inst_inst_feat = self.instrument_encoder(bboxes)
        inst_surr_feat = self.surrounding_encoder(semantic_maps)
        
        return torch.cat([inst_inst_feat, inst_surr_feat], dim=-1)

class SpatialFeatureExtractor(nn.Module):
    """Extracts spatial features using ResNet50, IIM, and signal embeddings"""
    def __init__(self, num_tools=7, num_phases=7, visual_dim=2048, iim_dim=128, 
                 tool_embed_dim=32, phase_embed_dim=32):
        super().__init__()
        
        # Visual feature extractor (ResNet50)
        resnet = resnet50(pretrained=False)
        self.visual_encoder = nn.Sequential(*list(resnet.children())[:-1])  # Remove final FC
        
        # Instrument Interaction Module
        self.iim = InstrumentInteractionModule(embed_dim=64)
        
        # Tool and phase signal embeddings
        self.tool_embedding = nn.Embedding(num_tools + 1, tool_embed_dim)  # +1 for no tool
        self.phase_embedding = nn.Embedding(num_phases, phase_embed_dim)
        
        self.output_dim = visual_dim + iim_dim + tool_embed_dim + phase_embed_dim
        
    def forward(self, frames, bboxes, semantic_maps, tool_signals, phase_signals):
        """
        Args:
            frames: (B, T, 3, H, W) - video frames
            bboxes: (B, T, M, 11) - instrument bounding boxes
            semantic_maps: (B, T, 7, H, W) - semantic maps
            tool_signals: (B, T) - tool presence signals (indices)
            phase_signals: (B, T) - phase signals (indices)
        Returns:
            (B, T, output_dim) - combined features
        """
        B, T = frames.shape[:2]
        
        # Extract visual features
        x = frames.view(B * T, *frames.shape[2:])
        visual_feat = self.visual_encoder(x)
        visual_feat = visual_feat.view(B, T, -1)
        
        # Extract interaction features
        interaction_feat = self.iim(bboxes, semantic_maps)
        
        # Embed tool and phase signals
        tool_feat = self.tool_embedding(tool_signals)
        phase_feat = self.phase_embedding(phase_signals)
        
        # Concatenate all features
        combined = torch.cat([visual_feat, interaction_feat, tool_feat, phase_feat], dim=-1)
        
        return combined

class DilatedResidualLayer(nn.Module):
    """Single dilated residual convolutional layer for MS-TCN"""
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                             padding=0, dilation=dilation)
        self.dropout = nn.Dropout(0.5)
        
        # Match dimensions for residual connection
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Conv1d(in_channels, out_channels, 1)
    
    def forward(self, x, causal_padding):
        """
        Args:
            x: (B, C, T)
            causal_padding: amount of padding to add on left
        Returns:
            (B, C', T)
        """
        # Apply causal padding (only on the left/past)
        x_padded = F.pad(x, (causal_padding, 0))
        
        # Convolution
        out = self.conv(x_padded)
        out = F.relu(out)
        out = self.dropout(out)
        
        # Residual connection
        if self.downsample is not None:
            x = self.downsample(x)
        
        # Trim to match output length and add residual
        if x.shape[-1] > out.shape[-1]:
            x = x[:, :, -out.shape[-1]:]
        
        return out + x

class SingleStageModel(nn.Module):
    """Single stage of MS-TCN with multiple dilated layers"""
    def __init__(self, num_layers, num_f_maps, dim, num_classes_phase, num_classes_tool):
        super().__init__()
        
        self.conv_in = nn.Conv1d(dim, num_f_maps, 1)
        
        # Stack of dilated residual layers
        self.layers = nn.ModuleList([
            DilatedResidualLayer(num_f_maps, num_f_maps, 3, 2**i)
            for i in range(num_layers)
        ])
        
        # Output layers for phase and tool anticipation
        self.conv_out_phase_reg = nn.Conv1d(num_f_maps, num_classes_phase, 1)
        self.conv_out_phase_cls = nn.Conv1d(num_f_maps, num_classes_phase * 3, 1)  # 3 categories
        
        self.conv_out_tool_reg = nn.Conv1d(num_f_maps, num_classes_tool, 1)
        self.conv_out_tool_cls = nn.Conv1d(num_f_maps, num_classes_tool * 3, 1)
        
        self.num_layers = num_layers
        
    def forward(self, x):
        """
        Args:
            x: (B, dim, T)
        Returns:
            dict with 'phase_reg', 'phase_cls', 'tool_reg', 'tool_cls' each (B, num_classes, T)
        """
        out = self.conv_in(x)

        # Apply dilated layers with causal padding
        for i, layer in enumerate(self.layers):
            causal_padding = 2**i * (3 - 1)  # dilation * (kernel_size - 1)
            out = layer(out, causal_padding)

        # Generate predictions
        phase_reg = self.conv_out_phase_reg(out)
        phase_cls = self.conv_out_phase_cls(out)

        tool_reg = self.conv_out_tool_reg(out)
        tool_cls = self.conv_out_tool_cls(out)

        # Return predictions and the internal features so refinement stages can use the
        # num_f_maps feature representation (not the high-dim original input)
        return {
            'phase_reg': phase_reg,
            'phase_cls': phase_cls,
            'tool_reg': tool_reg,
            'tool_cls': tool_cls,
            '_features': out
        }

class MultiStageTemporalCNN(nn.Module):
    """Multi-stage temporal convolutional network (MS-TCN) for anticipation"""
    def __init__(self, num_stages, num_layers, num_f_maps, dim, 
                 num_classes_phase, num_classes_tool):
        super().__init__()
        
        self.stage1 = SingleStageModel(num_layers, num_f_maps, dim, 
                                       num_classes_phase, num_classes_tool)
        
        # Additional stages take previous predictions as input
        self.stages = nn.ModuleList([
            SingleStageModel(num_layers, num_f_maps, 
                           num_f_maps + num_classes_phase + num_classes_tool,
                           num_classes_phase, num_classes_tool)
            for _ in range(num_stages - 1)
        ])
        
    def forward(self, x):
        """
        Args:
            x: (B, dim, T)
        Returns:
            List of predictions from each stage
        """
        # First stage: compute predictions and obtain internal features
        out = self.stage1(x)
        outputs = [out]

        # Refinement stages: each stage expects input channels = num_f_maps + num_classes_phase + num_classes_tool
        # Build x_stage by concatenating the previous stage's internal features (num_f_maps) with the
        # previous predictions (phase_reg and tool_reg) so the input channels match.
        for stage in self.stages:
            prev_features = out.get('_features', None)
            if prev_features is None:
                # Fallback to recomputing features from x using stage1.conv_in (shouldn't happen normally)
                prev_features = self.stage1.conv_in(x)
            x_stage = torch.cat([prev_features, out['phase_reg'], out['tool_reg']], dim=1)
            out = stage(x_stage)
            outputs.append(out)
        
        return outputs

class IIANet(nn.Module):
    """
    Complete IIA-Net: Instrument Interaction Aware Anticipation Network
    for surgical workflow anticipation
    """
    def __init__(self, num_tools=7, num_phases=7, num_stages=2, num_layers=8, 
                 num_f_maps=64, horizon_minutes=5):
        super().__init__()
        
        # Spatial feature extractor
        self.feature_extractor = SpatialFeatureExtractor(
            num_tools=num_tools,
            num_phases=num_phases,
            visual_dim=2048,
            iim_dim=128,
            tool_embed_dim=32,
            phase_embed_dim=32
        )
        
        feature_dim = self.feature_extractor.output_dim
        
        # Temporal model (MS-TCN)
        self.temporal_model = MultiStageTemporalCNN(
            num_stages=num_stages,
            num_layers=num_layers,
            num_f_maps=num_f_maps,
            dim=feature_dim,
            num_classes_phase=num_phases,
            num_classes_tool=num_tools
        )
        
        self.horizon = horizon_minutes
        
    def forward(self, frames, bboxes, semantic_maps, tool_signals, phase_signals):
        """
        Args:
            frames: (B, T, 3, 224, 224) - video frames
            bboxes: (B, T, M, 11) - instrument bounding boxes with categories
            semantic_maps: (B, T, 7, 56, 56) - semantic segmentation maps
            tool_signals: (B, T) - tool presence signals
            phase_signals: (B, T) - phase signals
        Returns:
            List of predictions from each MS-TCN stage
        """
        # Extract spatial features
        features = self.feature_extractor(frames, bboxes, semantic_maps, 
                                         tool_signals, phase_signals)
        
        # features: (B, T, feature_dim) -> transpose to (B, feature_dim, T) for conv1d
        features = features.transpose(1, 2)
        
        # Temporal modeling
        outputs = self.temporal_model(features)
        
        return outputs


def fps():
    """
    Measure FPS (frames per second) performance of IIA-Net
    Returns dict with timing information in milliseconds
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    model = IIANet(
        num_tools=7,
        num_phases=7,
        num_stages=2,
        num_layers=8,
        num_f_maps=64
    ).to(device)
    model.eval()
    
    # Create dummy inputs (batch_size=1, temporal_length=16 frames)
    B, T = 1, 16
    M = 5  # max instruments
    H_frame, W_frame = 224, 224
    H_seg, W_seg = 56, 56
    
    frames = torch.randn(B, T, 3, H_frame, W_frame).to(device)
    bboxes = torch.randn(B, T, M, 11).to(device)
    semantic_maps = torch.randn(B, T, 7, H_seg, W_seg).to(device)
    tool_signals = torch.randint(0, 7, (B, T)).to(device)
    phase_signals = torch.randint(0, 7, (B, T)).to(device)
    
    # Warmup
    print("Warming up...")
    with torch.no_grad():
        for _ in range(5):
            _ = model(frames, bboxes, semantic_maps, tool_signals, phase_signals)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Benchmark
    print("Benchmarking...")
    times = []
    num_runs = 20
    
    with torch.no_grad():
        for i in range(num_runs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            _ = model(frames, bboxes, semantic_maps, tool_signals, phase_signals)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms
            
            if (i + 1) % 5 == 0:
                print(f"  Run {i+1}/{num_runs}: {times[-1]:.2f} ms")
    
    # Calculate statistics
    times = np.array(times)
    
    results = {
        'mean_ms': float(np.mean(times)),
        'std_ms': float(np.std(times)),
        'min_ms': float(np.min(times)),
        'max_ms': float(np.max(times)),
        'median_ms': float(np.median(times)),
        'fps': 1000.0 / np.mean(times),
        'device': str(device),
        'batch_size': B,
        'temporal_length': T
    }
    
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("IIA-Net: Instrument Interaction Aware Anticipation Network")
    print("=" * 60)
    
    # Test model architecture
    print("\n1. Testing model architecture...")
    model = IIANet(num_tools=7, num_phases=7, num_stages=2)
    
    # Create dummy inputs
    B, T = 2, 10
    frames = torch.randn(B, T, 3, 224, 224)
    bboxes = torch.randn(B, T, 5, 11)
    semantic_maps = torch.randn(B, T, 7, 56, 56)
    tool_signals = torch.randint(0, 7, (B, T))
    phase_signals = torch.randint(0, 7, (B, T))
    
    outputs = model(frames, bboxes, semantic_maps, tool_signals, phase_signals)
    
    print(f"   Input shape: frames={frames.shape}")
    print(f"   Number of stages: {len(outputs)}")
    print(f"   Output shapes:")
    for key, val in outputs[-1].items():
        print(f"     {key}: {val.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Measure FPS
    print("\n2. Measuring inference performance...")
    results = fps()
    
    print("\n" + "=" * 60)
    print("PERFORMANCE RESULTS")
    print("=" * 60)
    for key, value in results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    print("=" * 60)


def test_latency(warmup: int = 5, runs: int = 10, T: int = 16):
    """
    Standardized test helper for IIANet: run warmup then timed runs with temporal length T.
    Returns dict with mean_ms and std_ms.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = IIANet(num_tools=7, num_phases=7, num_stages=4, num_layers=12, num_f_maps=64).to(device)
    model.eval()

    B = 1
    M = 5
    H_frame, W_frame = 224, 224
    H_seg, W_seg = 56, 56

    frames = torch.randn(B, T, 3, H_frame, W_frame).to(device)
    bboxes = torch.randn(B, T, M, 11).to(device)
    semantic_maps = torch.randn(B, T, 7, H_seg, W_seg).to(device)
    tool_signals = torch.randint(0, 7, (B, T)).to(device)
    phase_signals = torch.randint(0, 7, (B, T)).to(device)

    try:
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(frames, bboxes, semantic_maps, tool_signals, phase_signals)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

        times = []
        with torch.no_grad():
            for _ in range(runs):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start = time.perf_counter()
                _ = model(frames, bboxes, semantic_maps, tool_signals, phase_signals)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end = time.perf_counter()
                times.append((end - start) * 1000.0)

        arr = np.array(times)
        return {"mean_ms": float(np.mean(arr)), "std_ms": float(np.std(arr)), "device": str(device), "T": T}
    except Exception as e:
        return {"mean_ms": None, "std_ms": None, "error": str(e), "device": str(device), "T": T}