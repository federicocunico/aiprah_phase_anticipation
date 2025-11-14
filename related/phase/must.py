import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
import math


class MultiHeadPoolingAttention(nn.Module):
    """Multi-Head Pooling Attention from MViT"""
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,
                 pool_kernel=(3, 3, 3), pool_stride=(2, 2, 2), pool_padding=(1, 1, 1)):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # Pooling for Q, K, V
        self.pool_q = nn.MaxPool3d(kernel_size=pool_kernel, stride=pool_stride, 
                                     padding=pool_padding) if pool_stride[0] > 1 else nn.Identity()
        self.pool_k = nn.MaxPool3d(kernel_size=pool_kernel, stride=pool_stride,
                                     padding=pool_padding) if pool_stride[0] > 1 else nn.Identity()
        self.pool_v = nn.MaxPool3d(kernel_size=pool_kernel, stride=pool_stride,
                                     padding=pool_padding) if pool_stride[0] > 1 else nn.Identity()

    def forward(self, x, T, H, W):
        """
        Args:
            x: (B, N, C) where N = T*H*W
            T, H, W: temporal and spatial dimensions
        """
        B, N, C = x.shape
        
        # Handle optional class token: some upstream code concatenates a class token
        # at position 0, making N == T*H*W + 1. If present, separate it before
        # reshaping and reattach afterwards.
        has_cls = False
        cls_token = None
        expected_patches = T * H * W
        if N == expected_patches + 1:
            has_cls = True
            cls_token = x[:, :1, :]
            patch_tokens = x[:, 1:, :]
        else:
            patch_tokens = x

        # Reshape to 3D volume (patch_tokens has shape B x expected_patches x C)
        x_3d = patch_tokens.reshape(B, T, H, W, C).permute(0, 4, 1, 2, 3)  # (B, C, T, H, W)
        
        # Generate Q, K, V (compute from patch tokens only so class token isn't duplicated)
        qkv = self.qkv(patch_tokens).reshape(B, patch_tokens.shape[1], 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x_patches = (attn @ v).transpose(1, 2).reshape(B, -1, C)
        x_patches = self.proj(x_patches)

        # If there was a class token, reattach it to the front
        if has_cls:
            x = torch.cat([cls_token, x_patches], dim=1)
        else:
            x = x_patches
        x = self.proj_drop(x)
        
        return x


class MViTBlock(nn.Module):
    """Single MViT Transformer block"""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 pool_kernel=(1, 1, 1), pool_stride=(1, 1, 1)):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadPoolingAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            pool_kernel=pool_kernel, pool_stride=pool_stride, pool_padding=(1, 1, 1)
        )
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

    def forward(self, x, T, H, W):
        x = x + self.attn(self.norm1(x), T, H, W)
        x = x + self.mlp(self.norm2(x))
        return x


class SimplifiedMViT(nn.Module):
    """Simplified MViT backbone for surgical video"""
    def __init__(self, num_frames=16, img_size=224, patch_size=16, in_chans=3, 
                 embed_dim=96, depth=4, num_heads=1, mlp_ratio=4., drop_rate=0.):
        super().__init__()
        self.num_frames = num_frames
        self.patch_size = patch_size
        
        # Patch embedding
        self.patch_embed = nn.Conv3d(in_chans, embed_dim, 
                                     kernel_size=(3, patch_size, patch_size),
                                     stride=(2, patch_size, patch_size),
                                     padding=(1, 0, 0))
        
        # Calculate dimensions after patch embedding
        self.T = num_frames // 2
        self.H = img_size // patch_size
        self.W = img_size // patch_size
        num_patches = self.T * self.H * self.W
        
        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            MViTBlock(embed_dim, num_heads, mlp_ratio, True, drop_rate, drop_rate)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.embed_dim = embed_dim
        
    def forward(self, x):
        """
        Args:
            x: (B, T, C, H, W)
        Returns:
            cls_token: (B, embed_dim)
            patch_embeds: (B, N, embed_dim)
        """
        B = x.shape[0]
        
        # Rearrange to (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, embed_dim, T', H', W')
        
        # Flatten spatial-temporal dimensions
        x = x.flatten(2).transpose(1, 2)  # (B, N, embed_dim)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Apply Transformer blocks
        T, H, W = self.T, self.H, self.W
        for blk in self.blocks:
            x = blk(x, T, H, W)
        
        x = self.norm(x)
        
        # Split class token and patches
        cls_token = x[:, 0]  # (B, embed_dim)
        patch_embeds = x[:, 1:]  # (B, N, embed_dim)
        
        return cls_token, patch_embeds


class MultiTemporalCrossAttention(nn.Module):
    """Multi-Temporal Cross-Attention Module"""
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.wq = nn.Linear(dim, dim)
        self.wk = nn.Linear(dim, dim)
        self.wv = nn.Linear(dim, dim)
        
    def forward(self, query_seq, context_seq):
        """
        Args:
            query_seq: (B, T', D) - embeddings from one scale
            context_seq: (B, N*T', D) - concatenated embeddings from all scales
        Returns:
            (B, T', D)
        """
        B, T_prime, D = query_seq.shape
        
        # Generate Q, K, V
        Q = self.wq(query_seq).reshape(B, T_prime, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.wk(context_seq).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.wv(context_seq).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention
        attn = (Q @ K.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        out = (attn @ V).transpose(1, 2).reshape(B, T_prime, D)
        return out


class MultiTemporalSelfAttention(nn.Module):
    """Multi-Temporal Self-Attention Module"""
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.wq = nn.Linear(dim, dim)
        self.wk = nn.Linear(dim, dim)
        self.wv = nn.Linear(dim, dim)
        
    def forward(self, x):
        """
        Args:
            x: (B, (N+1)*T', D) - concatenation of original and cross-attention outputs
        Returns:
            (B, (N+1)*T', D)
        """
        B, N_total, D = x.shape
        
        Q = self.wq(x).reshape(B, N_total, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.wk(x).reshape(B, N_total, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.wv(x).reshape(B, N_total, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = (Q @ K.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        out = (attn @ V).transpose(1, 2).reshape(B, N_total, D)
        return out


class MultiTemporalAttentionModule(nn.Module):
    """Complete Multi-Temporal Attention Module with MTCA and SA"""
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.mtca = MultiTemporalCrossAttention(dim, num_heads)
        self.sa = MultiTemporalSelfAttention(dim, num_heads)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
    def forward(self, embeddings_list):
        """
        Args:
            embeddings_list: List of (B, T', D) tensors, one per scale
        Returns:
            List of processed embeddings
        """
        # Concatenate all embeddings for context
        context = torch.cat(embeddings_list, dim=1)  # (B, N*T', D)
        
        # Apply MTCA to each scale
        mtca_outputs = []
        for emb in embeddings_list:
            mtca_out = self.mtca(self.norm1(emb), self.norm1(context))
            mtca_outputs.append(mtca_out)
        
        # Apply SA with residual connection
        sa_outputs = []
        for i, (orig_emb, mtca_out) in enumerate(zip(embeddings_list, mtca_outputs)):
            # Concatenate original and MTCA output
            combined = torch.cat([orig_emb, torch.cat(mtca_outputs, dim=1)], dim=1)
            sa_out = self.sa(self.norm2(combined))
            # Extract the part corresponding to this scale
            sa_outputs.append(sa_out[:, :orig_emb.shape[1], :])
        
        return sa_outputs


class MultiTermFrameEncoder(nn.Module):
    """Multi-Term Frame Encoder with temporal pyramid and multi-temporal attention"""
    def __init__(self, num_scales=4, num_frames=16, img_size=224, num_phases=7,
                 embed_dim=96, depth=4, num_heads=8):
        super().__init__()
        self.num_scales = num_scales
        
        # Shared video backbone (MViT)
        self.video_backbone = SimplifiedMViT(
            num_frames=num_frames,
            img_size=img_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=1
        )
        
        # Multi-Temporal Attention Module
        self.mt_attention = MultiTemporalAttentionModule(embed_dim, num_heads=num_heads)
        
        # MLP to combine class tokens from all scales
        self.fusion_mlp = nn.Sequential(
            nn.Linear(embed_dim * num_scales, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        
        # Classification head for training
        self.phase_classifier = nn.Linear(embed_dim, num_phases)
        
        self.embed_dim = embed_dim
        
    def forward(self, video_pyramid):
        """
        Args:
            video_pyramid: List of (B, T, C, H, W) tensors, one per scale
                           with increasing temporal strides
        Returns:
            multi_term_embedding: (B, embed_dim) - fused embedding
            phase_logits: (B, num_phases) - phase predictions
        """
        B = video_pyramid[0].shape[0]
        
        # Extract features from each scale
        cls_tokens = []
        patch_embeds_list = []
        
        for scale_video in video_pyramid:
            cls_token, patch_embeds = self.video_backbone(scale_video)
            cls_tokens.append(cls_token)
            patch_embeds_list.append(patch_embeds)
        
        # Apply Multi-Temporal Attention
        attended_embeds = self.mt_attention(patch_embeds_list)
        
        # Get class tokens after attention (use first token from each scale)
        attended_cls_tokens = [emb[:, 0] for emb in attended_embeds]
        
        # Concatenate and fuse class tokens
        concatenated_cls = torch.cat(attended_cls_tokens, dim=1)  # (B, num_scales * embed_dim)
        multi_term_embedding = self.fusion_mlp(concatenated_cls)  # (B, embed_dim)
        
        # Phase classification
        phase_logits = self.phase_classifier(multi_term_embedding)
        
        return multi_term_embedding, phase_logits


class TemporalConsistencyModule(nn.Module):
    """Temporal Consistency Module using Transformer encoder"""
    def __init__(self, embed_dim=96, num_heads=8, num_layers=4, num_phases=7):
        super().__init__()
        
        # Positional encoding
        self.pos_encoder = nn.Parameter(torch.randn(1, 1000, embed_dim) * 0.02)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.classifier = nn.Linear(embed_dim, num_phases)
        
    def forward(self, frame_embeddings):
        """
        Args:
            frame_embeddings: (B, F', embed_dim) - sequence of multi-term frame embeddings
        Returns:
            phase_logits: (B, F', num_phases) - phase predictions for each frame
        """
        B, F_prime, D = frame_embeddings.shape
        
        # Add positional encoding
        x = frame_embeddings + self.pos_encoder[:, :F_prime, :]
        
        # Apply Transformer encoder
        x = self.transformer_encoder(x)
        
        # Classify each frame
        phase_logits = self.classifier(x)  # (B, F', num_phases)
        
        return phase_logits


class MuST(nn.Module):
    """
    MuST: Multi-Scale Transformers for Surgical Phase Recognition
    
    Complete model with Multi-Term Frame Encoder and Temporal Consistency Module
    """
    def __init__(self, num_scales=4, num_frames=16, img_size=224, num_phases=7,
                 embed_dim=96, depth=4, num_heads=8, tcm_layers=4):
        super().__init__()
        
        # Multi-Term Frame Encoder
        self.mtfe = MultiTermFrameEncoder(
            num_scales=num_scales,
            num_frames=num_frames,
            img_size=img_size,
            num_phases=num_phases,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads
        )
        
        # Temporal Consistency Module
        self.tcm = TemporalConsistencyModule(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=tcm_layers,
            num_phases=num_phases
        )
        
        self.num_scales = num_scales
        self.num_frames = num_frames
        
    def forward(self, video_pyramids, use_tcm=True):
        """
        Args:
            video_pyramids: (B, F', num_scales, T, C, H, W) - pyramid for each keyframe
                           F' is the number of keyframes in the sequence
                           OR List of length F' containing pyramids
            use_tcm: bool - whether to use TCM
        Returns:
            If use_tcm:
                tcm_logits: (B, F', num_phases)
                mtfe_logits: (B, F', num_phases)
            Else:
                mtfe_logits: (B, F', num_phases)
        """
        if isinstance(video_pyramids, torch.Tensor):
            B, F_prime, num_scales, T, C, H, W = video_pyramids.shape
            
            # Process each keyframe through MTFE
            all_embeddings = []
            all_mtfe_logits = []
            
            for f in range(F_prime):
                # Extract pyramid for this keyframe
                pyramid = [video_pyramids[:, f, s] for s in range(num_scales)]
                
                # Get multi-term embedding
                embedding, logits = self.mtfe(pyramid)
                all_embeddings.append(embedding)
                all_mtfe_logits.append(logits)
            
            # Stack embeddings
            frame_embeddings = torch.stack(all_embeddings, dim=1)  # (B, F', embed_dim)
            mtfe_logits = torch.stack(all_mtfe_logits, dim=1)  # (B, F', num_phases)
        else:
            # List of pyramids
            all_embeddings = []
            all_mtfe_logits = []
            
            for pyramid in video_pyramids:
                embedding, logits = self.mtfe(pyramid)
                all_embeddings.append(embedding)
                all_mtfe_logits.append(logits)
            
            frame_embeddings = torch.stack(all_embeddings, dim=1)
            mtfe_logits = torch.stack(all_mtfe_logits, dim=1)
        
        if use_tcm:
            # Apply TCM
            tcm_logits = self.tcm(frame_embeddings)
            return {
                'tcm_logits': tcm_logits,
                'mtfe_logits': mtfe_logits,
                'embeddings': frame_embeddings
            }
        else:
            return {
                'mtfe_logits': mtfe_logits,
                'embeddings': frame_embeddings
            }


def fps():
    """
    Measure FPS (frames per second) performance of MuST
    Returns dict with timing information in milliseconds
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    model = MuST(
        num_scales=4,
        num_frames=16,
        img_size=224,
        num_phases=7,
        embed_dim=96,
        depth=4,
        num_heads=8,
        tcm_layers=4
    ).to(device)
    model.eval()
    
    # Create dummy inputs
    # Simulate temporal pyramid with 4 scales, different sampling rates
    B = 1
    F_prime = 10  # 10 keyframes in sequence
    num_scales = 4
    T = 16  # frames per sequence
    C, H, W = 3, 224, 224
    
    # Create pyramid (B, F', num_scales, T, C, H, W)
    video_pyramids = torch.randn(B, F_prime, num_scales, T, C, H, W).to(device)
    
    # Warmup
    print("Warming up...")
    with torch.no_grad():
        for _ in range(5):
            _ = model(video_pyramids, use_tcm=True)
    
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
            _ = model(video_pyramids, use_tcm=True)
            
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
        'num_keyframes': F_prime,
        'num_scales': num_scales
    }
    
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("MuST: Multi-Scale Transformers for Surgical Phase Recognition")
    print("=" * 60)
    
    # Test model architecture
    print("\n1. Testing model architecture...")
    model = MuST(
        num_scales=4,
        num_frames=16,
        img_size=224,
        num_phases=7,
        embed_dim=96,
        depth=4,
        num_heads=8
    )
    
    # Create dummy pyramid
    B = 2
    F_prime = 5  # 5 keyframes
    num_scales = 4
    T = 16
    
    # Method 1: Single tensor
    video_pyramids = torch.randn(B, F_prime, num_scales, T, 3, 224, 224)
    
    print(f"   Input shape: {video_pyramids.shape}")
    
    # Forward pass
    outputs = model(video_pyramids, use_tcm=True)
    
    print(f"   Output shapes:")
    for key, val in outputs.items():
        if isinstance(val, torch.Tensor):
            print(f"     {key}: {val.shape}")
    
    # Test without TCM
    outputs_no_tcm = model(video_pyramids, use_tcm=False)
    print(f"   Without TCM:")
    for key, val in outputs_no_tcm.items():
        if isinstance(val, torch.Tensor):
            print(f"     {key}: {val.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    mtfe_params = sum(p.numel() for p in model.mtfe.parameters())
    tcm_params = sum(p.numel() for p in model.tcm.parameters())
    
    print(f"\n   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   MTFE parameters: {mtfe_params:,}")
    print(f"   TCM parameters: {tcm_params:,}")
    
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
    Standardized test helper for MuST: run warmup then timed runs with temporal length T.
    Returns dict with mean_ms and std_ms.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MuST(num_scales=4, num_frames=T, img_size=224, num_phases=7,
                 embed_dim=96, depth=4, num_heads=8, tcm_layers=4).to(device)
    model.eval()

    B = 1
    F_prime = 10
    num_scales = 4
    C, H, W = 3, 224, 224

    video_pyramids = torch.randn(B, F_prime, num_scales, T, C, H, W).to(device)

    try:
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(video_pyramids, use_tcm=True)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

        times = []
        with torch.no_grad():
            for _ in range(runs):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start = time.perf_counter()
                _ = model(video_pyramids, use_tcm=True)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end = time.perf_counter()
                times.append((end - start) * 1000.0)

        arr = np.array(times)
        return {"mean_ms": float(np.mean(arr)), "std_ms": float(np.std(arr)), "device": str(device), "T": T}
    except Exception as e:
        return {"mean_ms": None, "std_ms": None, "error": str(e), "device": str(device), "T": T}