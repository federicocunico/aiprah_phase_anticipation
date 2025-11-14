import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
import math


class VisionEncoder(nn.Module):
    """
    Vision Encoder based on ViT-B/16.
    Pre-trained on ImageNet and fine-tuned with AVT approach.
    
    Input: 248×248 pixels
    Output: 768D embeddings per frame
    """
    def __init__(self, img_size=248, patch_size=16, in_channels=3, 
                 embed_dim=768, depth=12, num_heads=12):
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
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True
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
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, 768, 15, 15)
        x = x.flatten(2).transpose(1, 2)  # (B, 225, 768)
        
        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, 226, 768)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Transformer
        x = self.transformer(x)
        
        # Use class token
        x = self.norm(x[:, 0])  # (B, 768)
        
        return x


class WindowedSelfAttention(nn.Module):
    """
    Windowed Self-Attention (WSA) encoder.
    
    Uses sliding windows of width W with no overlap to perform self-attention.
    Input length: L = 1440 frames
    Window width: W = 20
    Output: same length and dimensionality
    """
    def __init__(self, embed_dim=768, window_size=20, num_heads=8, num_layers=2):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.window_size = window_size
        
        # Transformer encoder for windows
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, embed_dim) where seq_len = L = 1440
        Returns:
            output: (batch, seq_len, embed_dim)
        """
        batch_size, seq_len, embed_dim = x.shape
        
        # Calculate number of windows
        num_windows = seq_len // self.window_size
        
        # Reshape into windows
        x_windows = x[:, :num_windows * self.window_size, :].view(
            batch_size, num_windows, self.window_size, embed_dim
        )  # (B, num_windows, W, D)
        
        # Process each window independently
        x_windows = x_windows.view(batch_size * num_windows, self.window_size, embed_dim)
        x_windows = self.transformer(x_windows)  # (B*num_windows, W, D)
        
        # Reshape back
        x_windows = x_windows.view(batch_size, num_windows, self.window_size, embed_dim)
        output = x_windows.reshape(batch_size, num_windows * self.window_size, embed_dim)
        
        # Handle remaining frames (if seq_len not divisible by window_size)
        if seq_len > num_windows * self.window_size:
            remaining = x[:, num_windows * self.window_size:, :]
            output = torch.cat([output, remaining], dim=1)
        
        return output


class GlobalKeyPooling(nn.Module):
    """
    Global key-pooling as used in SKiT.
    
    Compresses temporal features via cumulative max-pooling.
    Projects to lower dimension d=32, then compresses to M tokens.
    """
    def __init__(self, input_dim=768, key_dim=32, num_tokens=24):
        super().__init__()
        
        self.input_dim = input_dim
        self.key_dim = key_dim
        self.num_tokens = num_tokens
        
        # Project to key dimension
        self.proj = nn.Linear(input_dim, key_dim)
        
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim) where seq_len = L = 1440
        Returns:
            key_features: (batch, num_tokens, key_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to key space
        x = self.proj(x)  # (B, L, key_dim)
        
        # Compute pooled representation of first (L-M) frames
        M = self.num_tokens
        if seq_len - M > 0:
            p = torch.max(x[:, : seq_len - M, :], dim=1, keepdim=True)[0]  # (B, 1, key_dim)
        else:
            # Not enough frames to compute the pooled prefix: use zeros as prefix
            p = torch.zeros(x.shape[0], 1, self.key_dim, device=x.device)
        
        # For each of the most recent M frames, compute cumulative maximum
        recent_frames = x[:, seq_len - M:, :]  # (B, M, key_dim)
        
        key_features = []
        for m in range(M):
            # Include p and frames up to current position
            frames_upto_m = torch.cat([p, recent_frames[:, :m+1, :]], dim=1)  # (B, m+2, key_dim)
            km = torch.max(frames_upto_m, dim=1)[0]  # (B, key_dim)
            key_features.append(km)
        
        key_features = torch.stack(key_features, dim=1)  # (B, M, key_dim)
        
        return key_features


class IntervalPooling(nn.Module):
    """
    Interval-pooling for autoregressive decoder.
    
    Performs max-pooling over 60-second intervals.
    """
    def __init__(self, input_dim=768, interval=60):
        super().__init__()
        
        self.input_dim = input_dim
        self.interval = interval
        
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim) where seq_len = L
        Returns:
            pooled: (batch, num_intervals, input_dim)
        """
        batch_size, seq_len, input_dim = x.shape
        
        # Calculate number of intervals
        num_intervals = seq_len // self.interval
        
        # Reshape and pool
        x_intervals = x[:, :num_intervals * self.interval, :].view(
            batch_size, num_intervals, self.interval, input_dim
        )
        
        # Max-pool over each interval
        pooled = torch.max(x_intervals, dim=2)[0]  # (B, num_intervals, input_dim)
        
        return pooled


class PriorKnowledgeEmbedding(nn.Module):
    """
    Prior Knowledge Embedding using class transition probabilities.
    
    Computes P(y_{t+h_n*60} = j | y_t = i) from training data.
    """
    def __init__(self, num_classes=8, max_horizon=60):
        super().__init__()
        
        self.num_classes = num_classes
        self.max_horizon = max_horizon
        
        # Transition probability tensor: (num_classes, num_classes, max_horizon)
        # Will be filled during training/initialization
        self.register_buffer('transition_probs', 
                           torch.zeros(num_classes, num_classes, max_horizon))
        
    def compute_transition_probs(self, labels, horizon_minutes):
        """
        Compute transition probabilities from training data.
        
        Args:
            labels: List of label sequences from training data
            horizon_minutes: List of minute indices to compute probabilities for
        """
        # Initialize counts
        counts = torch.zeros(self.num_classes, self.num_classes, self.max_horizon)
        totals = torch.zeros(self.num_classes, self.max_horizon)
        
        # Count transitions (simplified - would need actual implementation)
        # This is a placeholder that should be computed from real training data
        # For now, initialize with uniform distribution
        for i in range(self.num_classes):
            for h in range(self.max_horizon):
                totals[i, h] = 1.0
                for j in range(self.num_classes):
                    counts[i, j, h] = 1.0 / self.num_classes
        
        # Compute probabilities
        for i in range(self.num_classes):
            for h in range(self.max_horizon):
                if totals[i, h] > 0:
                    self.transition_probs[i, :, h] = counts[i, :, h] / totals[i, h]
        
    def forward(self, current_class, future_minutes):
        """
        Args:
            current_class: (batch,) current predicted class indices
            future_minutes: (N,) future minute indices [h1, h2, ..., hN]
        Returns:
            embeddings: (batch, N, num_classes) probability embeddings
        """
        batch_size = current_class.shape[0]
        N = len(future_minutes)
        
        embeddings = []
        for n in range(N):
            h = future_minutes[n]
            h = min(h, self.max_horizon - 1)  # Clamp to max horizon
            
            # Get probabilities for each sample in batch
            batch_probs = []
            for b in range(batch_size):
                i = current_class[b].item()
                probs = self.transition_probs[i, :, h]  # (num_classes,)
                batch_probs.append(probs)
            
            batch_probs = torch.stack(batch_probs, dim=0)  # (batch, num_classes)
            embeddings.append(batch_probs)
        
        embeddings = torch.stack(embeddings, dim=1)  # (batch, N, num_classes)
        
        return embeddings


class SinglePassDecoder(nn.Module):
    """
    Single-Pass (SP) Decoder.
    
    Generates N output tokens in a single forward pass.
    Uses cross-attention with context tokens and optional prior knowledge.
    """
    def __init__(self, embed_dim=512, num_heads=8, num_layers=4, 
                 num_classes=8, use_prior=False):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.use_prior = use_prior
        
        # Input embedding
        if use_prior:
            self.input_embed = nn.Linear(num_classes, embed_dim)
        else:
            self.input_embed = nn.Embedding(num_classes, embed_dim)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, 100, embed_dim))
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output layer
        self.output_layer = nn.Linear(embed_dim, num_classes)
        
    def forward(self, context_tokens, prior_embeddings=None, num_tokens=30):
        """
        Args:
            context_tokens: (batch, M, embed_dim) context from encoder
            prior_embeddings: (batch, N, num_classes) optional prior knowledge
            num_tokens: N, number of future tokens to generate
        Returns:
            output: (batch, N, num_classes) predicted probabilities
        """
        batch_size = context_tokens.shape[0]
        
        if self.use_prior and prior_embeddings is not None:
            # Use prior knowledge as input
            tgt = self.input_embed(prior_embeddings)  # (B, N, embed_dim)
        else:
            # Use learned embeddings
            tgt_indices = torch.zeros(batch_size, num_tokens, dtype=torch.long, 
                                     device=context_tokens.device)
            tgt = self.input_embed(tgt_indices)  # (B, N, embed_dim)
        
        # Add positional encoding
        tgt = tgt + self.pos_encoding[:, :num_tokens, :]
        
        # Decoder forward
        output = self.decoder(tgt, context_tokens)  # (B, N, embed_dim)
        
        # Project to class probabilities
        output = self.output_layer(output)  # (B, N, num_classes)
        
        return output


class AutoregressiveDecoder(nn.Module):
    """
    Autoregressive (AR) Decoder using GPT-2 style architecture.
    
    Uses causal masking to predict next token iteratively.
    """
    def __init__(self, embed_dim=512, num_heads=8, num_layers=6, num_classes=8):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, 100, embed_dim))
        
        # Causal transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output layer
        self.output_layer = nn.Linear(embed_dim, num_classes)
        
    def forward(self, context_tokens, max_tokens=30):
        """
        Args:
            context_tokens: (batch, M, embed_dim) from encoder
            max_tokens: maximum number of tokens to generate
        Returns:
            output: (batch, max_tokens, num_classes)
        """
        batch_size = context_tokens.shape[0]
        device = context_tokens.device
        
        # Start with context tokens as initial sequence
        generated_sequence = context_tokens
        outputs = []
        
        for t in range(max_tokens):
            # Add positional encoding
            seq_with_pos = generated_sequence + self.pos_encoding[:, :generated_sequence.shape[1], :]
            
            # Create causal mask
            seq_len = generated_sequence.shape[1]
            causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(device)
            
            # Decode
            decoded = self.decoder(seq_with_pos, context_tokens, tgt_mask=causal_mask)
            
            # Get last token prediction
            last_token = decoded[:, -1:, :]  # (B, 1, embed_dim)
            output = self.output_layer(last_token)  # (B, 1, num_classes)
            
            outputs.append(output)
            
            # Append to sequence for next iteration
            generated_sequence = torch.cat([generated_sequence, last_token], dim=1)
        
        outputs = torch.cat(outputs, dim=1)  # (B, max_tokens, num_classes)
        
        return outputs


class SWAG(nn.Module):
    """
    SWAG: Surgical Workflow Anticipative Generation.
    
    Combines phase recognition and long-term anticipation using generative approach.
    
    Architecture:
    1. Vision Encoder (ViT-B/16)
    2. Windowed Self-Attention (WSA)
    3. Compression and Pooling (CP)
    4. Decoder (SP or AR)
    5. Recognition and Anticipation Heads
    
    Paper: "SWAG: long-term surgical workflow prediction with generative-based anticipation"
    """
    def __init__(self,
                 img_size=248,
                 num_classes=8,  # 7 phases + 1 EOS
                 seq_length=1440,  # L = 24 minutes * 60 fps
                 window_size=20,
                 num_context_tokens=24,  # M = 24
                 decoder_type='single_pass',  # 'single_pass' or 'autoregressive'
                 use_prior_knowledge=True,
                 max_horizon=60):
        super().__init__()
        
        self.img_size = img_size
        self.num_classes = num_classes
        self.seq_length = seq_length
        self.num_context_tokens = num_context_tokens
        self.decoder_type = decoder_type
        self.use_prior_knowledge = use_prior_knowledge
        self.max_horizon = max_horizon
        
        # 1. Vision Encoder
        self.vision_encoder = VisionEncoder(
            img_size=img_size,
            embed_dim=768
        )
        
        # 2. Windowed Self-Attention
        self.wsa_encoder = WindowedSelfAttention(
            embed_dim=768,
            window_size=window_size,
            num_heads=8,
            num_layers=2
        )
        
        # 3. Compression and Pooling
        if decoder_type == 'single_pass':
            self.pooling = GlobalKeyPooling(
                input_dim=768,
                key_dim=32,
                num_tokens=num_context_tokens
            )
            context_dim = 32
        else:  # autoregressive
            self.pooling = IntervalPooling(
                input_dim=768,
                interval=60
            )
            context_dim = 768
        
        # Project context to decoder dimension
        self.context_proj = nn.Linear(context_dim, 512)
        
        # 4. Prior Knowledge Embedding (optional)
        if use_prior_knowledge:
            self.prior_embedding = PriorKnowledgeEmbedding(
                num_classes=num_classes,
                max_horizon=max_horizon
            )
        
        # 5. Decoder
        if decoder_type == 'single_pass':
            self.decoder = SinglePassDecoder(
                embed_dim=512,
                num_heads=8,
                num_layers=4,
                num_classes=num_classes,
                use_prior=use_prior_knowledge
            )
        else:
            self.decoder = AutoregressiveDecoder(
                embed_dim=512,
                num_heads=8,
                num_layers=6,
                num_classes=num_classes
            )
        
        # 6. Recognition Head
        self.recognition_head = nn.Sequential(
            nn.Linear(768 + 32, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )
        
        # 7. Anticipation Head (for R2C task)
        self.anticipation_head_reg = nn.Linear(512, num_classes)
        
    def extract_features(self, frames):
        """
        Extract spatial and temporal features from video frames.
        
        Args:
            frames: (batch, seq_len, channels, height, width)
        Returns:
            wsa_features: (batch, seq_len, 768)
            context_tokens: (batch, M, context_dim)
        """
        batch_size, seq_len, c, h, w = frames.shape
        
        # Extract spatial features
        frames_flat = frames.view(batch_size * seq_len, c, h, w)
        spatial_features = self.vision_encoder(frames_flat)  # (B*T, 768)
        spatial_features = spatial_features.view(batch_size, seq_len, 768)
        
        # Windowed self-attention
        wsa_features = self.wsa_encoder(spatial_features)  # (B, T, 768)
        
        # Compression and pooling
        context_tokens = self.pooling(wsa_features)  # (B, M, context_dim)
        context_tokens = self.context_proj(context_tokens)  # (B, M, 512)
        
        return wsa_features, context_tokens
    
    def forward(self, frames, current_class=None, future_minutes=None, num_future=30):
        """
        Forward pass for training.
        
        Args:
            frames: (batch, seq_len, channels, height, width)
            current_class: (batch,) current class for prior knowledge
            future_minutes: (N,) future minute indices
            num_future: number of future phases to predict
        Returns:
            recognition_logits: (batch, num_classes)
            anticipation_logits: (batch, N, num_classes)
        """
        batch_size = frames.shape[0]
        
        # Extract features
        wsa_features, context_tokens = self.extract_features(frames)
        
        # Recognition: use last WSA feature + last context token
        last_wsa = wsa_features[:, -1, :]  # (B, 768)
        last_context = context_tokens[:, -1, :]  # (B, 512)
        
        # For recognition, need to fuse with key-pooled features
        # Simplified: just use last context
        rec_input = torch.cat([last_wsa, torch.zeros(batch_size, 32, device=frames.device)], dim=1)
        recognition_logits = self.recognition_head(rec_input)  # (B, num_classes)
        
        # Anticipation
        if self.use_prior_knowledge and current_class is not None and future_minutes is not None:
            prior_emb = self.prior_embedding(current_class, future_minutes)  # (B, N, num_classes)
            anticipation_logits = self.decoder(context_tokens, prior_emb, num_future)
        else:
            anticipation_logits = self.decoder(context_tokens, None, num_future)
        
        return recognition_logits, anticipation_logits
    
    def predict_remaining_time(self, frames):
        """
        Predict remaining time until next occurrence of each phase (R2C task).
        
        Args:
            frames: (batch, seq_len, channels, height, width)
        Returns:
            remaining_times: (batch, num_classes) in minutes
        """
        _, context_tokens = self.extract_features(frames)
        
        # Use mean context tokens
        mean_context = torch.mean(context_tokens, dim=1)  # (B, 512)
        
        # Predict remaining times
        remaining_times = self.anticipation_head_reg(mean_context)  # (B, num_classes)
        
        return remaining_times


def fps():
    """
    Measure execution time and FPS for the SWAG model.
    
    Performs:
    - 5 warmup rounds
    - 20 measurement runs
    - Returns timing statistics in milliseconds
    
    Returns:
        dict: Dictionary containing timing statistics
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model - Single-Pass with prior knowledge (SP*)
    model = SWAG(
        img_size=248,
        num_classes=8,
        seq_length=1440,
        window_size=20,
        num_context_tokens=24,
        decoder_type='single_pass',
        use_prior_knowledge=True,
        max_horizon=60
    )
    model = model.to(device)
    model.eval()
    
    # Input: use sequence length = 16 frames for FPS tests
    batch_size = 1
    seq_len = 16
    dummy_frames = torch.randn(batch_size, seq_len, 3, 248, 248).to(device)
    dummy_class = torch.zeros(batch_size, dtype=torch.long).to(device)
    dummy_future = torch.arange(1, 31).to(device)  # 30 minutes
    
    # Warmup rounds
    print("Performing 5 warmup rounds...")
    with torch.no_grad():
        for i in range(5):
            _, _ = model(dummy_frames, dummy_class, dummy_future, num_future=30)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
    
    # Measurement runs
    print("Performing 20 measurement runs...")
    times = []
    num_runs = 20
    
    with torch.no_grad():
        for i in range(num_runs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            recognition_out, anticipation_out = model(dummy_frames, dummy_class, 
                                                      dummy_future, num_future=30)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end = time.perf_counter()
            elapsed_ms = (end - start) * 1000
            times.append(elapsed_ms)
    
    times = np.array(times)
    
    # Calculate statistics
    results = {
        'model_name': 'swag',
        'total_ms': float(np.mean(times)),
        'std_ms': float(np.std(times)),
        'min_ms': float(np.min(times)),
        'max_ms': float(np.max(times)),
        'median_ms': float(np.median(times)),
        'num_runs': num_runs,
        'num_warmup': 5,
        'input_shape': list(dummy_frames.shape),
        'output_recognition_shape': list(recognition_out.shape),
        'output_anticipation_shape': list(anticipation_out.shape),
        'device': str(device),
        'num_parameters': sum(p.numel() for p in model.parameters()),
        'num_trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'decoder_type': 'single_pass_with_prior',
        'anticipation_horizon': '30 minutes'
    }
    
    return results


def main():
    """Test the model and run benchmarks."""
    print("=" * 70)
    print("SWAG - Surgical Workflow Anticipative Generation")
    print("=" * 70)
    
    # Create model (SP* variant)
    model = SWAG(
        img_size=248,
        num_classes=8,
        seq_length=1440,
        window_size=20,
        num_context_tokens=24,
        decoder_type='single_pass',
        use_prior_knowledge=True,
        max_horizon=60
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Decoder type: Single-Pass with Prior Knowledge (SP*)")
    
    # Test forward pass
    print(f"\nTesting forward pass...")
    dummy_batch = torch.randn(2, 120, 3, 248, 248)  # 2 minutes
    dummy_class = torch.zeros(2, dtype=torch.long)
    dummy_future = torch.arange(1, 31)  # Predict 30 minutes
    
    model.eval()
    with torch.no_grad():
        recognition_out, anticipation_out = model(dummy_batch, dummy_class, 
                                                  dummy_future, num_future=30)
    
    print(f"  Input shape: {list(dummy_batch.shape)}")
    print(f"  Recognition output shape: {list(recognition_out.shape)}")
    print(f"  Anticipation output shape: {list(anticipation_out.shape)}")
    print(f"    - Recognition: predicts current phase (8 classes)")
    print(f"    - Anticipation: predicts 30 future phases at 60-second intervals")
    
    # Test regression task
    print(f"\nTesting remaining time prediction (R2C)...")
    with torch.no_grad():
        remaining_times = model.predict_remaining_time(dummy_batch)
    
    print(f"  Remaining times shape: {list(remaining_times.shape)}")
    print(f"    - Predicts remaining time until next occurrence of each phase")
    
    # Run FPS benchmark
    print(f"\n{'=' * 70}")
    print("Running Performance Benchmark")
    print(f"{'=' * 70}")
    
    results = fps()
    
    print(f"\nBenchmark Results:")
    print(f"  Model: {results['model_name']}")
    print(f"  Device: {results['device']}")
    print(f"  Decoder: {results['decoder_type']}")
    print(f"  Input shape: {results['input_shape']}")
    print(f"  Anticipation horizon: {results['anticipation_horizon']}")
    print(f"\nTiming Statistics ({results['num_runs']} runs):")
    print(f"  Average: {results['total_ms']:.2f} ms")
    print(f"  Std Dev: {results['std_ms']:.2f} ms")
    print(f"  Median:  {results['median_ms']:.2f} ms")
    print(f"  Min:     {results['min_ms']:.2f} ms")
    print(f"  Max:     {results['max_ms']:.2f} ms")
    print(f"\nParameters:")
    print(f"  Total: {results['num_parameters']:,}")
    print(f"  Trainable: {results['num_trainable_parameters']:,}")
    
    print(f"\n{'=' * 70}")
    print("Key Features:")
    print(f"  - Unifies phase recognition and long-term anticipation")
    print(f"  - Predicts up to 60 minutes into the future")
    print(f"  - Uses prior knowledge (class transition probabilities)")
    print(f"  - Single-pass generation for efficiency")
    print(f"  - Supports both classification and regression tasks")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()


def test_latency(warmup: int = 5, runs: int = 10, T: int = 16):
    """
    Standardized test helper for SWAG: run warmup then timed runs with temporal length T.
    Returns dict with mean_ms and std_ms.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SWAG(img_size=248, num_classes=8, seq_length=1440, window_size=20,
                 num_context_tokens=24, decoder_type='single_pass', use_prior_knowledge=True).to(device)
    model.eval()

    batch_size = 1
    seq_len = T
    dummy_frames = torch.randn(batch_size, seq_len, 3, 248, 248).to(device)
    dummy_class = torch.zeros(batch_size, dtype=torch.long).to(device)
    dummy_future = torch.arange(1, 31).to(device)

    try:
        with torch.no_grad():
            for _ in range(warmup):
                _, _ = model(dummy_frames, dummy_class, dummy_future, num_future=30)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

        times = []
        with torch.no_grad():
            for _ in range(runs):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start = time.perf_counter()
                recognition_out, anticipation_out = model(dummy_frames, dummy_class, dummy_future, num_future=30)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end = time.perf_counter()
                times.append((end - start) * 1000.0)

        arr = np.array(times)
        return {"mean_ms": float(np.mean(arr)), "std_ms": float(np.std(arr)), "device": str(device), "T": T}
    except Exception as e:
        return {"mean_ms": None, "std_ms": None, "error": str(e), "device": str(device), "T": T}
