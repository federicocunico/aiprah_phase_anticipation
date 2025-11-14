import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
from torchvision.models import resnet50


class GumbelSoftmax(nn.Module):
    """Gumbel-Softmax for differentiable discrete sampling"""
    def __init__(self, temperature=1.0, hard=False):
        super().__init__()
        self.temperature = temperature
        self.hard = hard
    
    def forward(self, logits):
        """
        Args:
            logits: (B, T, num_classes) - unnormalized log probabilities
        Returns:
            (B, T, num_classes) - sampled one-hot vectors (soft or hard)
        """
        if self.training:
            # Sample from Gumbel distribution
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
            y = logits + gumbel_noise
            y = F.softmax(y / self.temperature, dim=-1)
            
            if self.hard:
                # Straight-through estimator
                y_hard = torch.zeros_like(y)
                y_hard.scatter_(-1, y.argmax(dim=-1, keepdim=True), 1.0)
                y = (y_hard - y).detach() + y
            
            return y
        else:
            # During inference, just use argmax
            y = F.softmax(logits, dim=-1)
            return y


class GeneratorEncoder(nn.Module):
    """Encoder that processes past video frames and outputs hidden state"""
    def __init__(self, num_phases=7, hidden_dim=32, cnn_feature_dim=2048):
        super().__init__()
        
        # Visual encoder (ResNet50)
        resnet = resnet50(pretrained=False)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])  # Remove final FC
        
        # LSTM encoder
        self.lstm = nn.LSTM(cnn_feature_dim, hidden_dim, batch_first=True)
        
        # Phase prediction head
        self.phase_head = nn.Linear(hidden_dim, num_phases)
        
        self.hidden_dim = hidden_dim
        
    def forward(self, video_frames):
        """
        Args:
            video_frames: (B, T, C, H, W) - past video frames
        Returns:
            h: (B, T, hidden_dim) - hidden states
            phases: (B, T, num_phases) - phase logits
            final_hidden: tuple - final LSTM hidden state
        """
        B, T, C, H, W = video_frames.shape
        
        # Extract CNN features for all frames
        x = video_frames.view(B * T, C, H, W)
        cnn_features = self.cnn(x)  # (B*T, cnn_feature_dim, 1, 1)
        cnn_features = cnn_features.view(B, T, -1)  # (B, T, cnn_feature_dim)
        
        # LSTM encoding
        h, (h_n, c_n) = self.lstm(cnn_features)  # h: (B, T, hidden_dim)
        
        # Phase predictions
        phases = self.phase_head(h)  # (B, T, num_phases)
        
        return h, phases, (h_n, c_n)


class GeneratorDecoder(nn.Module):
    """Decoder that predicts future phase sequences"""
    def __init__(self, num_phases=7, hidden_dim=32, noise_dim=8):
        super().__init__()
        
        # Initial state projection
        self.fc_init = nn.Linear(hidden_dim + noise_dim, hidden_dim)
        
        # LSTM decoder
        self.lstm = nn.LSTM(num_phases, hidden_dim, batch_first=True)
        
        # Phase prediction head
        self.phase_head = nn.Linear(hidden_dim, num_phases)
        
        # Gumbel-Softmax for discrete sampling
        self.gumbel_softmax = GumbelSoftmax(temperature=1.0, hard=True)
        
        self.hidden_dim = hidden_dim
        self.noise_dim = noise_dim
        self.num_phases = num_phases
        
    def forward(self, encoder_final_hidden, future_length, temperature=1.0):
        """
        Args:
            encoder_final_hidden: tuple (h_n, c_n) - final encoder hidden state
                h_n: (1, B, hidden_dim)
                c_n: (1, B, hidden_dim)
            future_length: int - number of future timesteps to predict
            temperature: float - Gumbel-Softmax temperature
        Returns:
            phase_logits: (B, future_length, num_phases) - predicted phase logits
            phase_samples: (B, future_length, num_phases) - discrete phase samples
        """
        h_n, c_n = encoder_final_hidden
        B = h_n.shape[1]
        
        # Sample noise and concatenate with final hidden state
        noise = torch.randn(B, self.noise_dim, device=h_n.device)
        h_n_flat = h_n.squeeze(0)  # (B, hidden_dim)
        init_input = torch.cat([h_n_flat, noise], dim=1)  # (B, hidden_dim + noise_dim)
        
        # Initialize decoder hidden state
        h_0 = self.fc_init(init_input).unsqueeze(0)  # (1, B, hidden_dim)
        c_0 = c_n  # Use encoder's cell state
        
        # Initialize with zeros for first input
        y_t = torch.zeros(B, self.num_phases, device=h_n.device)
        
        phase_logits_list = []
        phase_samples_list = []
        
        h_t, c_t = h_0, c_0
        
        # Autoregressive decoding
        for t in range(future_length):
            # LSTM step
            y_t_input = y_t.unsqueeze(1)  # (B, 1, num_phases)
            lstm_out, (h_t, c_t) = self.lstm(y_t_input, (h_t, c_t))
            lstm_out = lstm_out.squeeze(1)  # (B, hidden_dim)
            
            # Predict phase logits
            logits = self.phase_head(lstm_out)  # (B, num_phases)
            phase_logits_list.append(logits)
            
            # Sample discrete phase using Gumbel-Softmax
            self.gumbel_softmax.temperature = temperature
            y_t = self.gumbel_softmax(logits)  # (B, num_phases)
            phase_samples_list.append(y_t)
        
        phase_logits = torch.stack(phase_logits_list, dim=1)  # (B, future_length, num_phases)
        phase_samples = torch.stack(phase_samples_list, dim=1)  # (B, future_length, num_phases)
        
        return phase_logits, phase_samples


class Discriminator(nn.Module):
    """Discriminator that distinguishes real from fake phase sequences"""
    def __init__(self, num_phases=7, hidden_dim=32):
        super().__init__()
        
        # Past encoder
        self.past_encoder = nn.LSTM(num_phases, hidden_dim, batch_first=True)
        
        # Future encoder
        self.future_encoder = nn.LSTM(num_phases, hidden_dim, batch_first=True)
        
        # Discriminator head
        self.disc_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, past_phases, future_phases):
        """
        Args:
            past_phases: (B, T_past, num_phases) - past phase sequence
            future_phases: (B, T_future, num_phases) - future phase sequence
        Returns:
            prediction: (B, 1) - probability of sequence being real
        """
        # Encode past
        _, (h_past, _) = self.past_encoder(past_phases)
        
        # Encode future
        _, (h_future, _) = self.future_encoder(future_phases)
        
        # Use future encoder's final hidden state for discrimination
        h_future = h_future.squeeze(0)  # (B, hidden_dim)
        
        # Discriminator prediction
        prediction = self.disc_head(h_future)  # (B, 1)
        
        return prediction


class SUPRGANGenerator(nn.Module):
    """Complete generator: encoder + decoder"""
    def __init__(self, num_phases=7, hidden_dim=32, noise_dim=8, cnn_feature_dim=2048):
        super().__init__()
        self.encoder = GeneratorEncoder(num_phases, hidden_dim, cnn_feature_dim)
        self.decoder = GeneratorDecoder(num_phases, hidden_dim, noise_dim)
        
    def forward(self, past_video, future_length=15, num_samples=1, temperature=1.0):
        """
        Args:
            past_video: (B, T_past, C, H, W) - past video frames
            future_length: int - number of future timesteps to predict
            num_samples: int - number of trajectory samples to generate
            temperature: float - Gumbel-Softmax temperature
        Returns:
            past_phases: (B, T_past, num_phases) - past phase predictions
            future_phase_logits: (B, num_samples, T_future, num_phases)
            future_phase_samples: (B, num_samples, T_future, num_phases)
        """
        B = past_video.shape[0]
        
        # Encode past
        h, past_phase_logits, encoder_hidden = self.encoder(past_video)
        past_phases = F.softmax(past_phase_logits, dim=-1)
        
        # Generate multiple future samples
        future_logits_list = []
        future_samples_list = []
        
        for _ in range(num_samples):
            logits, samples = self.decoder(encoder_hidden, future_length, temperature)
            future_logits_list.append(logits)
            future_samples_list.append(samples)
        
        future_phase_logits = torch.stack(future_logits_list, dim=1)
        future_phase_samples = torch.stack(future_samples_list, dim=1)
        
        return past_phases, future_phase_logits, future_phase_samples


class SUPRGAN(nn.Module):
    """
    SUPR-GAN: SUrgical PRediction GAN for Event Anticipation
    
    Complete model including generator and discriminator for surgical phase prediction
    """
    def __init__(self, num_phases=7, hidden_dim=32, noise_dim=8, 
                 cnn_feature_dim=2048):
        super().__init__()
        
        self.generator = SUPRGANGenerator(
            num_phases=num_phases,
            hidden_dim=hidden_dim,
            noise_dim=noise_dim,
            cnn_feature_dim=cnn_feature_dim
        )
        
        self.discriminator = Discriminator(
            num_phases=num_phases,
            hidden_dim=hidden_dim
        )
        
        self.num_phases = num_phases
        
    def forward(self, past_video, future_length=15, num_samples=10, temperature=1.0):
        """
        Args:
            past_video: (B, T_past, C, H, W) - past video frames
            future_length: int - number of future timesteps to predict
            num_samples: int - number of trajectory samples
            temperature: float - Gumbel-Softmax temperature
        Returns:
            Dictionary with:
                - past_phases: (B, T_past, num_phases)
                - future_logits: (B, num_samples, T_future, num_phases)
                - future_samples: (B, num_samples, T_future, num_phases)
        """
        past_phases, future_logits, future_samples = self.generator(
            past_video, future_length, num_samples, temperature
        )
        
        return {
            'past_phases': past_phases,
            'future_logits': future_logits,
            'future_samples': future_samples
        }
    
    def discriminate(self, past_phases, future_phases):
        """
        Args:
            past_phases: (B, T_past, num_phases)
            future_phases: (B, T_future, num_phases)
        Returns:
            (B, 1) - real/fake prediction
        """
        return self.discriminator(past_phases, future_phases)


def variety_loss(predictions, ground_truth):
    """
    Variety loss: minimum distance between ground truth and best prediction
    
    Args:
        predictions: (B, num_samples, T, num_classes) - predicted trajectories
        ground_truth: (B, T, num_classes) - ground truth trajectory
    Returns:
        loss: scalar
    """
    B, num_samples, T, num_classes = predictions.shape
    
    # Expand ground truth to compare with all samples
    gt_expanded = ground_truth.unsqueeze(1).expand_as(predictions)
    
    # Cross-entropy loss for each sample
    predictions_flat = predictions.reshape(B * num_samples * T, num_classes)
    gt_flat = gt_expanded.reshape(B * num_samples * T, num_classes)
    
    ce_loss = F.binary_cross_entropy_with_logits(predictions_flat, gt_flat, reduction='none')
    ce_loss = ce_loss.view(B, num_samples, T, num_classes).sum(dim=(2, 3))  # (B, num_samples)
    
    # Take minimum over samples
    min_loss = ce_loss.min(dim=1)[0]  # (B,)
    
    return min_loss.mean()


def fps():
    """
    Measure FPS (frames per second) performance of SUPR-GAN
    Returns dict with timing information in milliseconds
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    model = SUPRGAN(
        num_phases=7,
        hidden_dim=32,
        noise_dim=8,
        cnn_feature_dim=2048
    ).to(device)
    model.eval()
    
    # Create dummy inputs (use 16-frame sequences for FPS tests)
    B = 1
    T_past = 16
    T_future = 15  # prediction horizon (unchanged)
    C, H, W = 3, 224, 224
    
    past_video = torch.randn(B, T_past, C, H, W).to(device)
    
    # Warmup
    print("Warming up...")
    with torch.no_grad():
        for _ in range(5):
            _ = model(past_video, future_length=T_future, num_samples=10)
    
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
            _ = model(past_video, future_length=T_future, num_samples=10)
            
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
        'past_length': T_past,
        'future_length': T_future
    }
    
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("SUPR-GAN: SUrgical PRediction GAN")
    print("=" * 60)
    
    # Test model architecture
    print("\n1. Testing model architecture...")
    model = SUPRGAN(num_phases=7, hidden_dim=32, noise_dim=8)
    
    # Create dummy inputs
    B = 2
    T_past = 15
    T_future = 15
    past_video = torch.randn(B, T_past, 3, 224, 224)
    
    outputs = model(past_video, future_length=T_future, num_samples=10)
    
    print(f"   Input shape: {past_video.shape}")
    print(f"   Output shapes:")
    for key, val in outputs.items():
        print(f"     {key}: {val.shape}")
    
    # Test discriminator
    print("\n2. Testing discriminator...")
    past_phases = outputs['past_phases']
    future_sample = outputs['future_samples'][:, 0, :, :]  # Take first sample
    disc_output = model.discriminate(past_phases, future_sample)
    print(f"   Discriminator output shape: {disc_output.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    gen_params = sum(p.numel() for p in model.generator.parameters())
    disc_params = sum(p.numel() for p in model.discriminator.parameters())
    
    print(f"\n   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Generator parameters: {gen_params:,}")
    print(f"   Discriminator parameters: {disc_params:,}")
    
    # Measure FPS
    print("\n3. Measuring inference performance...")
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
    Standardized test helper for SUPR-GAN: run warmup then timed runs with temporal length T.
    Returns dict with mean_ms and std_ms.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SUPRGAN(num_phases=7, hidden_dim=32, noise_dim=8, cnn_feature_dim=2048).to(device)
    model.eval()

    B = 1
    T_past = T
    T_future = 15
    C, H, W = 3, 224, 224

    past_video = torch.randn(B, T_past, C, H, W).to(device)

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(past_video, future_length=T_future, num_samples=10)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(runs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(past_video, future_length=T_future, num_samples=10)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000.0)

    arr = np.array(times)
    return {"mean_ms": float(np.mean(arr)), "std_ms": float(np.std(arr)), "device": str(device), "T": T}