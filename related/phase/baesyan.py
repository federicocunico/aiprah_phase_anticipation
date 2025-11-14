import torch
import torch.nn as nn
import time
import numpy as np

class BayesianAlexNetLSTM(nn.Module):
    """
    Bayesian AlexNet-LSTM for surgical instrument anticipation.
    
    Architecture from: "Rethinking Anticipation Tasks: Uncertainty-aware 
    Anticipation of Sparse Surgical Instrument Usage for Context-aware Assistance"
    
    Key features:
    - AlexNet-style CNN backbone with dropout (0.2) after each layer
    - LSTM (512 hidden units) with dropout on input and hidden state
    - Outputs: K regression values + 3K classification values
    - Input: (batch, seq_len, 3, 216, 384)
    """
    
    def __init__(self, num_instruments=5, dropout_rate=0.2):
        super(BayesianAlexNetLSTM, self).__init__()
        
        self.num_instruments = num_instruments
        self.dropout_rate = dropout_rate
        
        # AlexNet-style CNN with Bayesian layers (dropout after each layer)
        self.features = nn.Sequential(
            # Conv1: 64 filters, 11x11, stride 4
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Conv2: 192 filters, 5x5, stride 1
            nn.Conv2d(64, 192, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Conv3: 384 filters, 3x3, stride 1
            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            
            # Conv4: 256 filters, 3x3, stride 1
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            
            # Conv5: 256 filters, 3x3, stride 1
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Adaptive pooling to 6x6
            nn.AdaptiveAvgPool2d((6, 6))
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.relu_fc1 = nn.ReLU(inplace=True)
        self.dropout_fc1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(4096, 4096)
        self.relu_fc2 = nn.ReLU(inplace=True)
        self.dropout_fc2 = nn.Dropout(dropout_rate)
        
        # LSTM with dropout on input and hidden state
        self.lstm = nn.LSTM(
            input_size=4096,
            hidden_size=512,
            num_layers=1,
            batch_first=True,
            dropout=0  # We'll apply dropout manually
        )
        self.dropout_lstm = nn.Dropout(dropout_rate)
        
        # Output layer: K regression + 3K classification outputs
        # K instruments, each with: 1 regression + 3 classification (anticipating, present, background)
        self.fc_out = nn.Linear(512, num_instruments + 3 * num_instruments)
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch, seq_len, channels, height, width)
               Default: (batch, 128, 3, 216, 384)
        
        Returns:
            Output tensor of shape (batch, seq_len, K + 3K)
            where K = num_instruments (default 5)
            Total outputs: 5 regression + 15 classification = 20
        """
        batch_size, seq_len, c, h, w = x.size()
        
        # Process each frame through CNN
        cnn_features = []
        for t in range(seq_len):
            frame = x[:, t, :, :, :]  # (batch, 3, 216, 384)
            
            # CNN forward pass
            feat = self.features(frame)  # (batch, 256, 6, 6)
            feat = feat.view(batch_size, -1)  # (batch, 256*6*6=9216)
            
            # FC layers
            feat = self.fc1(feat)
            feat = self.relu_fc1(feat)
            feat = self.dropout_fc1(feat)
            
            feat = self.fc2(feat)
            feat = self.relu_fc2(feat)
            feat = self.dropout_fc2(feat)  # (batch, 4096)
            
            cnn_features.append(feat)
        
        # Stack features for LSTM
        cnn_features = torch.stack(cnn_features, dim=1)  # (batch, seq_len, 4096)
        
        # Apply dropout to LSTM input
        cnn_features = self.dropout_lstm(cnn_features)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(cnn_features)  # (batch, seq_len, 512)
        
        # Apply dropout to LSTM output
        lstm_out = self.dropout_lstm(lstm_out)
        
        # Final output layer
        output = self.fc_out(lstm_out)  # (batch, seq_len, K + 3K)
        
        return output


def fps():
    """
    Measure execution time and FPS for the Bayesian AlexNet-LSTM model.
    
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
    model = BayesianAlexNetLSTM(num_instruments=5, dropout_rate=0.2)
    model = model.to(device)
    model.eval()
    
    # Input for FPS: use sequence length = 16 frames (batch dimension still 1)
    dummy_input = torch.randn(1, 16, 3, 216, 384).to(device)
    
    # Warmup rounds
    print("Performing 5 warmup rounds...")
    with torch.no_grad():
        for i in range(5):
            _ = model(dummy_input)
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
            output = model(dummy_input)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end = time.perf_counter()
            elapsed_ms = (end - start) * 1000
            times.append(elapsed_ms)
    
    times = np.array(times)
    
    # Calculate statistics
    results = {
        'model_name': 'baesyan',
        'total_ms': float(np.mean(times)),
        'std_ms': float(np.std(times)),
        'min_ms': float(np.min(times)),
        'max_ms': float(np.max(times)),
        'median_ms': float(np.median(times)),
        'num_runs': num_runs,
        'num_warmup': 5,
        'input_shape': list(dummy_input.shape),
        'output_shape': list(output.shape),
        'device': str(device),
        'num_parameters': sum(p.numel() for p in model.parameters()),
        'num_trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
    }
    
    return results


def main():
    """Test the model and run benchmarks."""
    print("=" * 70)
    print("BayesianAlexNetLSTM - Surgical Instrument Anticipation Model")
    print("=" * 70)
    
    # Create model
    model = BayesianAlexNetLSTM(num_instruments=5, dropout_rate=0.2)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Test with dummy input
    print(f"\nTesting forward pass...")
    dummy_input = torch.randn(1, 128, 3, 216, 384)
    
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"  Input shape: {list(dummy_input.shape)}")
    print(f"  Output shape: {list(output.shape)}")
    print(f"  Expected output: (1, 128, 20)")
    print(f"    - 5 regression values (remaining time for 5 instruments)")
    print(f"    - 15 classification values (3 classes × 5 instruments)")
    
    # Run FPS benchmark
    print(f"\n{'=' * 70}")
    print("Running Performance Benchmark")
    print(f"{'=' * 70}")
    
    results = fps()
    
    print(f"\nBenchmark Results:")
    print(f"  Model: {results['model_name']}")
    print(f"  Device: {results['device']}")
    print(f"  Input shape: {results['input_shape']}")
    print(f"  Output shape: {results['output_shape']}")
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


if __name__ == "__main__":
    main()


def test_latency(warmup: int = 5, runs: int = 10, T: int = 16):
    """
    Standardized test helper: run `warmup` warmup inferences then `runs` timed runs
    using a temporal length T. Returns a dict with mean_ms and std_ms.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BayesianAlexNetLSTM(num_instruments=5, dropout_rate=0.2).to(device)
    model.eval()

    dummy_input = torch.randn(1, T, 3, 216, 384).to(device)

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(runs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(dummy_input)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000.0)

    arr = np.array(times)
    return {"mean_ms": float(np.mean(arr)), "std_ms": float(np.std(arr)), "device": str(device), "T": T}