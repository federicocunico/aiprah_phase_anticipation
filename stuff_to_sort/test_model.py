import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

# Dataset imports
from datasets.cholec80 import Cholec80Dataset
from datasets.heichole import HeiCholeDataset

# Model import
from models.temporal_model_swin import TemporalAnticipationModel, create_model


def test_dataset_forward_pass(
    dataset_name="cholec80",
    batch_size=6,
    seq_len=10,
    target_fps=1,
    use_rgbd=False,
    temporal_modeling="separate",
    device="cuda"
):
    """
    Simple test to verify dataset loading and model forward pass.
    
    Args:
        dataset_name: "cholec80" or "heichole"
        batch_size: Batch size for DataLoader
        seq_len: Sequence length for temporal windows
        target_fps: Target FPS for video subsampling
        use_rgbd: Whether to use RGB-D data
        temporal_modeling: Temporal modeling strategy
        device: Device to run on ("cuda" or "cpu")
    """
    print("="*60)
    print(f"Testing Dataset: {dataset_name}")
    print(f"Configuration: batch_size={batch_size}, seq_len={seq_len}, fps={target_fps}")
    print(f"RGB-D: {use_rgbd}, Temporal Mode: {temporal_modeling}")
    print("="*60)
    
    # Select dataset
    if dataset_name == "cholec80":
        data_dir = "./data/cholec80"
        DatasetClass = Cholec80Dataset
    elif dataset_name == "heichole":
        data_dir = "./data/heichole"
        DatasetClass = HeiCholeDataset
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Create dataset with mode="all" to load entire dataset
    print("\nLoading dataset...")
    dataset = DatasetClass(
        root_dir=data_dir,
        mode="all",  # Load all videos
        seq_len=seq_len,
        fps=target_fps
    )
    print(f"Total samples in dataset: {len(dataset)}")
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle for testing
        num_workers=4,
        pin_memory=True
    )
    print(f"Total batches: {len(dataloader)}")
    
    # Create model
    in_channels = 4 if use_rgbd else 3
    print(f"\nCreating model with {in_channels} input channels...")
    model = create_model(
        sequence_length=seq_len,
        num_classes=7,
        time_horizon=5,
        in_channels=in_channels,
        use_depth_enhancer=(use_rgbd),
        temporal_modeling=temporal_modeling
    )
    
    # Move model to device
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    print(f"Device: {device}")
    
    # Test forward pass on entire dataset
    print("\nRunning forward pass on entire dataset...")
    
    successful_batches = 0
    failed_batches = 0
    total_time = 0
    
    # Store some outputs for verification
    all_outputs = []
    all_phase_labels = []
    all_time_targets = []

    for batch_idx, (frames, metadata) in enumerate(tqdm(dataloader, desc="Processing batches")):
        a=1

    if False:
        with torch.no_grad():
            for batch_idx, (frames, metadata) in enumerate(tqdm(dataloader, desc="Processing batches")):
                try:
                    # Get appropriate input based on configuration
                    if use_rgbd:
                        if 'frames_rgbd' in metadata:
                            input_data = metadata['frames_rgbd'].to(device)
                        else:
                            # Fallback to RGB if RGB-D not available
                            print(f"\nWarning: Batch {batch_idx} - RGB-D requested but not available")
                            input_data = frames.to(device)
                    else:
                        input_data = frames.to(device)
                    
                    # Get targets for comparison
                    phase_labels = metadata['phase_label']
                    time_targets = metadata['time_to_next_phase']
                    
                    # Forward pass
                    start_time = time.time()
                    outputs = model(input_data)
                    end_time = time.time()
                    
                    total_time += (end_time - start_time)
                    
                    # Store outputs
                    all_outputs.append(outputs.cpu())
                    all_phase_labels.append(phase_labels)
                    all_time_targets.append(time_targets)
                    
                    successful_batches += 1
                    
                    # Print info for first batch
                    if batch_idx == 0:
                        print(f"\nFirst batch info:")
                        print(f"  Input shape: {input_data.shape}")
                        print(f"  Output shape: {outputs.shape}")
                        print(f"  Output range: [{outputs.min().item():.3f}, {outputs.max().item():.3f}]")
                        print(f"  Phase labels shape: {phase_labels.shape}")
                        print(f"  Time targets shape: {time_targets.shape}")
                    
                except Exception as e:
                    failed_batches += 1
                    print(f"\nError in batch {batch_idx}: {str(e)}")
                    if failed_batches > 5:
                        print("Too many failures, stopping test...")
                        break
        
        # Print summary
        print("\n" + "="*60)
        print("Test Summary:")
        print("="*60)
        print(f"Successful batches: {successful_batches}/{len(dataloader)}")
        print(f"Failed batches: {failed_batches}")
        print(f"Total forward pass time: {total_time:.2f}s")
        print(f"Average time per batch: {total_time/successful_batches:.3f}s")
        print(f"Average time per sample: {total_time/(successful_batches*batch_size):.3f}s")
        
        # Concatenate all outputs
        if all_outputs:
            all_outputs = torch.cat(all_outputs, dim=0)
            all_phase_labels = torch.cat(all_phase_labels, dim=0)
            all_time_targets = torch.cat(all_time_targets, dim=0)
            
            print(f"\nOverall statistics:")
            print(f"Total samples processed: {all_outputs.shape[0]}")
            print(f"Output value range: [{all_outputs.min():.3f}, {all_outputs.max():.3f}]")
            print(f"Mean output: {all_outputs.mean():.3f}")
            print(f"Std output: {all_outputs.std():.3f}")
            
            # Check output validity
            if torch.isnan(all_outputs).any():
                print("WARNING: NaN values detected in outputs!")
            if torch.isinf(all_outputs).any():
                print("WARNING: Inf values detected in outputs!")
        
        if successful_batches == len(dataloader):
            print("\n✅ All batches processed successfully!")
        else:
            print(f"\n⚠️ {failed_batches} batches failed")
    
        return successful_batches == len(dataloader)


def main():
    """Run tests with different configurations."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test dataset and model forward pass")
    parser.add_argument("--dataset", type=str, default="cholec80", 
                        choices=["cholec80", "heichole"],
                        help="Dataset to test")
    parser.add_argument("--batch_size", type=int, default=6,
                        help="Batch size for testing")
    parser.add_argument("--seq_len", type=int, default=30,
                        help="Sequence length")
    parser.add_argument("--fps", type=int, default=1,
                        help="Target FPS")
    parser.add_argument("--use_rgbd", action="store_true",
                        help="Use RGB-D data")
    parser.add_argument("--temporal_mode", type=str, default="separate",
                        choices=["separate", "3d_conv", "sequential"],
                        help="Temporal modeling mode")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"],
                        help="Device to use")
    parser.add_argument("--test_all_modes", action="store_true",
                        help="Test all temporal modeling modes")
    
    args = parser.parse_args()
    
    if args.test_all_modes:
        # Test all temporal modeling modes
        modes = ["separate", "3d_conv", "sequential"]
        results = {}
        
        for mode in modes:
            print(f"\n\n{'='*60}")
            print(f"TESTING MODE: {mode}")
            print(f"{'='*60}")
            
            success = test_dataset_forward_pass(
                dataset_name=args.dataset,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                target_fps=args.fps,
                use_rgbd=args.use_rgbd,
                temporal_modeling=mode,
                device=args.device
            )
            results[mode] = success
        
        # Print final summary
        print(f"\n\n{'='*60}")
        print("FINAL SUMMARY")
        print(f"{'='*60}")
        for mode, success in results.items():
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"{mode:.<20} {status}")
    else:
        # Test single configuration
        test_dataset_forward_pass(
            dataset_name=args.dataset,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            target_fps=args.fps,
            use_rgbd=args.use_rgbd,
            temporal_modeling=args.temporal_mode,
            device=args.device
        )


if __name__ == "__main__":
    main()