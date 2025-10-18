"""
Test script for network architecture visualizer.

Usage:
    python test_visualizer.py --config config_diffusion_128_v1.yaml
"""

import argparse
import yaml
import torch
from torch.utils.data import TensorDataset, DataLoader

from model_factory import build_diffusion_model
from utils.network_visualizer import NetworkArchitectureVisualizer


def parse_args():
    parser = argparse.ArgumentParser(description="Test network visualizer")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    args = parse_args()
    config = load_config(args.config)

    print("=" * 70)
    print("Testing Network Architecture Visualizer")
    print("=" * 70)

    # Create model
    print("\n1. Creating model...")
    model_config = config["model"]
    model = build_diffusion_model(model_config, verbose=True)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {num_params:,}")

    # Create dummy dataset and dataloader
    print("\n2. Creating dummy dataloader...")
    # Simulate latent shape: [B, C, D, H, W]
    # Use smaller spatial dims to avoid OOM during testing
    dummy_ld_latent = torch.randn(4, 8, 25, 16, 16)
    dummy_hd_latent = torch.randn(4, 8, 25, 16, 16)

    dataset = TensorDataset(dummy_ld_latent, dummy_hd_latent)

    # Create a simple dataset wrapper to match expected format
    class SimpleDataset:
        def __init__(self, ld_latent, hd_latent):
            self.ld_latent = ld_latent
            self.hd_latent = hd_latent

        def __len__(self):
            return len(self.ld_latent)

        def __getitem__(self, idx):
            return {
                "ld_latent": self.ld_latent[idx],
                "hd_latent": self.hd_latent[idx],
            }

    dataset = SimpleDataset(dummy_ld_latent, dummy_hd_latent)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    print(f"   Dataset size: {len(dataset)}")
    print(f"   Batch size: {2}")

    # Test visualizer
    print("\n3. Running visualizer...")
    # Use CPU to avoid OOM on busy GPU
    device = torch.device("cpu")
    print(f"   Device: {device}")

    model = model.to(device)

    visualizer = NetworkArchitectureVisualizer(model, config)

    try:
        diagram_path = visualizer.trace_and_visualize(
            train_loader=dataloader,
            device=device,
        )

        print(f"\n✓ Success!")
        print(f"   Diagram saved to: {diagram_path}")

        # Check if mermaid file was created
        import os
        mermaid_path = diagram_path.replace('.png', '.mmd').replace('.html', '.mmd').replace('.svg', '.mmd')
        if os.path.exists(mermaid_path):
            print(f"   Mermaid code saved to: {mermaid_path}")

            # Display first few lines
            with open(mermaid_path) as f:
                lines = f.readlines()
            print(f"\n   First 10 lines of Mermaid diagram:")
            for i, line in enumerate(lines[:10]):
                print(f"      {line.rstrip()}")

    except Exception as e:
        print(f"\n✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("\n" + "=" * 70)
    print("Test completed successfully!")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    exit(main())
