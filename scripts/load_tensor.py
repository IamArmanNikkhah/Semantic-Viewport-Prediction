#!/usr/bin/env python3
"""
Test script to load and print the contents of a given PyTorch tensor file generated from build_detections.py

Usage:
    python load_tensor.py <path_to_tensor_file>
"""

import argparse
import sys
import torch
from semantics.detection_type import Detection


def main():
    parser = argparse.ArgumentParser(
        description="Load and print the contents of a PyTorch tensor file"
    )
    parser.add_argument(
        "tensor_file",
        type=str,
        help="Path to the PyTorch tensor file (.pt or .pth)",
    )
    
    args = parser.parse_args()
    
    try:
        # Allow loading custom classes from the project
        with torch.serialization.safe_globals([Detection]):
            # Load the tensor file
            tensor = torch.load(args.tensor_file, weights_only=False)
        
        # Print the contents
        print(f"Loaded tensor from: {args.tensor_file}")
        print(f"Type: {type(tensor)}")
        
        if isinstance(tensor, torch.Tensor):
            print(f"Shape: {tensor.shape}")
            print(f"Dtype: {tensor.dtype}")
            print(f"Device: {tensor.device}")
            print("\nContents:")
            print(tensor)
        elif isinstance(tensor, dict):
            print(f"Dictionary with keys: {list(tensor.keys())}")
            print("\nContents:")
            for key, value in tensor.items():
                print(f"\n{key}:")
                if isinstance(value, torch.Tensor):
                    print(f"  Shape: {value.shape}, Dtype: {value.dtype}")
                    print(f"  {value}")
                else:
                    print(f"  {value}")
        else:
            print("\nContents:")
            print(tensor)
            
    except FileNotFoundError:
        print(f"Error: File not found: {args.tensor_file}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error loading tensor file: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
