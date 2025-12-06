#!/usr/bin/env python3
"""
Example usage of POUQ Python bindings
"""

import numpy as np
import pouq


def main():
    print("POUQ Python Bindings Example")
    print("=" * 40)
    
    # Generate sample data
    dim = 128
    n_samples = 1000
    print(f"\nGenerating {n_samples} samples with dimension {dim}...")
    data = np.random.randn(n_samples, dim).astype(np.float32)
    print(f"Data shape: {data.shape}")
    print(f"Data dtype: {data.dtype}")
    
    # Train the model
    print("\nTraining POUQ model...")
    try:
        code, rec_para = pouq.train(dim, data)
        print(f"Code shape: {code.shape}")
        print(f"RecPara shape: {rec_para.shape}")
        print("Training completed successfully!")
    except Exception as e:
        print(f"Error during training: {e}")
        return
    
    # Test distance calculation
    print("\nTesting distance calculation...")
    try:
        query = np.random.randn(dim).astype(np.float32)
        dist = pouq.distance(dim, code, rec_para, query.flatten())
        print(f"Distance: {dist:.6f}")
    except Exception as e:
        print(f"Error during distance calculation: {e}")
    
    # Test utility functions
    print("\nTesting utility functions...")
    try:
        # Get sorted data
        sorted_data = pouq.get_sorted_data(data, d=0, dim=dim)
        print(f"Sorted data shape: {sorted_data.shape}")
        
        # Count frequency
        data_map, freq_map = pouq.count_freq(sorted_data)
        print(f"Data map size: {len(data_map)}")
        print(f"Freq map size: {len(freq_map)}")
        
        # Segment
        lowers, uppers = pouq.segment(data_map, freq_map, do_count_freq=True)
        print(f"Segments: {len(lowers)}")
        
        # Optimize
        if len(data_map) > 0:
            bound = pouq.optimize(
                init_lower=float(np.min(data_map)),
                init_upper=float(np.max(data_map)),
                data_map=data_map,
                freq_map=freq_map,
                do_count_freq=True,
            )
            print(f"Optimized bounds: [{bound.lower:.6f}, {bound.upper:.6f}]")
    except Exception as e:
        print(f"Error during utility function testing: {e}")
    
    print("\nExample completed!")


if __name__ == "__main__":
    main()

