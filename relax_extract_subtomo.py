#!/usr/bin/env python
"""
relax_extract_subtomo.py - A simple program to extract subtomo. Use as a tool for testing
Intentionally written with duplicate function as util.subtomo so it can function as a standalone.
Written by Claude Sonnet 3.7, modified by Huy Bui, McGill 2025
"""

import argparse
import mrcfile
import numpy as np

def extract_subtomogram(tomogram_path, center, size, output_path):
    """
    Extract a single subtomogram of given size centered at the provided coordinate.
    
    Parameters:
    -----------
    tomogram_path : str
        Path to the MRC file containing the tomogram.
    center : tuple of int
        (x, y, z) coordinates of the center of the subtomogram.
    size : list of int
        Size of the subtomogram as [size_x, size_y, size_z].
    output_path : str
        Path to save the extracted subtomogram as an MRC file.
    """
    # Check if the size is even
    if any(x % 2 != 0 for x in size):
        raise ValueError("Subtomogram size must be even in all dimensions.")
    
    with mrcfile.open(tomogram_path, permissive=True) as mrc:
        tomogram = mrc.data
        dim_z, dim_y, dim_x = tomogram.shape
        print(f"Tomogram dimensions: {dim_x}, {dim_y}, {dim_z}")
        
        center_x, center_y, center_z = center
        size_x, size_y, size_z = size
        
        half_x, half_y, half_z = size_x // 2, size_y // 2, size_z // 2
        
        # Define the extraction boundaries
        x_start, x_end = max(0, center_x - half_x), min(dim_x, center_x + half_x)
        y_start, y_end = max(0, center_y - half_y), min(dim_y, center_y + half_y)
        z_start, z_end = max(0, center_z - half_z), min(dim_z, center_z + half_z)
        
        subtomo = tomogram[z_start:z_end, y_start:y_end, x_start:x_end]
        
        # Handle edge cases with padding if near the edges
        if subtomo.shape != (size_z, size_y, size_x):
            padded_subtomo = np.zeros((size_z, size_y, size_x), dtype=subtomo.dtype)
            
            pad_x_start = half_x - (center_x - x_start)
            pad_y_start = half_y - (center_y - y_start)
            pad_z_start = half_z - (center_z - z_start)
            
            padded_subtomo[
                pad_z_start:pad_z_start + subtomo.shape[0],
                pad_y_start:pad_y_start + subtomo.shape[1],
                pad_x_start:pad_x_start + subtomo.shape[2]
            ] = subtomo
            
            subtomo = padded_subtomo
        
        # Save the extracted subtomogram
        with mrcfile.new(output_path, overwrite=True) as output_mrc:
            output_mrc.set_data(subtomo.astype(np.float32))
        
        print(f"Subtomogram saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract a single subtomogram from a tomogram.")
    parser.add_argument("--i", required=True, help="Input tomogram file (MRC format).")
    parser.add_argument("--center", required=True, help="Center coordinates (x,y,z) as comma-separated values.")
    parser.add_argument("--box_size", required=True, help="Box size (size_x,size_y,size_z) as comma-separated values.")
    parser.add_argument("--o", required=True, help="Output subtomogram file (MRC format).")
    
    args = parser.parse_args()
    
    center = tuple(map(int, args.center.split(",")))
    size = list(map(int, args.box_size.split(",")))
    
    extract_subtomogram(args.i, center, size, args.o)
