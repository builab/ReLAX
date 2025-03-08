#!/usr/bin/env python
# relax_generate_ref2d.py

from util.subtomo import average_z_sections
from util.align import save_mrc

import argparse
import numpy as np
import mrcfile
import os



def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Generate 2D references from 3D MRC files')
    parser.add_argument('--i', required=True, help='Input 3D MRC file')
    parser.add_argument('--o', required=True, help='Output 2D MRC file')
    parser.add_argument('--z_slices', type=int, default=10, help='Number of Z slices to average')
   
    # Parse arguments
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.i):
        print(f"Error: Input file {args.i} does not exist!")
        return 1
    
    # Read the input MRC file
    try:
        print(f"Reading input file: {args.i}")
        with mrcfile.open(args.i) as mrc:
            volume_data = mrc.data
            print(f"Input dimensions: {volume_data.shape}")
    except Exception as e:
        print(f"Error reading MRC file: {e}")
        return 1
    
    # Generate 2D projection by averaging z-slices
    print(f"Generating 2D projection by averaging {args.z_slices} z-slices...")
    projection = average_z_sections([volume_data], args.z_slices)
    print(f"Output dimensions: {projection[0].shape}")
    
    # Save the result
    try:
        save_mrc(projection[0], args.o)
    except Exception as e:
        print(f"Error saving output MRC file: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())