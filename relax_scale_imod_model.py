#!/usr/bin/env python3
"""
relax_scale_imod_model.py - A program to scale IMOD model for volume of different voxel size
Written by Claude Sonnet 3.7, modified by Huy Bui, McGill 2025
"""

import argparse
import os

from util.imod import scale_imod_model

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Scale IMOD model coordinates by a specified factor.')
    parser.add_argument('--i', '--input', required=True, help='Input IMOD model file (.mod)')
    parser.add_argument('--o', '--output', required=True, help='Output IMOD model file (.mod)')
    parser.add_argument('--scale', type=float, required=True, help='Scaling factor for X, Y, Z coordinates')
        
    # Parse arguments
    args = parser.parse_args()
    
    # Validate input file exists
    if not os.path.exists(args.i):
        print(f"Error: Input file '{args.i}' does not exist.")
        return 1
    
    # Run the scaling function
    success = scale_imod_model(args.i, args.o, args.scale)
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())