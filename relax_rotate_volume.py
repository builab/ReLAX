#!/usr/bin/env python
"""
relax_rotate_volume.py - A program to rotate MRC volumes using Euler angles in ZYZ convention.
Written by Claude Sonnet 3.7, modified by Huy Bui, McGill 2025
"""

from util.subtomo import rotate_subtomogram_zyz

import argparse
import mrcfile
import numpy as np



def main():
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description='Rotate MRC volume using Euler angles (ZYZ convention)')
    parser.add_argument('--i', required=True, help='Input MRC file')
    parser.add_argument('--o', required=True, help='Output MRC file')
    parser.add_argument('--rot', type=float, default=0, help='Rotation angle alpha around Z axis (degrees)')
    parser.add_argument('--tilt', type=float, default=0, help='Rotation angle beta around Y axis (degrees)')
    parser.add_argument('--psi', type=float, default=0, help='Rotation angle gamma around Z axis (degrees)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Print parameters
    print(f"Input file: {args.i}")
    print(f"Output file: {args.o}")
    print(f"Rotation angles (ZYZ Euler convention):")
    print(f"  rot (alpha): {args.rot} degrees")
    print(f"  tilt (beta): {args.tilt} degrees")
    print(f"  psi (gamma): {args.psi} degrees")
    
    try:
        # Open input MRC file
        with mrcfile.open(args.i, mode='r') as mrc:
            # Get data and metadata
            data = mrc.data.copy()
            voxel_size = mrc.voxel_size
            origin = mrc.header.origin
            
            # Display volume information
            print(f"Volume dimensions: {data.shape}")
            print(f"Voxel size: {voxel_size}")
            
            # Apply rotation
            print("Applying rotation...")
            rotated_data = rotate_subtomogram_zyz(data, args.rot, args.tilt, args.psi)
            
            # Create output MRC file
            with mrcfile.new(args.o, overwrite=True) as mrc_out:
                mrc_out.set_data(rotated_data)
                
                # Copy metadata from input
                if voxel_size is not None:
                    mrc_out.voxel_size = voxel_size
                
                if origin is not None:
                    mrc_out.header.origin = origin
                
                # Add note about rotation
                mrc_out.header.nlabl = 1
                mrc_out.header.label[0] = f"Rotated by ZYZ angles: {args.rot}, {args.tilt}, {args.psi}".encode()
            
            print(f"Rotated volume saved to {args.o}")
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
