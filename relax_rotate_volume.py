#!/usr/bin/env python
"""
relax_rotate_volume.py - A program to rotate MRC volumes using Euler angles in ZYZ convention
with a single interpolation step, respecting MRC axis convention.
Tested to be the same as Relion convention
Intentionally written with duplicate function as util.subtomo so it can function as a standalone.
"""

import argparse
import mrcfile
import numpy as np
from scipy.ndimage import affine_transform

def rotate_subtomogram_zyz_single_interpolation(subtomo, alpha, beta, gamma):
    """
    Rotate a subtomogram using Euler angles in ZYZ convention
    using a single interpolation step via a combined rotation matrix.
    Specifically handles MRC convention where Z is axis 0.
    
    Parameters:
    -----------
    subtomo : numpy.ndarray
        3D subtomogram array in MRC convention (Z, Y, X) order
    alpha : float
        First rotation angle around Z axis (in degrees)
    beta : float
        Second rotation angle around Y axis (in degrees)
    gamma : float
        Third rotation angle around Z axis (in degrees)
        
    Returns:
    --------
    rotated_subtomo : numpy.ndarray
        Rotated subtomogram
    """
    # Convert to float32 to avoid data type issues with affine_transform
    # This handles float16 input arrays that scipy can't process directly
    subtomo_float32 = subtomo.astype(np.float32)
    
    # Convert angles to radians
    alpha_rad = np.deg2rad(alpha)
    beta_rad = np.deg2rad(beta)
    gamma_rad = np.deg2rad(gamma)
    
    # Create rotation matrices for each axis
    # Remember that for MRC, the axes are: Z=0, Y=1, X=2
    # But scipy's affine_transform works in reversed order (X,Y,Z)
    
    # Build the rotation matrices for the ZYZ convention
    # First rotation around Z axis (alpha)
    Rz_alpha = np.array([
        [np.cos(alpha_rad), -np.sin(alpha_rad), 0],
        [np.sin(alpha_rad), np.cos(alpha_rad), 0],
        [0, 0, 1]
    ])
    
    # Second rotation around Y axis (beta)
    Ry_beta = np.array([
        [np.cos(beta_rad), 0, np.sin(beta_rad)],
        [0, 1, 0],
        [-np.sin(beta_rad), 0, np.cos(beta_rad)]
    ])
    
    # Third rotation around Z axis (gamma)
    Rz_gamma = np.array([
        [np.cos(gamma_rad), -np.sin(gamma_rad), 0],
        [np.sin(gamma_rad), np.cos(gamma_rad), 0],
        [0, 0, 1]
    ])
    
    # Calculate the combined rotation matrix (R = Rz(alpha) * Ry(beta) * Rz(gamma))
    R = np.dot(Rz_alpha, np.dot(Ry_beta, Rz_gamma))
    
    # For MRC convention (Z,Y,X), we need to swap axes for affine_transform
    # which expects (X,Y,Z), so we reverse the rotation matrix
    R_mrc = R[::-1, ::-1]
    
    # Get the center of the volume
    center = np.array(subtomo.shape) // 2
    
    # For MRC convention, we reverse the center coordinates
    center_mrc = center[::-1]
    
    # Calculate the offset for the affine transform
    offset = center_mrc - np.dot(R_mrc, center_mrc)
    
    # Apply the rotation with a single interpolation
    rotated = affine_transform(
        subtomo_float32,
        matrix=R_mrc,
        offset=offset,
        order=3,  # cubic interpolation
        mode='constant',
        cval=0.0
    )
    
    return rotated

def main():
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description='Rotate MRC volume using Euler angles (ZYZ convention)')
    parser.add_argument('--i', required=True, help='Input MRC file')
    parser.add_argument('--o', required=True, help='Output MRC file')
    parser.add_argument('--rot', type=float, default=0, help='Rotation angle alpha around Z axis (degrees)')
    parser.add_argument('--tilt', type=float, default=0, help='Rotation angle beta around Y axis (degrees)')
    parser.add_argument('--psi', type=float, default=0, help='Rotation angle gamma around Z axis (degrees)')
    parser.add_argument('--order', type=int, default=3, choices=[0, 1, 2, 3, 4, 5], 
                        help='Interpolation order (0=nearest, 1=linear, 3=cubic)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Print parameters
    print(f"Input file: {args.i}")
    print(f"Output file: {args.o}")
    print(f"Rotation angles (ZYZ Euler convention):")
    print(f"  rot (alpha): {args.rot} degrees")
    print(f"  tilt (beta): {args.tilt} degrees")
    print(f"  psi (gamma): {args.psi} degrees")
    print(f"Interpolation order: {args.order}")
    
    try:
        # Open input MRC file
        with mrcfile.open(args.i, mode='r') as mrc:
            # Get data and metadata
            data = mrc.data.copy()
            voxel_size = mrc.voxel_size
            origin = mrc.header.origin
            
            # Display volume information
            print(f"Volume dimensions: {data.shape} (Z, Y, X)")
            print(f"Voxel size: {voxel_size}")
            
            # Apply rotation
            print("Applying rotation with single interpolation...")
            rotated_data = rotate_subtomogram_zyz_single_interpolation(
                data, 
                args.rot, 
                args.tilt, 
                args.psi
            )
            
            # Create output MRC file
            with mrcfile.new(args.o, overwrite=True) as mrc_out:
                mrc_out.set_data(rotated_data.astype(data.dtype))
                
                # Copy metadata from input
                if voxel_size is not None:
                    mrc_out.voxel_size = voxel_size
                
                if origin is not None:
                    mrc_out.header.origin = origin
                
                # Add note about rotation
                mrc_out.header.nlabl = 1
                rotation_note = f"Rotated by ZYZ angles: {args.rot}, {args.tilt}, {args.psi} (single interpolation)"
                mrc_out.header.label[0] = rotation_note.encode()
            
            print(f"Rotated volume saved to {args.o}")
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())