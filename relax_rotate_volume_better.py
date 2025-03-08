#!/usr/bin/env python
"""
relax_rotate_volume.py - A program to rotate MRC volumes using Euler angles in ZYZ convention
with a single interpolation step, respecting MRC axis convention.
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
    # Convert angles to radians
    alpha_rad = np.deg2rad(alpha)
    beta_rad = np.deg2rad(beta)
    gamma_rad = np.deg2rad(gamma)
    
    # Create rotation matrices for each axis
    # In MRC convention: axis 0 = Z, axis 1 = Y, axis 2 = X
    
    # Rotation around Z axis (alpha) - affects X,Y coordinates (axes 2,1)
    Rz_alpha = np.array([
        [1, 0, 0],
        [0, np.cos(alpha_rad), -np.sin(alpha_rad)],
        [0, np.sin(alpha_rad), np.cos(alpha_rad)]
    ])
    
    # Rotation around Y axis (beta) - affects X,Z coordinates (axes 2,0)
    Ry_beta = np.array([
        [np.cos(beta_rad), 0, np.sin(beta_rad)],
        [0, 1, 0],
        [-np.sin(beta_rad), 0, np.cos(beta_rad)]
    ])
    
    # Rotation around Z axis (gamma) - affects X,Y coordinates (axes 2,1)
    Rz_gamma = np.array([
        [1, 0, 0],
        [0, np.cos(gamma_rad), -np.sin(gamma_rad)],
        [0, np.sin(gamma_rad), np.cos(gamma_rad)]
    ])
    
    # Combined rotation matrix for ZYZ Euler angles: R = Rz(alpha) * Ry(beta) * Rz(gamma)
    R = np.dot(Rz_alpha, np.dot(Ry_beta, Rz_gamma))
    
    # Get the center of the volume
    center = np.array(subtomo.shape) // 2
    
    # Make a copy to avoid modifying the original, and ensure float32 type
    rotated = subtomo.copy().astype(np.float32)
    
    # Create the affine transformation matrix
    # Note: affine_transform rotates around the origin, so we need to shift,
    # apply rotation, and shift back
    offset = center - np.dot(R, center)
    
    # Apply the rotation with a single interpolation
    # Note: We transpose the rotation matrix because scipy expects the transform
    # in the opposite convention compared to our definition
    rotated = affine_transform(
        rotated,
        matrix=R.T,
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
            
            # Calculate rotation matrix and apply rotation with single interpolation
            print("Applying rotation with single interpolation...")
            
            # Convert angles to radians
            alpha_rad = np.deg2rad(args.rot)
            beta_rad = np.deg2rad(args.tilt)
            gamma_rad = np.deg2rad(args.psi)
            
            # Create rotation matrices for each axis - accounting for MRC convention
            # In MRC: axis 0 = Z, axis 1 = Y, axis 2 = X
            
            # Rotation around Z axis (alpha) - affects X,Y coordinates (axes 2,1)
            Rz_alpha = np.array([
                [1, 0, 0],
                [0, np.cos(alpha_rad), -np.sin(alpha_rad)],
                [0, np.sin(alpha_rad), np.cos(alpha_rad)]
            ])
            
            # Rotation around Y axis (beta) - affects X,Z coordinates (axes 2,0)
            Ry_beta = np.array([
                [np.cos(beta_rad), 0, np.sin(beta_rad)],
                [0, 1, 0],
                [-np.sin(beta_rad), 0, np.cos(beta_rad)]
            ])
            
            # Rotation around Z axis (gamma) - affects X,Y coordinates (axes 2,1)
            Rz_gamma = np.array([
                [1, 0, 0],
                [0, np.cos(gamma_rad), -np.sin(gamma_rad)],
                [0, np.sin(gamma_rad), np.cos(gamma_rad)]
            ])
            
            # Combined rotation matrix for ZYZ Euler angles: R = Rz(alpha) * Ry(beta) * Rz(gamma)
            R = np.dot(Rz_alpha, np.dot(Ry_beta, Rz_gamma))
            
            # Get the center of the volume
            center = np.array(data.shape) // 2
            
            # Calculate offset for affine transform
            offset = center - np.dot(R, center)
            
            # Apply the rotation with a single interpolation
            rotated_data = affine_transform(
                data,
                matrix=R.T,  # Transpose because of scipy's convention
                offset=offset,
                order=args.order,  # Use specified interpolation order
                mode='constant',
                cval=0.0
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