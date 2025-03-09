#!/usr/bin/env python
# -*- coding: utf-8 -*-

import mrcfile
import numpy as np
from scipy import ndimage
from skimage.registration import phase_cross_correlation
from skimage.transform import rotate
import argparse
import os
import csv

def load_mrc(filepath, is_stack=False):
    """
    Load MRC file as numpy array.
    
    Parameters:
    -----------
    filepath : str
        Path to MRC file
    is_stack : bool
        If True, load as stack, otherwise load as single image
        
    Returns:
    --------
    data : numpy.ndarray or list of numpy.ndarray
        2D image data from MRC file or list of 2D images if is_stack=True
    """
    with mrcfile.open(filepath) as mrc:
        data = mrc.data
        
        if is_stack:
            # Ensure data is treated as a stack
            if data.ndim < 3:
                # Convert single image to stack with one image
                data = np.expand_dims(data, axis=0)
            return data
        else:
            # Return single image
            if data.ndim > 2:
                data = data[0]  # Take first slice if 3D
            return data

def save_mrc(data, filepath):
    """
    Save numpy array as MRC file.
    
    Parameters:
    -----------
    data : numpy.ndarray
        2D image data
    filepath : str
        Path where MRC file will be saved
    """
    with mrcfile.new(filepath, overwrite=True) as mrc:
        mrc.set_data(data.astype(np.float32))

def save_mrc_stack(stack, filepath):
    """
    Save stack of numpy arrays as MRC stack file.
    
    Parameters:
    -----------
    stack : numpy.ndarray
        3D array with stack of 2D images
    filepath : str
        Path where MRC stack file will be saved
    """
    with mrcfile.new(filepath, overwrite=True) as mrc:
        mrc.set_data(stack.astype(np.float32))

def calculate_cross_correlation(img1, img2):
    """
    Calculate normalized cross-correlation coefficient between two images.
    
    Parameters:
    -----------
    img1, img2 : numpy.ndarray
        Input images
        
    Returns:
    --------
    cc : float
        Normalized cross-correlation coefficient
    """
    img1_norm = (img1 - np.mean(img1)) / (np.std(img1) * len(img1.flatten()))
    img2_norm = (img2 - np.mean(img2)) / np.std(img2)
    cc = np.sum(img1_norm * img2_norm)
    return cc

def align_images_legacy(image, reference, angle_range=(-180, 180), angle_step=1):
    """
    Align a 2D image with a reference by testing different rotation angles
    and determining the optimal translation.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Image to be aligned
    reference : numpy.ndarray
        Reference image
    angle_range : tuple
        Range of rotation angles to try (min, max)
    angle_step : float
        Step size for angle testing
        
    Returns:
    --------
    best_angle : float
        Optimal rotation angle
    shift_y, shift_x : float
        Optimal translation in y and x directions
    best_cc : float
        Cross-correlation coefficient at the optimal alignment
    aligned_image : numpy.ndarray
        The aligned image
    """
    best_cc = -np.inf
    best_angle = 0
    best_shift = (0, 0)
    best_aligned = None
    
    angles = np.arange(angle_range[0], angle_range[1], angle_step)
    
    for angle in angles:
        # Rotate the image
        rotated = rotate(image, angle, preserve_range=True, mode='constant')
        
        # Find the shift using phase correlation
        shift, error, diffphase = phase_cross_correlation(reference, rotated)
        
        # Apply the shift
        shifted = ndimage.shift(rotated, shift)
        
        # Calculate cross-correlation
        cc = calculate_cross_correlation(shifted, reference)
        
        # Update best alignment if better correlation found
        if cc > best_cc:
            best_cc = cc
            best_angle = angle
            best_shift = shift
            best_aligned = shifted.copy()
    
    return best_angle, best_shift[1], best_shift[0], best_cc, best_aligned
    
def align_images(image, reference, angle_range=(-180, 180), angle_step=1, max_shift_x=30, max_shift_y=30):
    """
    Align a 2D image with a reference by testing different rotation angles
    and determining the optimal translation, with limits on shift values.
    
    Parameters:
    -----------
    image : numpy.ndarray
        Image to be aligned
    reference : numpy.ndarray
        Reference image
    angle_range : tuple
        Range of rotation angles to try (min, max)
    angle_step : float
        Step size for angle testing
    max_shift_x : int
        Maximum allowed shift in x direction (pixels)
    max_shift_y : int
        Maximum allowed shift in y direction (pixels)
        
    Returns:
    --------
    best_angle : float
        Optimal rotation angle
    shift_y, shift_x : float
        Optimal translation in y and x directions
    best_cc : float
        Cross-correlation coefficient at the optimal alignment
    aligned_image : numpy.ndarray
        The aligned image
    """
    best_cc = -np.inf
    best_angle = 0
    best_shift = (0, 0)
    best_aligned = None
    
    angles = np.arange(angle_range[0], angle_range[1], angle_step)
    
    for angle in angles:
        # Rotate the image
        rotated = rotate(image, angle, preserve_range=True, mode='constant')
        
        # Find the shift using phase correlation
        shift, error, diffphase = phase_cross_correlation(reference, rotated)
        
        # Limit the shift values to the specified maximum
        shift_y, shift_x = shift
        shift_y = np.clip(shift_y, -max_shift_y, max_shift_y)
        shift_x = np.clip(shift_x, -max_shift_x, max_shift_x)
        limited_shift = (shift_y, shift_x)
        
        # Apply the limited shift
        shifted = ndimage.shift(rotated, limited_shift)
        
        # Calculate cross-correlation
        cc = calculate_cross_correlation(shifted, reference)
        
        # Update best alignment if better correlation found
        if cc > best_cc:
            best_cc = cc
            best_angle = angle
            best_shift = limited_shift
            best_aligned = shifted.copy()
    
    return best_angle, best_shift[1], best_shift[0], best_cc, best_aligned

def align_image_stack_with_refs(image_stack, reference_stack, angle_range=(-180, 180), angle_step=1, max_shift_x=30, max_shift_y=30):
    """
    Align each image in a stack with each reference in a reference stack.
    
    Parameters:
    -----------
    image_stack : numpy.ndarray
        Stack of 2D images to be aligned
    reference_stack : numpy.ndarray
        Stack of 2D reference images
    angle_range : tuple
        Range of rotation angles to try (min, max)
    angle_step : float
        Step size for angle testing
        
    Returns:
    --------
    results : list of dict
        List of dictionaries with alignment results for each image-reference pair
    aligned_images : dict
        Dictionary of aligned images keyed by (image_id, ref_id)
    """
    results = []
    aligned_images = {}
    
    # Ensure both stacks have the same dimensions
    min_shape = [
        min(dim1, dim2) 
        for dim1, dim2 in zip(image_stack[0].shape, reference_stack[0].shape)
    ]
    
    # Process each image with each reference
    for img_id in range(len(image_stack)):
        image = image_stack[img_id].copy()
        image = image[:min_shape[0], :min_shape[1]]
        
        for ref_id in range(len(reference_stack)):
            reference = reference_stack[ref_id].copy()
            reference = reference[:min_shape[0], :min_shape[1]]
            
            # Align image with reference
            angle, shift_x, shift_y, cc, aligned_image = align_images(
                image, reference, angle_range, angle_step, max_shift_x, max_shift_y
            )
            
            # Store results
            result = {
                'ImageID': img_id,
                'RefID': ref_id,
                'RotAngle': angle,
                'Shift_X': shift_x,
                'Shift_Y': shift_y,
                'CC': cc
            }
            results.append(result)
            
            # Store aligned image
            aligned_images[(img_id, ref_id)] = aligned_image
            
    return results, aligned_images

def main():
    parser = argparse.ArgumentParser(description='Align MRC image stack with multiple references.')
    parser.add_argument('--images', required=True, help='Path to the MRC image stack to be aligned')
    parser.add_argument('--references', required=True, help='Path to the reference MRC image stack')
    parser.add_argument('--output-dir', default='.', help='Directory to save aligned images and results')
    parser.add_argument('--angle-range', type=float, nargs=2, default=(-180, 180),
                        help='Range of rotation angles to try (min max)')
    parser.add_argument('--angle-step', type=float, default=1, help='Step size for testing rotation angles')
    parser.add_argument('--max-shift', type=int, default=20, help='Maximum shift in pixels')
    parser.add_argument('--save-aligned', action='store_true', help='Save all aligned images')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load image and reference stacks
    image_stack = load_mrc(args.images, is_stack=True)
    reference_stack = load_mrc(args.references, is_stack=True)
    
    print(f"Loaded {len(image_stack)} images and {len(reference_stack)} references")
    
    # Align images
    results, aligned_images = align_image_stack_with_refs(
        image_stack, reference_stack, args.angle_range, args.angle_step, args.max_shift, args.max_shift
    )
    
    # Save results to CSV
    csv_path = os.path.join(args.output_dir, "alignment_results.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['ImageID', 'RefID', 'RotAngle', 'Shift_X', 'Shift_Y', 'CC']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    print(f"Alignment results saved to: {csv_path}")
    
    # Determine best reference for each image based on CC
    best_alignments = {}
    for result in results:
        img_id = result['ImageID']
        if img_id not in best_alignments or result['CC'] > best_alignments[img_id]['CC']:
            best_alignments[img_id] = result
    
    # Save best aligned images
    best_aligned_stack = []
    average = []
    for img_id in range(len(image_stack)):
        if img_id in best_alignments:
            ref_id = best_alignments[img_id]['RefID']
            best_aligned_stack.append(aligned_images[(img_id, ref_id)])
            
    # Save the average of all the aligned image
    if best_aligned_stack:
        best_aligned_stack = np.array(best_aligned_stack)
        best_aligned_path = os.path.join(args.output_dir, "best_aligned.mrcs")
        save_mrc_stack(best_aligned_stack, best_aligned_path)
        print(f"Best aligned images saved to: {best_aligned_path}")
        
        # Save the average image as a separate MRC file
        average_image = np.mean(best_aligned_stack, axis=0)
        average_path = os.path.join(args.output_dir, "average.mrc")
        # Assuming save_mrc takes a single 2D image
        save_mrc(average_image, average_path)
        print(f"Average of best aligned images saved to: {average_path}")
    
    # Optionally save all aligned images
    if args.save_aligned:
        for (img_id, ref_id), aligned_img in aligned_images.items():
            aligned_path = os.path.join(args.output_dir, f"aligned_img{img_id}_ref{ref_id}.mrc")
            save_mrc(aligned_img, aligned_path)
        print(f"All aligned images saved to: {args.output_dir}")
    
    # Save best alignment results to a separate CSV
    best_csv_path = os.path.join(args.output_dir, "best_alignments.csv")
    with open(best_csv_path, 'w', newline='') as csvfile:
        fieldnames = ['ImageID', 'RefID', 'RotAngle', 'Shift_X', 'Shift_Y', 'CC']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for img_id in range(len(image_stack)):
            if img_id in best_alignments:
                writer.writerow(best_alignments[img_id])
    
    print(f"Best alignment results saved to: {best_csv_path}")
    
    

if __name__ == "__main__":
    main()