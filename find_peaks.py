#!/usr/bin/env python3
"""
Peak finding script for 3D cross-correlation MRC files.
Finds local maxima in 3D correlation maps and outputs peak coordinates and intensities.

Usage:
    python find_peaks.py --i corr.mrc --o peak.csv --npeaks 2000 --distance 5
"""

import argparse
import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter
from scipy.spatial.distance import cdist
import mrcfile

def read_mrc_file(filename):
    """Read MRC file and return 3D numpy array."""
    try:
        with mrcfile.open(filename, mode='r') as mrc:
            data = mrc.data.copy()
            # Convert to standard float64 to avoid dtype issues
            data = data.astype(np.float64)
        return data
    except Exception as e:
        raise RuntimeError(f"Error reading MRC file {filename}: {e}")

def find_local_maxima_3d(data, min_distance=1):
    """
    Find local maxima in 3D data using maximum filter.
    
    Parameters:
    -----------
    data : np.ndarray
        3D input array
    min_distance : float
        Minimum distance between peaks (in voxels)
        
    Returns:
    --------
    peaks : np.ndarray
        Array of peak coordinates (z, y, x)
    intensities : np.ndarray
        Array of intensity values at peak locations
    """
    # Create footprint for maximum filter (sphere-like neighborhood)
    # Convert to integer for footprint size
    footprint_size = int(2 * min_distance + 1)
    footprint = np.ones((footprint_size, footprint_size, footprint_size))
    
    # Find local maxima using maximum filter
    local_maxima = maximum_filter(data, footprint=footprint) == data
    
    # Additional filtering: only keep peaks above a reasonable threshold
    # Use a percentile-based threshold to reduce the number of peaks
    threshold = np.percentile(data, 95)  # Keep only top 5% of values
    local_maxima = local_maxima & (data > threshold)
    
    # Get coordinates of local maxima
    peak_coords = np.where(local_maxima)
    peak_coords = np.column_stack(peak_coords)  # (z, y, x) format
    
    # Get intensities at peak locations
    intensities = data[local_maxima]
    
    return peak_coords, intensities

def enforce_minimum_distance_efficient(peak_coords, intensities, min_distance, max_peaks=None):
    """
    Efficient minimum distance enforcement using spatial indexing.
    
    Parameters:
    -----------
    peak_coords : np.ndarray
        Array of peak coordinates (z, y, x)
    intensities : np.ndarray
        Array of intensity values
    min_distance : float
        Minimum distance between peaks
    max_peaks : int or None
        Maximum number of peaks to return
        
    Returns:
    --------
    filtered_coords : np.ndarray
        Filtered peak coordinates
    filtered_intensities : np.ndarray
        Filtered intensity values
    """
    if len(peak_coords) == 0:
        return peak_coords, intensities
    
    # Sort peaks by intensity (descending) and limit early if needed
    sort_idx = np.argsort(-intensities)
    
    # If we have way more peaks than needed, pre-filter to save time
    if max_peaks is not None and len(sort_idx) > max_peaks * 10:
        sort_idx = sort_idx[:max_peaks * 10]
        print(f"Pre-filtering to top {max_peaks * 10} peaks for efficiency")
    
    sorted_coords = peak_coords[sort_idx]
    sorted_intensities = intensities[sort_idx]
    
    # Use a more efficient approach: process peaks in order and use spatial hashing
    selected_coords = []
    selected_intensities = []
    
    for i, (coord, intensity) in enumerate(zip(sorted_coords, sorted_intensities)):
        if len(selected_coords) == 0:
            # First peak is always selected
            selected_coords.append(coord)
            selected_intensities.append(intensity)
        else:
            # Check distance to all previously selected peaks
            selected_array = np.array(selected_coords)
            distances = np.sqrt(np.sum((selected_array - coord)**2, axis=1))
            
            # Only add if far enough from all selected peaks
            if np.all(distances >= min_distance):
                selected_coords.append(coord)
                selected_intensities.append(intensity)
                
                # Stop if we have enough peaks
                if max_peaks is not None and len(selected_coords) >= max_peaks:
                    break
        
        # Progress indicator for large datasets
        if i > 0 and i % 10000 == 0:
            print(f"Processed {i} peaks, selected {len(selected_coords)}")
    
    return np.array(selected_coords), np.array(selected_intensities)

def find_peaks_3d(data, max_peaks=None, min_distance=1):
    """
    Find peaks in 3D correlation map.
    
    Parameters:
    -----------
    data : np.ndarray
        3D correlation map
    max_peaks : int or None
        Maximum number of peaks to return
    min_distance : float
        Minimum distance between peaks (in voxels)
        
    Returns:
    --------
    peak_coords : np.ndarray
        Peak coordinates in (z, y, x) format
    intensities : np.ndarray
        Peak intensity values
    """
    # Find initial local maxima
    peak_coords, intensities = find_local_maxima_3d(data, min_distance)
    
    if len(peak_coords) == 0:
        print("No peaks found in the correlation map.")
        return np.array([]), np.array([])
    
    print(f"Found {len(peak_coords)} initial local maxima")
    
    # Enforce minimum distance constraint
    peak_coords, intensities = enforce_minimum_distance_efficient(
        peak_coords, intensities, min_distance, max_peaks
    )
    
    print(f"After distance filtering: {len(peak_coords)} peaks")
    
    # Limit to maximum number of peaks (keeping strongest ones)
    if max_peaks is not None and len(peak_coords) > max_peaks:
        # Already sorted by intensity in enforce_minimum_distance
        peak_coords = peak_coords[:max_peaks]
        intensities = intensities[:max_peaks]
        print(f"Limited to top {max_peaks} peaks")
    
    return peak_coords, intensities

def save_peaks_to_csv(peak_coords, intensities, output_file):
    """
    Save peak coordinates and intensities to CSV file.
    
    Parameters:
    -----------
    peak_coords : np.ndarray
        Peak coordinates in (z, y, x) format
    intensities : np.ndarray
        Peak intensity values
    output_file : str
        Output CSV filename
    """
    if len(peak_coords) == 0:
        # Create empty CSV with headers
        df = pd.DataFrame(columns=['x', 'y', 'z', 'intensity'])
    else:
        # Convert (z, y, x) to (x, y, z) for output
        df = pd.DataFrame({
            'x': peak_coords[:, 2],  # x coordinate
            'y': peak_coords[:, 1],  # y coordinate  
            'z': peak_coords[:, 0],  # z coordinate
            'intensity': intensities
        })
    
    df.to_csv(output_file, index=False)
    print(f"Saved {len(df)} peaks to {output_file}")

def main():
    parser = argparse.ArgumentParser(
        description='Find peaks in 3D cross-correlation MRC files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python find_peaks.py --i corr.mrc --o peak.csv --npeaks 2000 --distance 5
    python find_peaks.py --i corr.mrc --o peak.csv --npeaks 1000 --distance 3
        """
    )
    
    parser.add_argument('--i', '--input', dest='input_file', required=True,
                        help='Input MRC file containing 3D correlation map')
    parser.add_argument('--o', '--output', dest='output_file', required=True,
                        help='Output CSV file for peak coordinates and intensities')
    parser.add_argument('--npeaks', type=int, default=None,
                        help='Maximum number of peaks to find (default: no limit)')
    parser.add_argument('--distance', type=float, default=1,
                        help='Minimum distance between peaks in voxels (default: 1)')
    
    args = parser.parse_args()
    
    try:
        # Read MRC file
        print(f"Reading MRC file: {args.input_file}")
        correlation_map = read_mrc_file(args.input_file)
        print(f"Correlation map shape: {correlation_map.shape}")
        print(f"Intensity range: {correlation_map.min():.6f} to {correlation_map.max():.6f}")
        
        # Find peaks
        print(f"Finding peaks with max_peaks={args.npeaks}, min_distance={args.distance}")
        peak_coords, intensities = find_peaks_3d(
            correlation_map, 
            max_peaks=args.npeaks, 
            min_distance=args.distance
        )
        
        # Save results
        save_peaks_to_csv(peak_coords, intensities, args.output_file)
        
        if len(peak_coords) > 0:
            print(f"\nPeak statistics:")
            print(f"  Number of peaks: {len(peak_coords)}")
            print(f"  Intensity range: {intensities.min():.6f} to {intensities.max():.6f}")
            print(f"  Mean intensity: {intensities.mean():.6f}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())
