#!/usr/bin/env python
"""
relax_add_volumes.py - A program to add MRC volumes
Now it is duplicated with io.py
"""


import argparse
import mrcfile
import numpy as np

def average_mrc_volumes(input_list, output_file):
    """
    Average multiple MRC volumes from a list of input files.
    
    Parameters:
    -----------
    input_list : str
        Path to text file containing list of MRC file paths
    output_file : str
        Path to output averaged MRC file
    """
    # Read the list of input files
    with open(input_list, 'r') as f:
        input_files = [line.strip() for line in f if line.strip()]
    
    # Validate input files
    if not input_files:
        raise ValueError("No input files found in the list.")
    
    # Read the first file to get the volume dimensions
    with mrcfile.mmap(input_files[0], mode='r') as first_mrc:
        # Get the data type and shape of the first volume
        data_type = first_mrc.data.dtype
        volume_shape = first_mrc.data.shape
    
    # Initialize the accumulator for volumes
    volume_sum = np.zeros(volume_shape, dtype=data_type)
    
    # Accumulate volumes
    for file_path in input_files:
        with mrcfile.mmap(file_path, mode='r') as mrc:
            # Validate that all volumes have the same shape
            if mrc.data.shape != volume_shape:
                raise ValueError(f"Inconsistent volume shape in file: {file_path}")
            
            # Add to the volume sum
            volume_sum += mrc.data
    
    # Calculate the average
    volume_average = volume_sum / len(input_files)
    
    # Write the averaged volume to output file
    with mrcfile.new(output_file, overwrite=True) as mrc:
        mrc.set_data(volume_average)
    
    print(f"Averaged {len(input_files)} volumes. Output saved to {output_file}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Average multiple subtomogram MRC volumes.',
        epilog='Example: relax_add_volumes.py --list list.txt --o average.mrc'
    )
    
    parser.add_argument(
        '--list', 
        required=True, 
        help='Text file containing paths to input MRC files (one per line)'
    )
    
    parser.add_argument(
        '--o', 
        required=True, 
        help='Output averaged MRC file path'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Call the averaging function
    average_mrc_volumes(args.list, args.o)

if __name__ == '__main__':
    main()