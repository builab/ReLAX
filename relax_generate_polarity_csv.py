#!/usr/bin/env python3
# Simple script to generate polarity files with either polarity 0 or 1

from util.io import get_obj_ids_from_model

import numpy as np
import argparse
import glob
import os
import csv
        
# main
def main():
    parser = argparse.ArgumentParser(description="Process input file and generate STAR files.")
    
    # Add arguments with default values
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the input dir containing IMOD models")
    parser.add_argument("--mod_suffix", type=str, default="", help="Suffix of IMOD models without .mod")

    # Parse arguments
    args = parser.parse_args()

    input_dir=args.input_dir
    
    print(f'Model suffix: {args.mod_suffix}')
    print(f'Model directory: {input_dir}')

    pattern = os.path.join(input_dir, f"*{args.mod_suffix}.mod")
    
    # Use glob to find all files matching the pattern
    matching_files = glob.glob(pattern)
    
    # Open CSV files for writing
    with open('polarity_0.csv', 'w', newline='') as csvfile_0, open('polarity_1.csv', 'w', newline='') as csvfile_1:
        writer_0 = csv.writer(csvfile_0)
        writer_1 = csv.writer(csvfile_1)
        
        # Write headers
        writer_0.writerow(['rlnTomoName', 'ObjectID', 'Polarity'])
        writer_1.writerow(['rlnTomoName', 'ObjectID', 'Polarity'])
        
        for input_file in sorted(matching_files):
            filename = os.path.basename(input_file)
            tomo_name = filename.removesuffix(args.mod_suffix + ".mod")
            obj_ids = get_obj_ids_from_model(input_file)
            
            # Write to polarity_0.csv and polarity_1.csv
            for obj_id in obj_ids:
                writer_0.writerow([tomo_name, int(obj_id), 0])  # Write to polarity_0.csv
                writer_1.writerow([tomo_name, int(obj_id), 1])  # Write to polarity_1.csv

    print('polarity_0.csv and polarity_1.csv file written')
    
if __name__ == "__main__":
    main()
