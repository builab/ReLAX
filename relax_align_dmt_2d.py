#!/usr/bin/env python3
# Script to find polarity by reading star file and align
# Need to fix rec name different from tomoname
# Authors: HB, 02/2025

from util.subtomo import generate_2d_stack, low_pass_2D
from util.io import create_starfile, read_starfile_into_cilia_object, create_dir, get_filament_ids_from_object
from util.align import load_mrc, save_mrc_stack, align_image_stack_with_refs, save_mrc

import os
import argparse
import numpy as np
import pandas as pd
import starfile
import mrcfile
import glob


def nearest_even_integer(z):
    """
    Returns the nearest even integer to z_slices.
    """
    rounded = round(z)  # Round to the nearest integer
    if rounded % 2 == 0:
        return rounded  # Already even
    else:
        # Adjust to the nearest even integer
        return rounded + 1 if z > rounded else rounded - 1

                       
def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Script to align cross-section using fixed polarity.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing .star files")
    parser.add_argument("--tomo_dir", type=str, required=True, help="Directory containing .mrc files")
    parser.add_argument("--tomo_pattern", type=str, required=True, help="Tomo filename pattern (e.g., '{base_name}_8.48Apx.mrc', 'rec_{base_name}.mrc')")
    parser.add_argument("--stack_dir", type=str, required=True, help="Output directory containing .mrcs files")
    parser.add_argument("--box_size", type=int, default=106, required=True, help="Subtomo box size (~900 Angstrom)")
    parser.add_argument("--ref", type=str, required=True, help="Reference stack")
    parser.add_argument("--z_slices_in_nm", type=int, default=16, help="Z slices thickness to average in nm")
    parser.add_argument("--angpix", type=float, required=True, help="Pixel size in Angstrom")
    parser.add_argument("--lowpass", type=float, default=30, help="Low pass filter in Angstrom")
    parser.add_argument('--angle-range', type=float, nargs=2, default=(-180, 180),
                        help='Range of rotation angles to try (min max)')
    parser.add_argument('--angle-step', type=float, default=1, help='Step size for testing rotation angles')
    parser.add_argument('--max-shift', type=int, default=30, help='Maximum shift in X and Y')
   
    # Parse arguments
    args = parser.parse_args()
    box_size = args.box_size
    angpix = args.angpix
    z_slices = nearest_even_integer(args.z_slices_in_nm*10/angpix)
    lowpass = args.lowpass
    
    print(f'Star file dir: {args.input_dir} ')
    print(f'Tomogram folder: {args.tomo_dir} ')
    print(f'Pixel size of tomogram: {angpix} Angstrom')
    print(f'Output stack dir: {args.stack_dir} ')
    print(f'Box size: {box_size} pixels')
    print(f'Z slices number to average: {z_slices}')
    print(f'Low pass filter: {lowpass} Angstrom')

    # Reading the reference
    reference_stack = load_mrc(args.ref, is_stack=True)

    create_dir(args.stack_dir)
    star_files = glob.glob(os.path.join(args.input_dir, "*.star"))

    if not star_files:
        print(f"No .star files found in {args.input_dir}.")
        return
    
    # Making the stacks
    for star_file in sorted(star_files):
        # Get the base name of the .star file (without extension)
        base_name = os.path.basename(star_file)
        if base_name.endswith(".star"):
            base_name = base_name[:-5]  # Remove the last 5 characters (".star")  
              
        # Construct the expected .mrc file path
        mrc_file = os.path.join(args.tomo_dir, args.tomo_pattern.format(base_name=base_name))
        
        # Check if the .mrc file exists
        if os.path.exists(mrc_file):
            print(f"-----> Processing {mrc_file}  <-----")
        else:
            print(f"No corresponding .mrc file found for {star_file} in {args.tomo_dir}.")
        # Read star file
        
        out_stack_file = os.path.join(args.stack_dir, base_name + '.mrcs')
        aligned_csv_file = out_stack_file.replace('.mrcs', '.csv')
        aligned_stack_file = os.path.join(args.stack_dir, base_name + '_aln.mrcs')
        average_file = os.path.join(args.stack_dir, base_name + '_avg.mrcs')

        print(f"Generating {out_stack_file}.")
        
        cilia_object = read_starfile_into_cilia_object(star_file)
        combined_stack = []
        img_list = []
        for obj_id, obj_data in enumerate(cilia_object, start=1):
            stack = generate_2d_stack(mrc_file, obj_data, box_size, z_slices)
            combined_stack = np.vstack((stack, *combined_stack))
            filament_list  = get_filament_ids_from_object(cilia_object, obj_id)
            img_ids = list(range(len(filament_list)))
            for img_id in img_ids:
                img_list.append({'ObjectID': obj_id, 'FilamentID': filament_list[img_id], 'ImageID': img_id})
            # Perform alignment as well

        # Convert the results to a DataFrame
        stack_df = pd.DataFrame(img_list)
        #print(stack_df['ImageID'].apply(type))
        #stack_df['ImageID'] = stack_df['ImageID'].astype(int)

        # Save the DataFrame to a CSV file
        for i, slice in enumerate(combined_stack):
            stack[i] = low_pass_2D(slice, lowpass, angpix)
            with mrcfile.new(out_stack_file, overwrite=True) as mrc:
                # Need to set proper header later
                mrc.set_data(stack.astype(np.float32))

        # Align images
        print(f'Aligning {out_stack_file} with reference {args.ref}')
        results, aligned_stack = align_image_stack_with_refs(
            combined_stack, reference_stack, args.angle_range, args.angle_step, args.max_shift, args.max_shift
        )
        results_df = pd.DataFrame(results)
        #print(results_df['ImageID'].apply(type))
        merged_df = pd.merge(stack_df, results_df, on='ImageID', how='inner')
        merged_df.to_csv(aligned_csv_file, index=False)
        print(f'Writing alignment to {aligned_csv_file} ')
        # Save best aligned images (due to possiblity of multi-ref)
        
        aligned_images = [aligned_image for (img_id, r_id), aligned_image in aligned_stack.items() if r_id == 0]
        aligned_images = np.stack(aligned_images)
        save_mrc_stack(aligned_images, aligned_stack_file)
        
        merged_df = merged_df.sort_values(by='ObjectID', ascending=True)
        
        average_stack =[]
        # Save the average image as a separate MRC file (NOT WORKING PROPERLY WITH > 1 cilia)
        for obj_id, group in merged_df.groupby('ObjectID'):
            #print(obj_id)
            img_ids = group['ImageID'].tolist()
            #print(img_ids)
            images = aligned_images[img_ids]
            average_image = np.mean(images, axis=0)
            average_stack.append(average_image)

        # Assuming save_mrc takes a single 2D image
        save_mrc_stack(np.stack(average_stack), average_file)
        print(f"Average of aligned images saved to: {average_file}")
        
if __name__ == "__main__":
    main()

