#!/usr/bin/env python3
# Script to find polarity by reading star file and align
# Authors: HB, 02/2025

from util.subtomo import extract_subtomograms, rotate_subtomograms_zyz, average_z_sections, low_pass_2D, generate_tomogram_cross_section
from util.io import create_starfile, read_starfile_into_cilia_object
from util.geom import process_cross_section
import mrcfile

import os
import argparse
import numpy as np
import pandas as pd
import starfile

# main
def get_median_row(group):
    median_value = group['rlnTomoParticleId'].median()
    closest_row = group.iloc[(group['rlnTomoParticleId'] - median_value).abs().argmin()]
    return closest_row
    
def generate_2d_stack(tomogram_file, star_file, box_size, tomo_angpix, angpix, z_slices_to_avg):

    print(f'Read {star_file}')
    df = starfile.read(star_file)
    #lowpass = 40

    result_df = df.groupby('rlnHelicalTubeID', group_keys=False).apply(get_median_row)
    
    #print(result_df)
    #create_starfile(result_df, 'tmp.star')

    centers = result_df[['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']].astype(int).values.tolist()
    #centers *= tomo_angpix/angpix
    #centers = int(centers).tolist()
    print("Extracting subtomograms")
    subtomograms = extract_subtomograms(tomogram_file, centers, [box_size, box_size, box_size])
    print(subtomograms[0].dtype)
    #for i, subtomo in enumerate(subtomograms):
    #    with mrcfile.new(f"subtomo_{i}.mrc", overwrite=True) as mrc:
    #        mrc.set_data(subtomo.astype(np.float32))
    #        print(f"Subtomogram {i} saved to 'subtomo_{i}.mrc'")
    
    eulers = np.array(result_df[['rlnAngleRot', 'rlnAngleTilt', 'rlnAnglePsi']])
    print(eulers)
    # Reverse
    euler_angles = (-eulers[:, [2, 1, 0]]).tolist()
    rotated_subtomograms = rotate_subtomograms_zyz(subtomograms, euler_angles)
    

    
    # Average Z sections for each rotated subtomogram
    averaged_slices = average_z_sections(rotated_subtomograms, z_slices_to_avg)
    print(f"Created {len(averaged_slices)} averaged slices of shape {averaged_slices[0].shape}")
    
    
    # Test the generate cross section
    #objects = read_starfile_into_cilia_object(star_file)
    #tomo_z_slices_to_avg = 2
    #for i, obj_data in enumerate(objects):
    #    cross_section = process_cross_section(obj_data)
    #    cross_section_2D = generate_tomogram_cross_section(tomogram_file, cross_section, tomo_z_slices_to_avg)
    #    with mrcfile.new(f"Cross_section.mrc", overwrite=True) as mrc:
    #        mrc.set_data(low_pass_2D(cross_section_2D[0], lowpass, angpix).astype(np.float32))
    #        print(f"Cross section saved to Cross_section.mrc")


    # Example: Save the rotated subtomograms
    #for i, subtomo in enumerate(subtomograms):
    #    with mrcfile.new(f"subtomo_{i}.mrc", overwrite=True) as mrc:
    #        mrc.set_data(subtomo.astype(np.float32))
    #        print(f"Subtomogram {i} saved to 'subtomo_{i}.mrc'")
            
    #Example: Save the rotated subtomograms
    #for i, subtomo in enumerate(rotated_subtomograms):
    #    with mrcfile.new(f"rotated_subtomo_{i}.mrc", overwrite=True) as mrc:
    #        mrc.set_data(subtomo.astype(np.float32))
    #        print(f"Rotated subtomogram {i} saved to 'rotated_subtomo_{i}.mrc'")

    # Example: Save the averaged slices from rotated subtomograms
    #for i, avg_slice in enumerate(averaged_slices):
    #    with mrcfile.new(f"rotated_avg_slice_{i}.mrc", overwrite=True) as mrc:
    #        mrc.set_data(low_pass_2D(avg_slice, lowpass, angpix).astype(np.float32))
    #        print(f"Averaged slice {i} from rotated subtomogram saved to 'rotated_avg_slice_{i}.mrc'")
    stack = np.stack(averaged_slices, axis=0)       
    return stack
                            
if __name__ == "__main__":
    file_list = ["CU428base_001_8.48Apx.mrc", "CU428base_003_8.48Apx.mrc", "CU428base_004_8.48Apx.mrc", "CU428base_005_8.48Apx.mrc", "CU428base_006_8.48Apx.mrc", "CU428base_007_8.48Apx.mrc", "CU428base_008_8.48Apx.mrc", "CU428base_009_8.48Apx.mrc"]
    #file_list = ["CU428base_001_8.48Apx.mrc"]

    box_size = 106
    # Specify number of Z slices to average (must be even & in relation to pixel size)
    z_slices_to_avg = 16
    angpix = 8.48
    lowpass = 30
    tomo_angpix = 2.12
    
    for file_path in file_list:
        if os.path.exists(file_path):
            star_file = file_path.replace('.mrc', '.star')
            out_stack_file = file_path.replace('.mrc', '_c1.mrcs')
            stack = generate_2d_stack(file_path, star_file, box_size, tomo_angpix, angpix, z_slices_to_avg)
            for i, slice in enumerate(stack):
                stack[i] = low_pass_2D(slice, lowpass, angpix)

            with mrcfile.new(out_stack_file, overwrite=True) as mrc:
                # Need to set proper header later
                mrc.set_data(stack.astype(np.float32))
