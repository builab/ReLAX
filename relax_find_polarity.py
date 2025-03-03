#!/usr/bin/env python3
# Script to find polarity by reading star file and align
# Authors: HB, 02/2025

from util.subtomo import extract_subtomograms, rotate_subtomograms_zyz, average_z_sections, low_pass_2D, generate_tomogram_cross_section
from util.io import create_starfile, read_starfile_into_cilia_object
from util.geom import process_cross_section
import mrcfile

import argparse
import numpy as np
import pandas as pd
import starfile

# main
def get_median_row(group):
    median_value = group['rlnTomoParticleId'].median()
    closest_row = group.iloc[(group['rlnTomoParticleId'] - median_value).abs().argmin()]
    return closest_row
    
def main():

    df = starfile.read('CU428lowmag_11.star')
    tomogram_file = "/Users/kbui2/Desktop/Sessions/CU428lowmag_11_14.00Apx_refined.mrc"
    tomo_angpix = 3.37
    angpix = 14.00
    lowpass = 40

    result_df = df.groupby('rlnHelicalTubeID', group_keys=False).apply(get_median_row)
    
    print(result_df)
    create_starfile(result_df, 'tmp.star')

    box_size = 64
    centers = result_df[['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']].astype(int).values.tolist()
    #centers *= tomo_angpix/angpix
    #centers = int(centers).tolist()
    print(centers)
    subtomograms = extract_subtomograms(tomogram_file, centers, [box_size, box_size, box_size])
    
    eulers = np.array(result_df[['rlnAngleRot', 'rlnAngleTilt', 'rlnAnglePsi']])
    # Reverse
    euler_angles = (-eulers[:, [2, 1, 0]]).tolist()
    rotated_subtomograms = rotate_subtomograms_zyz(subtomograms, euler_angles)
    
    # Specify number of Z slices to average (must be even)
    z_slices_to_avg = 10
    
    # Average Z sections for each rotated subtomogram
    averaged_slices = average_z_sections(rotated_subtomograms, z_slices_to_avg)
    print(f"Created {len(averaged_slices)} averaged slices of shape {averaged_slices[0].shape}")
    
    
    # Test the generate cross section
    objects = read_starfile_into_cilia_object('CU428lowmag_11.star')
    tomo_z_slices_to_avg = 50
    for i, obj_data in enumerate(objects):
        cross_section = process_cross_section(obj_data)
        cross_section_2D = generate_tomogram_cross_section(tomogram_file, cross_section, tomo_z_slices_to_avg)
        with mrcfile.new(f"Cross_section.mrc", overwrite=True) as mrc:
            mrc.set_data(low_pass_2D(cross_section_2D[0], lowpass, angpix).astype(np.float32))
            print(f"Cross section saved to Cross_section.mrc")


    # Example: Save the rotated subtomograms
    for i, subtomo in enumerate(subtomograms):
        with mrcfile.new(f"subtomo_{i}.mrc", overwrite=True) as mrc:
            mrc.set_data(subtomo.astype(np.float32))
            print(f"Subtomogram {i} saved to 'subtomo_{i}.mrc'")
            
    # Example: Save the rotated subtomograms
    for i, subtomo in enumerate(rotated_subtomograms):
        with mrcfile.new(f"rotated_subtomo_{i}.mrc", overwrite=True) as mrc:
            mrc.set_data(subtomo.astype(np.float32))
            print(f"Rotated subtomogram {i} saved to 'rotated_subtomo_{i}.mrc'")

    # Example: Save the averaged slices from rotated subtomograms
    for i, avg_slice in enumerate(averaged_slices):
        with mrcfile.new(f"rotated_avg_slice_{i}.mrc", overwrite=True) as mrc:
            mrc.set_data(low_pass_2D(avg_slice, lowpass, angpix).astype(np.float32))
            print(f"Averaged slice {i} from rotated subtomogram saved to 'rotated_avg_slice_{i}.mrc'")
                        
if __name__ == "__main__":
    main()
