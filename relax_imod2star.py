#!/usr/bin/env python3
# Script to convert IMOD star file to Relion 5 star file
# Testing now with 2 cilia
# Authors: Molly & HB, 02/2025
# TODO: We need to plot in function for calculate_rot_angles, also, calculate a elliptically
# TODO: Draw the filament in the propagate_rot_to_entire_cilia as well for visualization

# To check if HelicalTubuleID = 11-19 also work

from util.geom import process_cross_section, rotate_cross_section, calculate_rot_angles, propagate_rot_to_entire_cilia, plot_ellipse_cs
from util.io import run_model2point, create_starfile, process_imod_point_file

import argparse
import numpy as np
import pandas as pd
import starfile

# main
def main():

    parser = argparse.ArgumentParser(description="Process input file and generate STAR files.")
    
    # Add arguments with default values
    parser.add_argument("--input", type=str, required=True, help="Path to the input file (e.g., input.txt)")
    parser.add_argument("--spacing", type=float, default=82.0, help="Repeating unit spacing (default: 82.0)")
    parser.add_argument("--angpix", type=float, default=8.48, help="Pixel size in Angstroms (default: 8.48)")
    parser.add_argument("--tomo_angpix", type=float, default=2.12, help="Pixel size of unbinned tomogram in Angstroms (default: 2.12)")
    parser.add_argument("--fit", type=str, default="simple", help="Fitting type: simple or ellipse")
    parser.add_argument("--polarity", type=str, default="", help="Polarity file for angle prediction.")
    parser.add_argument("--reorder", type=float, default="0", help="Reorder filament using ellipse fit")
    parser.add_argument("--output", type=str, default="out.star", help="Output STAR file for interpolation (default: output_interpolation_with_angles.star)")

    # Parse arguments
    args = parser.parse_args()

    input_file=args.input
    spacing=args.spacing
    angpix=args.angpix
    output=args.output
    fit_method = args.fit
    tomo_angpix = args.tomo_angpix
    reorder = float(args.reorder)
    polarity = args.polarity
    
    print(f'Input model file: {input_file}')
    print(f'Output star file: {output}')
    print(f'Pixel size of input model: {angpix} Angstrom')
    print(f'Pixel size of original tomogram: {tomo_angpix} Angstrom')
    print(f'Fitting method: {fit_method}')
    print(f'Reorder doublet: {reorder} (0: no, 1: yes)')
    if reorder > 0 and fit_method != 'ellipse':
        print('Reorder only available with ellipse fitting')
    print(f'Repeating unit to interpolate: {spacing} Angstrom')
    if polarity != "":
        print(f'Polarity file: {polarity} ')
    else:
        print(f'Fitting without polarity')
    
    # Convert IMOD to txt file
    input_txt = input_file.replace(".mod", ".txt")
    run_model2point(input_file, input_txt)

    # interpolation and acquire data
    objects = process_imod_point_file(input_txt, spacing, angpix, tomo_angpix)  
    
    #print(objects)
        
    # create starfile with interpolation and angle
    create_starfile(objects, output.replace(".star", "_init.star"))
     
    new_objects = [] 
    # find cross section
    for i, obj_data in enumerate(objects):
        cross_section = process_cross_section(obj_data)
        rotated_cross_section = rotate_cross_section(cross_section)

        output_cs = input_file.replace(".mod", f"_{i+1}.png")
        # Plot cross section
        plot_ellipse_cs(rotated_cross_section, output_cs)
        # obtain rot 
        updated_cross_section = calculate_rot_angles(rotated_cross_section, fit_method)
        #print(rot_angles)
        #final_rotated_cross_section = merge_rot_with_original(cross_section, rot_angles)
        # create starfile for final 
        #new_objects.append(propagate_rot_to_entire_cilia(final_rotated_cross_section, obj_data))
        new_objects.append(propagate_rot_to_entire_cilia(updated_cross_section, obj_data))
        
    create_starfile(new_objects, output)
            

if __name__ == "__main__":
    main()
