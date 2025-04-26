#!/usr/bin/env python3
# Script to convert IMOD star file to Relion 5 star file
# Not yet testing now with 2 cilia
# Authors: Molly & HB, 02/2025
# python ~/Documents/GitHub/ReLAX/relax_imod2star.py --i input_mod --o particles.star --tomo_angpix 3.37 --angpix 14.00 --mod_suffix _14.00Apx_doublets.mod --fit ellipse
# NOTE: a & b axis might be reversed

from util.io import create_dir, create_starfile, create_particles_starfile, imod2star, read_polarity_csv, sanitize_particles_star, add_particle_names, create_data_optics

import argparse
import numpy as np
import pandas as pd
import starfile
import glob
import os, sys

# main
def main():

    parser = argparse.ArgumentParser(description="Process input file and generate STAR files.")
    
    # Add arguments with default values
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the input dir containing IMOD models")
    parser.add_argument("--output_dir", type=str, default="outstar", help="Output STAR file directory")
    parser.add_argument("--spacing", type=float, default=82.0, help="Repeating unit spacing (default: 82.0)")
    parser.add_argument("--angpix", type=float, default=8.48, help="Pixel size in Angstroms (default: 8.48)")
    parser.add_argument("--tomo_angpix", type=float, default=2.12, help="Pixel size of unbinned tomogram in Angstroms (default: 2.12)")
    parser.add_argument("--fit", type=str, default="simple", help="Fitting type: simple or ellipse")
    parser.add_argument("--polarity", type=str, default="", help="Polarity file for angle prediction.")
    parser.add_argument("--reorder", action="store_true", help="Reorder filament using ellipse fit with correct polarity")
    parser.add_argument("--mod_suffix", type=str, default="", help="Suffix of IMOD models without .mod")
    parser.add_argument("--star_format", type=str, default="warp", help="Star file format (warp|relion5)")
    parser.add_argument("--tomo_size", type=lambda s: tuple(map(int, s.split(','))), default=(1023, 1440, 500), 
                    help="Size of the tomogram generated as 'x,y,z'")
    parser.add_argument("--write_particles", action="store_true", help="Write particles.star file if this flag is set")

    # Parse arguments
    args = parser.parse_args()

    input_dir=args.input_dir
    spacing=args.spacing
    angpix=args.angpix
    output_dir=args.output_dir
    fit_method = args.fit
    tomo_angpix = args.tomo_angpix
    reorder = args.reorder
    
    print(f'Model suffix: {args.mod_suffix}')
    print(f'Pixel size of input model: {angpix} Angstrom')
    print(f'Output star file format: \"{args.star_format}\"')
    print(f'Pixel size of original tomogram: {tomo_angpix} Angstrom')
    print(f'Fitting method: {fit_method}')
    print(f'Reorder doublet: {reorder}')
    print(f'Default tomo Size: {args.tomo_size}')
    
    if args.star_format not in ['warp', 'relion5']:
        print(f"ERROR: Unsupported star format '{args.star_format}'. Must be 'warp' or 'relion5'.")
        sys.exit(1)
    
    print(f'Particles star file format: {args.star_format}')

    print(f'Repeating unit to interpolate: {spacing} Angstrom')
    if args.polarity != "":
        print(f'Polarity file: {args.polarity} ')
    else:
        print(f'Fitting without polarity') 
    print(f'Write particles.star: {args.write_particles}')

    create_dir(output_dir)
    
    if args.polarity != "":
        df_polarity = read_polarity_csv(args.polarity)
    else:
        df_polarity = pd.DataFrame(columns=['rlnTomoName', 'ObjectID', 'Polarity'])

    pattern = os.path.join(input_dir, f"*{args.mod_suffix}.mod")
    
    # Use glob to find all files matching the pattern & sort
    matching_files = glob.glob(pattern)
    matching_files.sort()
    
    if not matching_files:
        print(f"No files found matching the pattern: {args.mod_suffix} in directory {input_dir}")
        sys.exit(1)  # Exit the program with a non-zero status code
    
    # 20250331
    df_particles = []
    for input_file in matching_files:
        filename = os.path.basename(input_file)
        tomo_name = filename.removesuffix(args.mod_suffix + ".mod")
        output_star_file = os.path.join(output_dir, tomo_name + ".star")
        print(f'-----> Processing {filename} <-----')
        df_cilia = imod2star(input_file, output_star_file, angpix, tomo_angpix, spacing, fit_method, df_polarity, args.mod_suffix, reorder)
        for i, obj_data in enumerate(df_cilia):
            df_particles.append(obj_data)

    if args.write_particles:
        print('----- Writing combined particle file -----')
        df_all_particles = sanitize_particles_star(pd.concat(df_particles, ignore_index=True), args.star_format, angpix, args.tomo_size)
        df_all_particles = add_particle_names(df_all_particles)
        df_optics = create_data_optics(
            df_particles=df_all_particles,
            tomo_angpix=tomo_angpix,  # Tilt series pixel size (Å)
            angpix=angpix,       # Subtomogram pixel size (Å)
            Cs=2.7,            # Spherical aberration (mm)
            voltage=300       # Microscope voltage (kV)
        )
        particlesfile = f'particles_{args.star_format}.star'
        create_particles_starfile(df_optics, df_all_particles, particlesfile)
        print(f'{particlesfile} successfuly written!')
    
if __name__ == "__main__":
    main()