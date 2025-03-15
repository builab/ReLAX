#!/usr/bin/env python3
# Quantify cilia distortion by reading aligned particles.star and output the elliptical fit
# Fut
# Using Ellipse Fitting

from util.io import create_dir, create_starfile, process_cross_section
from util.geom import  plot_ellipse_cs, rotate_cross_section
import argparse
import numpy as np
import pandas as pd
import starfile
import glob
import os

# main
def main():

    parser = argparse.ArgumentParser(description="Process input file and generate STAR files.")
    
    # Add arguments with default values
    parser.add_argument("--input_star", type=str, required=True, help="Path to the particle star file")
    parser.add_argument("--output_dir", type=str, default="graph", help="Output graph file directory")
    parser.add_argument("--angpix", type=float, default=8.48, help="Pixel size in Angstroms (default: 8.48)")
    parser.add_argument("--star_format", type=str, default="relion5", help="Suffix of IMOD models without .mod")
    parser.add_argument("--tomo_size", type=lambda s: tuple(map(int, s.split(','))), default=(1023, 1440, 500), 
                    help="Size of the tomogram generated as 'x,y,z'")
                    
    print('Ignore the Relion5 format for now')
    
    # Parse arguments
    args = parser.parse_args()

    out_csv = 'cilia_distortion.csv'
    
    # Remove pixel size later, reading from star file
    print(f'Input file: {args.input_star}')
    print(f'Pixel size of file: {args.angpix} Angstrom')
    print(f'Output graph dir: \"{args.output_dir}\"')
    print(f'Star file format: {args.star_format}')
    print(f'Elliptical Distortion List: {out_csv}')


    create_dir(args.output_dir)
    
    df_particles = starfile.read(args.input_star)
    # Check if it is the one with optics group or not
    # For now, just have no optics group
    # Convert from Relion format to warp format
    # Taking care of with & without rlnOriginX,Y,Z
    
    # Densify the star file (guess the polarity based on Psi angle)
    
    
    bins = [0, 10, 20, 30]  # Define the bin edges
    labels = ['1', '2', '3']  # Define the labels for the bins
    df_particles['HelicalTubeID_Range'] = pd.cut(df_particles['rlnHelicalTubeID'], bins=bins, labels=labels)

    grouped = df_particles.groupby(['rlnTomoName', 'HelicalTubeID_Range'])

    distortion_list = []
    for (tomo_name, cilia_id), df_cilia in grouped:

    #for tomo_name, tomo in df_particles.groupby('rlnTomoName'):
       print(f"-----> Processing tomo: {tomo_name} and Cilia {cilia_id}")
       # Not dealing with > 1 cilia right now as well
       # For > 1 cilia, separate by rlnHelicalTubeID
       #max_filamentID = df_cilia['rlnHelicalTubeID'].max()
       #print(f'Maximum no. of HelicalTubeID {max_filamentID}')

       cross_section = process_cross_section(df_cilia)
       rotated_cross_section = rotate_cross_section(cross_section)
       #print(rotated_cross_section[['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']])
       # Multiply to generate Angstrom Coordinate
       rotated_cross_section[['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']] = rotated_cross_section[['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']] * args.angpix / 10

       output_cs = os.path.join(args.output_dir, tomo_name.replace('tomostar', '') + '_Cilia' + cilia_id + '.png')
       print(f'Writing graph {output_cs} for {tomo_name}')
       elliptical_distortion = plot_ellipse_cs(rotated_cross_section, output_cs)
       distortion_list.append({'rlnTomoName': tomo_name, 'ObjectID': cilia_id, 'EllipticalDistortion': elliptical_distortion})
            # Perform alignment as well

    # Convert the results to a DataFrame
    distortion_df = pd.DataFrame(distortion_list)
    distortion_df.to_csv(out_csv, index=False, float_format='%.3f')
    print(f'Writing elliptical distortion to {out_csv} ')


if __name__ == "__main__":
    main()