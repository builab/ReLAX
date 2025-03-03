"""
IO package for ReLAX
Written by Molly Yu, McGill 
TODO: In the future, make option for create_starfile with Warp or Relion4 way or perhaps Relion 5 way
"""

import starfile
import subprocess
import pandas as pd
import numpy as np
import os

"""
The X, Y, Z should be calculated using unbinned pixel
"""

from util.geom import interpolate_spline, calculate_tilt_psi_angles

def run_model2point(input_mod, output_txt):
    """
    Runs the IMOD model2point command with the given input and output files.
    Args:
        input_mod (str): Path to the input .mod file.
        output_txt (str): Path to the output .txt file.
    """
    # Define the command as a list of arguments
    """
    Runs IMOD model2point command.
    """
    command = ["model2point", "-Object", "-Contour", input_mod, output_txt]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Success: {input_mod} processed and saved to {output_txt}")
    except subprocess.CalledProcessError as e:
        print(f"Error processing {input_mod}: {e.stderr.decode()}")  
        
def run_point2model(input_txt, output_mod):
    """
    Runs the IMOD point2model command with the given input and output files.
    Args:
        input_txt (str): Path to the input .mod file.
        output_mod (str): Path to the output .txt file.
    """
    command = ["point2model", input_txt, output_mod]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Success: {input_txt} processed and saved to {output_mod}")
    except subprocess.CalledProcessError as e:
        print(f"Error processing {input_txt}: {e.stderr.decode()}")  
              
def process_imod_point_file(input_file, mod_suffix, spacing, angpix, tomo_angpix, polarity_file):
    """
    Reads IMOD .txt file, interpolates points, and computes angles.
    Returns a list of DataFrames, each representing points in the same object.
    Implement polarity file, if 0, keep the same. If 1, flip base on y. polarity = -1 (no polarity)
    """
    # Convert IMOD to txt file
    input_txt = input_file.replace(".mod", ".txt")
    run_model2point(input_file, input_txt)
    
    # TomoName = base_name(input_file)
    base_name = os.path.basename(input_file)
    #tomo_name = os.path.splitext(base_name)[0]
    tomo_name = base_name.removesuffix(mod_suffix)
    
    with open(input_txt, "r") as file:
        lines = [list(map(float, line.strip().split())) for line in file]
    
    df = pd.DataFrame(lines, columns=["Object", "Filament", "X", "Y", "Z"])
    df[["X", "Y", "Z"]] *= angpix / tomo_angpix
    
    #print(df)
    objects = []
    count = 0
    df_polarity = pd.DataFrame(columns=['rlnTomoName', 'ObjectId', 'Polarity'])
    
    if polarity_file != "":
        df_polarity = pd.read_csv('polarity.csv', header=None, names=['rlnTomoName', 'ObjectId', 'Polarity'])
    
    for obj_id, group in df.groupby("Object"):
        results = []
        # Match polarity
        try:
            polarity = df_polarity.loc[(df_polarity['rlnTomoName'] == tomo_name) & (df_polarity['ObjectId'] == obj_id), 'Polarity'].values[0]
            print(f"Found polarity of tomo {tomo_name} and object {obj_id}: {polarity}")
            polarity_prob = 1
        except IndexError:
            print(f"No polarity found for tomo {tomo_name} and object {obj_id}.")
            polarity_prob = 0.5
            polarity = -1
        
        for filament_id, filament_group in group.groupby("Filament"):
            #print(filament_id)
            #print(filament_group[["X", "Y", "Z"]])
            if polarity < 1:
                interpolated_pts = interpolate_spline(filament_group[["X", "Y", "Z"]].values, tomo_angpix, spacing)
            else: # Polarity 1
                interpolated_pts = interpolate_spline(np.flipud(filament_group[["X", "Y", "Z"]].values), tomo_angpix, spacing)

            #print(interpolated_pts)
            for i in range(len(interpolated_pts) - 1):
                vector = interpolated_pts[i + 1] - interpolated_pts[i]
                rot, tilt, psi = calculate_tilt_psi_angles(vector)
                tomo_part_id = count + i + 1
                image_name = f"{tomo_name}/{tomo_part_id:d}"
                results.append([int(obj_id - 1) * 10 + int(filament_id), *interpolated_pts[i] / angpix * tomo_angpix, rot, tilt, psi, tomo_name, tomo_part_id, image_name, polarity_prob])
            count = tomo_part_id
        objects.append(pd.DataFrame(results, columns=["rlnHelicalTubeID", "rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ", "rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi", "rlnTomoName", "rlnTomoParticleId", "rlnTomoParticleName", "rlnAnglePsiProbability"]))
    
    return objects 
     
        
def create_starfile(df_list, output_star_file):
    """
    Saves list of DataFrames to a STAR file.
    Can work with a straight df as well
    """
    # Check if it is a list then combine
    if isinstance(df_list, list):
        df_merged = pd.concat(df_list, ignore_index=True)
        star_data = {"particles": df_merged}
    else:
        star_data = {"particles": df_list}

    starfile.write(star_data, output_star_file, overwrite=True)
    print(f"Successfully created STAR file: {output_star_file}")
    
def read_starfile_into_cilia_object(input_star_file):
    """
    Read star file and separate into object for easy processing
    TEMPORARY
    Not done for > 1 cilia yet and nothing regarding OpticsGroup block
    """
    obj = starfile.read(input_star_file)
    return [obj]
    