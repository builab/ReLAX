import starfile
import subprocess
import pandas as pd

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
        
        
def process_imod_point_file(input_file, spacing, angpix, tomo_angpix):
    """
    Reads IMOD .txt file, interpolates points, and computes angles.
    Returns a list of DataFrames, each representing points in the same object.
    """
    with open(input_file, "r") as file:
        lines = [list(map(float, line.strip().split())) for line in file]
    
    df = pd.DataFrame(lines, columns=["Object", "Filament", "X", "Y", "Z"])
    df[["X", "Y", "Z"]] *= angpix / tomo_angpix
    
    #print(df)
    objects = []
    for obj_id, group in df.groupby("Object"):
        results = []
        for filament_id, filament_group in group.groupby("Filament"):
            #print(filament_id)
            #print(filament_group[["X", "Y", "Z"]])
            interpolated_pts = interpolate_spline(filament_group[["X", "Y", "Z"]].values, tomo_angpix, spacing)
            #print(interpolated_pts)
            for i in range(len(interpolated_pts) - 1):
                vector = interpolated_pts[i + 1] - interpolated_pts[i]
                rot, tilt, psi = calculate_tilt_psi_angles(vector)
                results.append([int(obj_id - 1) * 10 + int(filament_id), *interpolated_pts[i] / angpix, rot, tilt, psi])
        
        objects.append(pd.DataFrame(results, columns=["rlnHelicalTubeID", "rlnCoordinateX", "rlnCoordinateY", "rlnCoordinateZ", "rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"]))
    
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