"""
IO package for ReLAX
Written by Molly Yu & Huy Bui, McGill 
"""

import starfile
import subprocess
import pandas as pd
import numpy as np
import os
from typing import List, Dict, Union, Tuple, Optional

from util.geom import (
    interpolate_spline, 
    calculate_tilt_psi_angles, 
    process_cross_section, 
    rotate_cross_section, 
    calculate_rot_angles, 
    propagate_rot_to_entire_cilia, 
    plot_ellipse_cs
)

def create_dir(directory):
    """Create a directory if it does not exist."""
    try:
        os.makedirs(directory)
        print(f"Directory created: {directory}")
    except FileExistsError:
        print(f"Directory already exists: {directory}")  

def run_imod_command(command: List[str], input_file: str, output_file: str) -> bool:
    """
    Runs an IMOD command with the given input and output files.
    
    Args:
        command: List of command arguments
        input_file: Path to the input file
        output_file: Path to the output file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error processing {input_file}: {e.stderr.decode()}")  
        return False

def run_model2point(input_mod: str, output_txt: str) -> bool:
    """
    Runs the IMOD model2point command with the given input and output files.
    
    Args:
        input_mod: Path to the input .mod file
        output_txt: Path to the output .txt file
        
    Returns:
        bool: True if successful, False otherwise
    """
    command = ["model2point", "-Object", "-Contour", input_mod, output_txt]
    return run_imod_command(command, input_mod, output_txt)
        
def run_point2model(input_txt: str, output_mod: str) -> bool:
    """
    Runs the IMOD point2model command with the given input and output files.
    
    Args:
        input_txt: Path to the input .txt file
        output_mod: Path to the output .mod file
        
    Returns:
        bool: True if successful, False otherwise
    """
    command = ["point2model", input_txt, output_mod]
    return run_imod_command(command, input_txt, output_mod)

def get_obj_ids_from_model(model_file):
    """
    Get the obj_id in the IMOD model file
    Args:
        model_file: Path to the input .mod file
    
    Returns:
        obj_ids: returns a set object containing unique id    
    """
    input_txt = model_file.replace(".mod", ".txt")
    run_model2point(model_file, input_txt)
    # Extract tomogram name
    with open(input_txt, "r") as file:
        lines = [list(map(float, line.strip().split())) for line in file]
    obj_list = [row[0] for row in lines]
    return set(obj_list)
    
    
def get_filament_ids_from_object(cilia_object, obj_id):
    """
    Get the filament_id in the object file (1+ cilia)
    Args:
        cilia object: List of df_star for each cilia
        obj_id: object id
    Returns:
        filament_ids: returns a sorted set filament containing unique id    
    """
    df_star = cilia_object[obj_id - 1]
    unique_sorted_tub_ids = df_star['rlnHelicalTubeID'].drop_duplicates().sort_values().tolist()
    return unique_sorted_tub_ids
    
def read_polarity_csv(polarity_file: str) -> Optional[pd.DataFrame]:
    """
    Read the polarity file. Ensures it has exactly 3 columns.
    
    Args:
        polarity_file: Path to the polarity CSV file
        
    Returns:
        DataFrame with polarity data or None if error occurs
    """
    try:
        df_polarity = pd.read_csv(polarity_file)     
        if df_polarity.shape[1] != 3:
            raise ValueError(f"Incorrect number of columns in {polarity_file}: Expected 3, got {df_polarity.shape[1]}")
        
        return df_polarity

    except Exception as e:
        print(f"Error reading {polarity_file}: {e}")
        return None
        
def polarity_lookup(df_polarity: pd.DataFrame, tomo_name: str, obj_id: int) -> int:
    """
    Look up polarity from the polarity dataframe
    
    Args:
        df_polarity: DataFrame containing polarity information
        tomo_name: Name of the tomogram
        obj_id: Object ID to look up
        
    Returns:
        int: Polarity value (0 or 1), or -1 if not found
    """
    try:
        mask = (df_polarity['rlnTomoName'] == tomo_name) & (df_polarity['ObjectID'] == obj_id)
        polarity = df_polarity.loc[mask, 'Polarity'].values[0]
    except IndexError:
        print(f"No polarity found for tomo {tomo_name} and object {obj_id}.")
        polarity = -1
    return polarity
              
def process_imod_point_file(
    input_file: str, 
    mod_suffix: str, 
    spacing: float, 
    angpix: float, 
    tomo_angpix: float, 
    df_polarity: pd.DataFrame
) -> List[pd.DataFrame]:
    """
    Reads IMOD .txt file, interpolates points, and computes angles.
    
    During interpolation, ReLAX stores subtomo coordinates in unbinned pixel.
    When output, stored in current pixel size coordinate like WARP.
    
    Args:
        input_file: Path to the input .mod file
        mod_suffix: Suffix to remove from the mod file name
        spacing: Spacing for interpolation
        angpix: Pixel size in Angstroms
        tomo_angpix: Tomogram pixel size in Angstroms
        df_polarity: DataFrame with polarity information
        
    Returns:
        List of DataFrames, each representing points in the same object
    """
    # Convert IMOD to txt file
    input_txt = input_file.replace(".mod", ".txt")
    run_model2point(input_file, input_txt)
    
    # Extract tomogram name
    base_name = os.path.basename(input_file)
    tomo_name = base_name.removesuffix(mod_suffix + ".mod")
    
    # Read the txt file
    with open(input_txt, "r") as file:
        lines = [list(map(float, line.strip().split())) for line in file]
    
    # Create and process DataFrame
    df = pd.DataFrame(lines, columns=["Object", "Filament", "X", "Y", "Z"])
    df[["X", "Y", "Z"]] *= angpix / tomo_angpix
    
    objects = []
    tomo_part_id_counter = 0
    
    # Process each object
    for obj_id, group in df.groupby("Object"):
        results = []
        polarity = polarity_lookup(df_polarity, tomo_name, obj_id)
        
        # Set polarity probability
        polarity_prob = 1.0 if polarity >= 0 else 0.5
        
        print(f'Fitting {tomo_name} Cilia {obj_id} with polarity value of {polarity}')
        
        # Process each filament in the object
        for filament_id, filament_group in group.groupby("Filament"):
            points = filament_group[["X", "Y", "Z"]].values
            
            # Apply polarity if needed
            if polarity == 1:
                points = np.flipud(points)
                
            # Interpolate points
            interpolated_pts, cum_distances_angst = interpolate_spline(points, tomo_angpix, spacing)

            # Calculate angles for each interpolated point
            for i in range(len(interpolated_pts) - 1):
                vector = interpolated_pts[i + 1] - interpolated_pts[i]
                rot, tilt, psi = calculate_tilt_psi_angles(vector)
                
                # Assign ID and create image name
                tomo_part_id = tomo_part_id_counter + i + 1
                helical_tube_id = (int(obj_id) - 1) * 10 + int(filament_id)
                
                # Scale coordinates back to tomogram pixel size
                coords = interpolated_pts[i] / angpix * tomo_angpix
                
                results.append([
                    tomo_name, 
                    helical_tube_id, 
                    cum_distances_angst[i],
                    *coords, 
                    rot, 
                    tilt, 
                    psi, 
                    tomo_part_id, 
                    polarity_prob
                ])
                
            tomo_part_id_counter = tomo_part_id
            
        # Create DataFrame for this object
        columns = [
            "rlnTomoName", 
            "rlnHelicalTubeID", 
            "rlnHelicalTrackLengthAngst", 
            "rlnCoordinateX", 
            "rlnCoordinateY", 
            "rlnCoordinateZ", 
            "rlnAngleRot", 
            "rlnAngleTilt", 
            "rlnAnglePsi", 
            "rlnTomoParticleId", 
            "rlnAnglePsiProbability"
        ]
        objects.append(pd.DataFrame(results, columns=columns))
    
    return objects 
        
def create_starfile(df_list: Union[List[pd.DataFrame], pd.DataFrame], output_star_file: str) -> None:
    """
    Saves list of DataFrames or a single DataFrame to a STAR file.
    
    Args:
        df_list: List of DataFrames or a single DataFrame
        output_star_file: Path to the output STAR file
    """
    # Check if input is a list or a single DataFrame
    if isinstance(df_list, list):
        df_merged = pd.concat(df_list, ignore_index=True)
        star_data = {"particles": df_merged}
    else:
        star_data = {"particles": df_list}

    starfile.write(star_data, output_star_file, overwrite=True)
    print(f"Successfully created STAR file: {output_star_file}")
    
def read_starfile_into_cilia_object(input_star_file: str) -> List:
    """
    Read star file and separate into object for easy processing.
    
    TEMPORARY: Not done for > 1 cilia yet and nothing regarding OpticsGroup block
    
    Args:
        input_star_file: Path to the input STAR file
        
    Returns:
        List containing the star file object
    """
    obj = starfile.read(input_star_file)
    return [obj]

def sanitize_particles_star(df_particles: pd.DataFrame, star_format: str) -> pd.DataFrame:
    """
    Drop unnecessary columns and add necessary columns for df_particles before writing.
    
    Args:
        df_particles: DataFrame containing particle data
        star_format: Format of the STAR file ('warp', 'relion5', 'relion4')
        
    Returns:
        DataFrame with cleaned/prepared particle data
        
    Note:
        NOT YET FULLY IMPLEMENTED
    """
    # Drop temporary columns
    clean_df = df_particles.drop(columns=["rlnTomoParticleId", "rlnAnglePsiProbability"])
    
    # Add prior angle columns
    clean_df['rlnAngleTiltPrior'] = clean_df['rlnAngleTilt']
    clean_df['rlnAnglePsiPrior'] = clean_df['rlnAnglePsi']
    
    # In the future, probably also add rlnTomoSubtomogramTilt/Psi/Rot
    return clean_df

def imod2star(
    input_file: str, 
    output_star_file: str, 
    angpix: float, 
    tomo_angpix: float, 
    spacing: float, 
    fit_method: str, 
    df_polarity: pd.DataFrame, 
    mod_suffix: str, 
    reorder: bool
) -> List[pd.DataFrame]:
    """
    Convert IMOD model file to STAR file.
    
    Args:
        input_file: Path to the input .mod file
        output_star_file: Path to the output .star file
        angpix: Pixel size in Angstroms
        tomo_angpix: Tomogram pixel size in Angstroms
        spacing: Spacing for interpolation
        fit_method: Method for fitting
        df_polarity: DataFrame with polarity information
        mod_suffix: Suffix to remove from the mod file name
        reorder: Whether to reorder points
        
    Returns:
        List of DataFrames with processed data
    """
    print(f'Input model file: {input_file}')
    print(f'Output star file: {output_star_file}')
    
    # Interpolation, estimate psi and tilt
    objects = process_imod_point_file(input_file, mod_suffix, spacing, angpix, tomo_angpix, df_polarity)  
        
    new_objects = [] 
    
    # Process each object
    for i, obj_data in enumerate(objects):
        # Process cross section
        cross_section = process_cross_section(obj_data)
        rotated_cross_section = rotate_cross_section(cross_section)
        
        # Plot cross section
        output_cs = output_star_file.replace(".star", f"_Cilia{i+1}.png")
        plot_ellipse_cs(rotated_cross_section, output_cs)

        # Calculate rotation angles
        updated_cross_section = calculate_rot_angles(rotated_cross_section, fit_method)

        # Propagate rotation to the entire cilia
        new_objects.append(propagate_rot_to_entire_cilia(updated_cross_section, obj_data))
        
    # Create final star file
    create_starfile(new_objects, output_star_file)
    return new_objects