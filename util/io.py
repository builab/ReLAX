"""
IO package for ReLAX
Written by Molly Yu & Huy Bui, McGill 
"""

import matplotlib.pyplot as plt
import os
from typing import List, Dict, Union, Tuple, Optional
import pandas as pd
import numpy as np
import starfile

from util.imod import run_model2point, run_point2model, get_obj_ids_from_model, scale_imod_model
from util.geom import (
    robust_interpolate_spline, 
    calculate_tilt_psi_angles, 
    process_cross_section, 
    rotate_cross_section, 
    calculate_rot_angles, 
    propagate_rot_to_entire_cilia, 
    plot_ellipse_cs,
    fit_ellipse,
    renumber_filament_ids,
    get_filament_order,
    plot_cs
)

def create_dir(directory: str) -> None:
    """Create a directory if it does not exist."""
    try:
        os.makedirs(directory)
        print(f"Directory created: {directory}")
    except FileExistsError:
        print(f"Directory already exists: {directory}")

def get_filament_ids_from_object(cilia_object: List[pd.DataFrame], obj_id: int) -> List[int]:
    """
    Get the filament_id in the object file (1+ cilia).
    Args:
        cilia_object: List of df_star for each cilia.
        obj_id: Object id.
    Returns:
        List containing unique filament IDs sorted.
    """
    df_star = cilia_object[obj_id - 1]
    unique_sorted_tub_ids = df_star['rlnHelicalTubeID'].drop_duplicates().sort_values().tolist()
    return unique_sorted_tub_ids

def read_polarity_csv(polarity_file: str) -> Optional[pd.DataFrame]:
    """
    Read the polarity file. Ensures it has exactly 3 columns.
    Args:
        polarity_file: Path to the polarity CSV file.
    Returns:
        DataFrame with polarity data or None if error occurs.
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
    Look up polarity from the polarity dataframe.
    Args:
        df_polarity: DataFrame containing polarity information.
        tomo_name: Name of the tomogram.
        obj_id: Object ID to look up.
    Returns:
        Polarity value (0 or 1), or -1 if not found.
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
    Args:
        input_file: Path to the input .mod file.
        mod_suffix: Suffix to remove from the mod file name.
        spacing: Spacing for interpolation.
        angpix: Pixel size in Angstroms.
        tomo_angpix: Tomogram pixel size in Angstroms.
        df_polarity: DataFrame with polarity information.
    Returns:
        List of DataFrames, each representing points in the same object.
    """
    input_txt = input_file.replace(".mod", ".txt")
    run_model2point(input_file, input_txt)
    
    base_name = os.path.basename(input_file)
    tomo_name = base_name.removesuffix(mod_suffix + ".mod")
    
    with open(input_txt, "r") as file:
        lines = [list(map(float, line.strip().split())) for line in file]
    
    df = pd.DataFrame(lines, columns=["Object", "Filament", "X", "Y", "Z"])
    df[["X", "Y", "Z"]] *= angpix / tomo_angpix
    
    objects = []
    tomo_part_id_counter = 0
    
    for obj_id, group in df.groupby("Object"):
        results = []
        polarity = polarity_lookup(df_polarity, tomo_name, obj_id)
        polarity_prob = 1.0 if polarity >= 0 else 0.5
        
        print(f'Fitting {tomo_name} Cilia {obj_id} with polarity value of {polarity}')
        
        for filament_id, filament_group in group.groupby("Filament"):
            points = filament_group[["X", "Y", "Z"]].values
            if polarity == 1:
                points = np.flipud(points)
            interpolated_pts, cum_distances_angst = robust_interpolate_spline(points, tomo_angpix, spacing)
            
            for i in range(len(interpolated_pts) - 1):
                vector = interpolated_pts[i + 1] - interpolated_pts[i]
                rot, tilt, psi = calculate_tilt_psi_angles(vector)
                tomo_part_id = tomo_part_id_counter + i + 1
                helical_tube_id = (int(obj_id) - 1) * 10 + int(filament_id)
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
        df_list: List of DataFrames or a single DataFrame.
        output_star_file: Path to the output STAR file.
    """
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
    Args:
        input_star_file: Path to the input STAR file.
    Returns:
        List containing the star file object.
    """
    star_data = starfile.read(input_star_file)
    return [star_data]

def sanitize_particles_star(df_particles: pd.DataFrame, star_format: str, angpix: float, tomo_size: tuple) -> pd.DataFrame:
    """
    Drop unnecessary columns and add necessary columns for df_particles before writing.
    Args:
        df_particles: DataFrame containing particle data.
        star_format: Format of the STAR file ('warp', 'relion5', 'relion4').
    Returns:
        DataFrame with cleaned/prepared particle data.
    Note:
    """
    clean_df = df_particles.drop(columns=["rlnTomoParticleId", "rlnAnglePsiProbability"], errors='ignore')

    if star_format == 'warp':
        clean_df['rlnAngleTiltPrior'] = clean_df['rlnAngleTilt']
        clean_df['rlnAnglePsiPrior'] = clean_df['rlnAnglePsi']
        clean_df['rlnTomoName'] = clean_df['rlnTomoName'] + '.tomostar'
        return clean_df.drop(columns=['rlnCenteredCoordinateXAngst', 'rlnCenteredCoordinateYAngst', 'rlnCenteredCoordinateZAngst'], errors='ignore')
    elif star_format == 'relion5':
        clean_df['rlnAngleTiltPrior'] = clean_df['rlnAngleTilt']
        clean_df['rlnAnglePsiPrior'] = clean_df['rlnAnglePsi']
        clean_df = warp2relion5(clean_df, angpix, tomo_size)
        return clean_df.drop(columns=['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ'])
    else:
        print('Unrecognized or unsupported format. Star file format supported: \'relion5\' and \'warp\' only')

def warp2relion5(df_particles, angpix, tomo_size):
    """ Add column to warp star to relion5. Copy from Alister Burt
    """
    xyz = df_particles[['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']].to_numpy()
    volume_center = np.array(tomo_size) / 2 # replace with x/y/z dims of your tomogram
    xyz_centered = xyz - volume_center
    xyz_centered_angstroms = xyz_centered * angpix
    df_particles[['rlnCenteredCoordinateXAngst', 'rlnCenteredCoordinateYAngst', 'rlnCenteredCoordinateZAngst']] = xyz_centered_angstroms
    # Remove .tomostar
    df_particles['rlnTomoName'] = df_particles['rlnTomoName'].str.replace(r'\.tomostar$', '', regex=True)
    return df_particles
    
def relion2warp(df_particles, angpix, tomo_size):
    """ Add columns to relion5 star to warp
    """
    volume_center = np.array(tomo_size) / 2 # replace with x/y/z dims of your tomogram
    xyz_centered_angstroms = df_particles[['rlnCenteredCoordinateXAngst', 'rlnCenteredCoordinateYAngst', 'rlnCenteredCoordinateZAngst']].to_numpy() 
    xyz_centered = xyz_centered_angstroms / angpix
    xyz = xyz_centered + volume_center
    df_particles[['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']] = xyz
    # Add .tomostar
    df_particles['rlnTomoName'] = df_particles['rlnTomoName'].str.replace(
        r'^(.*)(?<!\.tomostar)$',  # Pattern: "PREFIX_DIGITS" (no .tomostar)
        r'\1.tomostar',                          # Appends .tomostar if needed
        regex=True
    )
    return df_particles
     
def create_data_optics(
    df_particles,
    tomo_angpix,
    angpix,
    Cs=2.7,
    voltage=300,
    ctf_premulti=1,
    img_dim=2,
    img_size=128,
    amp_contrast=0.07
):
    """
    Read df_particles and parameters
    Return df_optics in Relion 5 format
    """
    # Get unique sorted rlnTomoName values and assign OpticsGroup numbers
    unique_tomo_names = sorted(df_particles["rlnTomoName"].unique())
    
    # Create df_optics with rlnOpticsGroup as index (starting from 1)
    df_optics = pd.DataFrame(
        {
            "rlnOpticsGroup": range(1, len(unique_tomo_names) + 1),
            "rlnOpticsGroupName": [f"OpticsGroup{i}" for i in range(1, len(unique_tomo_names) + 1)],
            "rlnSphericalAberration": Cs,
            "rlnVoltage": voltage,
            "rlnTomoTiltSeriesPixelSize": tomo_angpix,
            "rlnCtfDataAreCtfPremultiplied": ctf_premulti,
            "rlnImageDimensionality": img_dim,
            "rlnTomoSubtomogramBinning": angpix / tomo_angpix,
            "rlnImagePixelSize": angpix,
            "rlnImageSize": img_size,
            "rlnAmplitudeContrast": amp_contrast
        }
    ).set_index("rlnOpticsGroup")

    
    return df_optics
    
def create_particles_starfile(df_optics, df_particles, output_star_file: str) -> None:
    """
    Write particles star file compatible with multi table
    """
    # Check if any optics group has 2D images
    if (df_optics["rlnImageDimensionality"] == 2).any():
        df_general = pd.DataFrame({"rlnTomoSubTomosAre2DStacks": [1]})
    else:
        df_general = pd.DataFrame()  # Empty if no 2D groups
    
    starfile.write(
        {'general': df_general, 'optics': df_optics, 'particles': df_particles}, output_star_file
    )
    
def add_particle_names(df_particles):
    """
    Adds column 'rlnTomoParticleName' with format: 
    - '{tomo_name}/1', '{tomo_name}/2', ... (if only 1 tomogram)
    - '{tomo_name}/1', '{tomo_name}/2', ... per tomogram (if multiple tomograms).
    Removes '.tomostar' from tomogram names if present.
    """
    # Remove '.tomostar' from rlnTomoName (if present)
    clean_tomo_name = df_particles['rlnTomoName'].str.replace(r'\.tomostar$', '', regex=True)
    
    # Case 1: Only 1 unique tomogram (assign sequential numbers)
    if len(clean_tomo_name.unique()) == 1:
        df_particles['rlnTomoParticleName'] = (
            clean_tomo_name + '/' + (df_particles.index + 1).astype(str)
        )
    
    # Case 2: Multiple tomograms (group and number particles per tomogram)
    else:
        df_particles['rlnTomoParticleName'] = (
            clean_tomo_name + '/' + 
            (df_particles.groupby(clean_tomo_name).cumcount() + 1).astype(str)
        )
    
    return df_particles
    
    
def process_object_data(
    obj_data: pd.DataFrame, 
    output_star_file: str, 
    fit_method: str, 
    reorder: bool, 
    obj_idx: int
) -> pd.DataFrame:
    """
    Process the cross section and calculate rotation angles for an object.
    Args:
        obj_data: DataFrame with object data.
        output_star_file: Path to the output STAR file.
        fit_method: Method for fitting.
        reorder: Whether to reorder points.
        obj_idx: Index of the object.
    Returns:
        DataFrame with processed data.
    """
    
    cross_section = process_cross_section(obj_data)
    create_starfile([cross_section], output_star_file.replace('.star', '_cs.star'))

    rotated_cross_section = rotate_cross_section(cross_section)
    updated_cross_section = calculate_rot_angles(rotated_cross_section, fit_method)

    # Drawing path
    output_cs = output_star_file.replace(".star", f"_Cilia{obj_idx + 1}.png") 
    ellipse_params = fit_ellipse(rotated_cross_section['rlnCoordinateX'], rotated_cross_section['rlnCoordinateY'])       
    if ellipse_params['a'] is None or ellipse_params['b'] is None:
        print(f"WARNING: Ellipse fitting failed for object {obj_idx}. Skipping plot.")
    else:
        plot_ellipse_cs(rotated_cross_section, output_cs)  # creates the PLOT
    
    df_star = propagate_rot_to_entire_cilia(updated_cross_section, obj_data)  
    
    sorted_filament_ids = get_filament_order(updated_cross_section, fit_method)
        
    if reorder:
        print('Reorder the doublet number: ', sorted_filament_ids)
        df_sorted, sorted_cross_section = renumber_filament_ids(df_star, sorted_filament_ids, updated_cross_section)
        return df_sorted
    
    return df_star


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
        input_file: Path to the input .mod file.
        output_star_file: Path to the output .star file.
        angpix: Pixel size in Angstroms.
        tomo_angpix: Tomogram pixel size in Angstroms.
        spacing: Spacing for interpolation.
        fit_method: Method for fitting.
        df_polarity: DataFrame with polarity information.
        mod_suffix: Suffix to remove from the mod file name.
        reorder: Whether to reorder points.
    Returns:
        List of DataFrames with processed data.
    """
    print(f'Input model file: {input_file}')
    print(f'Output star file: {output_star_file}')
    
    objects = process_imod_point_file(input_file, mod_suffix, spacing, angpix, tomo_angpix, df_polarity)
    new_objects = [process_object_data(obj_data, output_star_file, fit_method, reorder, i) for i, obj_data in enumerate(objects)]
    
    create_starfile(new_objects, output_star_file)
    return new_objects
    
def make_common_star(df_particles, angpix, tomo_size):
    """Script to add column in relion5 and warp format"""
    # Check if star_file is relion5 or warp
    if 'rlnCoordinateX' in df_particles.columns:
        star_format = 'warp'
    elif 'rlnCenteredCoordinateXAngst' in df_particles.columns:
        star_format = 'relion5'
    print(f'Star file format is {star_format}')
    
    if star_format == 'warp':
        return warp2relion5(df_particles, angpix, tomo_size)
    elif star_format == 'relion5':
        return relion2warp(df_particles, angpix, tomo_size)
    
