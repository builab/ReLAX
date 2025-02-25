#!/usr/bin/python
# Script to convert IMOD star file to Relion 5 star file
# Testing now with 2 cilia
# Authors: Molly & HB, 02/2025
# TODO: We need to plot in function for calculate_rot_angles, also, calculate a elliptically
# TODO: Draw the filament in the propagate_rot_to_entire_cilia as well for visualization

# To check if HelicalTubuleID = 11-19 also work
import argparse
import numpy as np
import pandas as pd
from scipy.interpolate import splprep, splev
import starfile
import subprocess

def interpolate_spline(points, angpix, spacing):
    """
    Interpolate points along a line with a specified spacing using splines.
    TODO: Need to deal with error if this is too short? Min 5 particles?
    """
    points = np.array(points)*angpix
    #print(points)
    tck, _ = splprep(points.T, s=0)  # Create spline representation
    distances = np.linspace(0, 1, int(np.ceil(points.shape[0] * 100)))  # Dense samples. 100 times the points?
    
    # Evaluate the spline at dense samples
    dense_points = np.array(splev(distances, tck)).T
    
    # Calculate cumulative distances along the spline
    diffs = np.diff(dense_points, axis=0)
    cumulative_distances = np.insert(np.cumsum(np.sqrt((diffs**2).sum(axis=1))), 0, 0)
    
    # Resample at equal intervals
    resampled_distances = np.arange(0, cumulative_distances[-1], spacing)
    return np.array([np.interp(resampled_distances, cumulative_distances, dense_points[:, i]) for i in range(3)]).T
    
def calculate_tilt_psi_angles(v):
    """
    Calculate the ZYZ Euler angles (Rot, Tilt, Psi) for a vector v.
    Rot is not calculated in this function yet
    These angles rotate the vector to align with the Z-axis.
    """
    v = v / np.linalg.norm(v)
    rot = 0
    #tilt = np.pi - np.arccos(v[2])
    #psi = -np.pi/2 + np.arctan2(v[0], v[1]) # Old formula
    tilt = np.arccos(-v[2])
    psi = np.arctan2(-v[1], v[0])
    return np.degrees(rot), np.degrees(tilt), np.degrees(psi)
    

def process_input_file(input_file, spacing, angpix, tomo_angpix):
    """
    Reads IMOD .txt file, interpolates points, and computes angles.
    Returns a list of DataFrames, each representing points in the same object.
    """
    with open(input_file, "r") as file:
        lines = [list(map(float, line.strip().split())) for line in file]
    
    df = pd.DataFrame(lines, columns=["Object", "Filament", "X", "Y", "Z"])
    df[["X", "Y", "Z"]] *= angpix / tomo_angpix
    
    
    objects = []
    for obj_id, group in df.groupby("Object"):
        results = []
        for filament_id, filament_group in group.groupby("Filament"):
            interpolated_pts = interpolate_spline(filament_group[["X", "Y", "Z"]].values, angpix, spacing)
            
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

def define_plane(normal_vector, reference_point):
    return normal_vector, reference_point
    
def calculate_perpendicular_distance(point, plane_normal, reference_point):
    return np.abs(np.dot(plane_normal, point - reference_point)) / np.linalg.norm(plane_normal)

def find_cross_section_points(data, plane_normal, reference_point):
    """
    Find the points on each filament closest to the cross-sectional plane,
    add 90 degrees to the Psi angle, and return them.
    """
    cross_section_points = []
    grouped = data.groupby('rlnHelicalTubeID')

    for filament_id, group in grouped:
        points = group[['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']].values
        distances = np.array([calculate_perpendicular_distance(point, plane_normal, reference_point) for point in points])
        closest_idx = np.argmin(distances)
        closest_point = group.iloc[closest_idx]

        # Add 90 degrees to the Psi angle
        #Psi = closest_point['rlnAnglePsi'] + 90.0

        cross_section_points.append({
            'rlnHelicalTubeID': filament_id,
            'rlnCoordinateX': closest_point['rlnCoordinateX'],
            'rlnCoordinateY': closest_point['rlnCoordinateY'],
            'rlnCoordinateZ': closest_point['rlnCoordinateZ'],
            'rlnAngleRot': 0,
            'rlnAngleTilt': closest_point['rlnAngleTilt'],
            'rlnAnglePsi': closest_point['rlnAnglePsi'],
        })
    return pd.DataFrame(cross_section_points)


def find_shortest_filament(data):
    grouped = data.groupby('rlnHelicalTubeID')
    shortest_length, shortest_midpoint, shortest_filament_id = float('inf'), None, None
    for filament_id, group in grouped:
        filament_points = group[['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']].values
        min_point, max_point = filament_points.min(axis=0), filament_points.max(axis=0)
        length = np.linalg.norm(max_point - min_point)
        if length < shortest_length:
            shortest_length, shortest_midpoint, shortest_filament_id = length, (min_point + max_point) / 2, filament_id
    return shortest_filament_id, shortest_midpoint

def calculate_normal_vector(filament_points):
    vectors = np.diff(filament_points, axis=0)
    normal_vector = np.sum(vectors, axis=0)
    return normal_vector / np.linalg.norm(normal_vector)

def process_cross_section(data):
    shortest_filament_id, shortest_midpoint = find_shortest_filament(data)
    filament_points = data[data['rlnHelicalTubeID'] == shortest_filament_id][['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']].values
    normal_vector = calculate_normal_vector(filament_points)
    plane_normal, plane_point = define_plane(normal_vector, shortest_midpoint)
    return find_cross_section_points(data, plane_normal, plane_point)

def rotate_cross_section(cross_section_points):
    """
    Rotates cross-section points by Psi and Tilt angles to transform into Z plane
    """
    psi = 90 - cross_section_points['rlnAnglePsi'].median()
    tilt = cross_section_points['rlnAngleTilt'].median()
    final_rotated_points = []
    rotated_points = []
    for (_, row) in cross_section_points.iterrows():
        x, y, z = row['rlnCoordinateX'], row['rlnCoordinateY'], row['rlnCoordinateZ']
        
        # Rotation around Z-axis by Psi
        psi_rad = np.radians(psi)
        Rz = np.array([
            [np.cos(-psi_rad), -np.sin(-psi_rad), 0],
            [np.sin(-psi_rad),  np.cos(-psi_rad), 0],
            [0,                0,               1]
        ])
        rotated_point = Rz @ np.array([x, y, z])
        
        # Rotation around X-axis by Tilt
        tilt_rad = np.radians(tilt)
        Ry = np.array([
            [1, 0,              0             ],
            [0, np.cos(-tilt_rad), -np.sin(-tilt_rad)],
            [0, np.sin(-tilt_rad),  np.cos(-tilt_rad)]
        ])
        rotated_point = Ry @ rotated_point
        
        final_rotated_points.append({'rlnCoordinateX': rotated_point[0], 'rlnCoordinateY': rotated_point[1], 'rlnCoordinateZ': rotated_point[2], 'rlnAngleRot' : row['rlnAngleRot'], 'rlnAngleTilt' : 0, 'rlnAnglePsi' : 0, 'rlnHelicalTubeID': row['rlnHelicalTubeID']})
    
    return pd.DataFrame(final_rotated_points)
    
def calculate_rot_angles(rotated_cross_section):
	""" Calculate the rotation angle in a cross section """
    updated_cross_section = rotated_cross_section
    rot_angles = []
    n = len(rotated_cross_section)
    for i in range(n):
        prev_idx, next_idx = (i - 1) % n, (i + 1) % n
        x_prev, y_prev = rotated_cross_section.iloc[prev_idx][['rlnCoordinateX', 'rlnCoordinateY']]
        x_next, y_next = rotated_cross_section.iloc[next_idx][['rlnCoordinateX', 'rlnCoordinateY']]
        delta_x, delta_y = x_next - x_prev, y_next - y_prev
        rot = -np.degrees(np.arctan2(delta_x, delta_y))
        rot_angles.append(rot)
    
    updated_cross_section['rlnAngleRot'] = rot_angles   
    return updated_cross_section

#def merge_rot_with_original(cross_section, rot_angles):
#    """ In principal, this function is not needed """
#    cross_section['rlnAngleRot'] = rot_angles
#    return cross_section  

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

def propagate_rot_to_entire_cilia(cross_section, original_data):
    # Create mappings for adjusted values
    # In princial, this is not needed
    #psi_mapping = cross_section_final.set_index('rlnHelicalTubeID')['rlnAnglePsi'].to_dict()
    rot_mapping = cross_section.set_index('rlnHelicalTubeID')['rlnAngleRot'].to_dict()
    
    # Print the mappings to ensure they are correct
    #print("Psi mapping:", psi_mapping)
    #print("Rot mapping:", rot_mapping)

    # Propagate the values to the entire original dataset
    #original_data['rlnAnglePsi'] = original_data['rlnHelicalTubeID'].map(psi_mapping)
    original_data['rlnAngleRot'] = original_data['rlnHelicalTubeID'].map(rot_mapping)
    
    #print(original_data[['rlnHelicalTubeID', 'rlnAnglePsi', 'rlnAngleRot']].head())
    return original_data


# main
def main():

    parser = argparse.ArgumentParser(description="Process input file and generate STAR files.")
    
    # Add arguments with default values
    parser.add_argument("--input", type=str, required=True, help="Path to the input file (e.g., input.txt)")
    parser.add_argument("--spacing", type=float, default=82.0, help="Repeating unit spacing (default: 82.0)")
    parser.add_argument("--angpix", type=float, default=8.48, help="Pixel size in Angstroms (default: 8.48)")
    parser.add_argument("--tomo_angpix", type=float, default=2.12, help="Pixel size of unbinned tomogram in Angstroms (default: 2.12)")
    parser.add_argument("--output", type=str, default="out.star", help="Output STAR file for interpolation (default: output_interpolation_with_angles.star)")

    # Parse arguments
    args = parser.parse_args()

    input_file=args.input
    spacing=args.spacing
    angpix=args.angpix
    output=args.output
    tomo_angpix = args.tomo_angpix
    
    # Convert IMOD to txt file
    input_txt = input_file.replace(".mod", ".txt")
    run_model2point(input_file, input_txt)

    # interpolation and acquire data
    objects = process_input_file(input_txt, spacing, angpix, tomo_angpix)  
    
    #print(objects)
        
    # create starfile with interpolation and angle
    create_starfile(objects, output.replace(".star", "_init.star"))
     
    new_objects = [] 
    # find cross section
    for i, obj_data in enumerate(objects):
        cross_section = process_cross_section(obj_data)
        rotated_cross_section = rotate_cross_section(cross_section)

        # obtain rot 
        updated_cross_section = calculate_rot_angles(rotated_cross_section)
        #print(rot_angles)

        #final_rotated_cross_section = merge_rot_with_original(cross_section, rot_angles)
        # create starfile for final 
        #new_objects.append(propagate_rot_to_entire_cilia(final_rotated_cross_section, obj_data))
        new_objects.append(propagate_rot_to_entire_cilia(updated_cross_section, obj_data))
        
    create_starfile(new_objects, output)
            

if __name__ == "__main__":
    main()
