"""
IO package for ReLAX
Written by Molly Yu & Huy Bui, McGill 
"""

import subprocess
import os
from typing import List, Dict, Union, Tuple, Optional

 
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
    
def scale_imod_model(input_mod: str, output_mod: str, scale_factor: float) -> bool:
    """
    Reads an IMOD model, scales the X, Y, Z coordinates by a scaling factor,
    and outputs a new IMOD model with the scaled coordinates.
    
    Args:
        input_mod: Path to the input .mod file
        output_mod: Path to the output .mod file
        scale_factor: Factor to scale the X, Y, Z coordinates by
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Create temporary file names
    temp_dir = os.path.dirname(output_mod) or "."
    temp_txt = os.path.join(temp_dir, "temp_points.txt")
    temp_scaled_txt = os.path.join(temp_dir, "temp_scaled_points.txt")
    
    try:
        # Step 1: Convert model to point file
        if not run_model2point(input_mod, temp_txt):
            print(f"Failed to convert {input_mod} to point file")
            return False
        
        # Step 2: Read point file, scale coordinates, and write to new file
        with open(temp_txt, 'r') as f_in, open(temp_scaled_txt, 'w') as f_out:
            for line in f_in:
                parts = line.strip().split()
                if len(parts) >= 4:  # Ensure we have object, contour, and at least X coordinate
                    # Extract components
                    object_id = parts[0]
                    contour_id = parts[1]
                    
                    # Scale the X, Y, Z coordinates and ensure they're integers
                    x = int(round(float(parts[2]) * scale_factor))
                    y = int(round(float(parts[3]) * scale_factor))
                    z = int(round(float(parts[4]) * scale_factor)) if len(parts) > 4 else 0
                    
                    # Write the scaled coordinates to the output file
                    # If there are additional values beyond X, Y, Z, maintain them
                    additional_values = " ".join(parts[5:]) if len(parts) > 5 else ""
                    scaled_line = f"{object_id} {contour_id} {x} {y} {z} {additional_values}".strip()
                    f_out.write(scaled_line + "\n")                    
                else:
                    # If line doesn't match expected format, copy as is
                    f_out.write(line)
        
        # Step 3: Convert scaled point file back to model
        if not run_point2model(temp_scaled_txt, output_mod):
            print(f"Failed to convert scaled points to model {output_mod}")
            return False
            
        print(f"Successfully scaled {input_mod} by factor {scale_factor} to {output_mod}")
        return True
    
    finally:
        # Clean up temporary files
        for temp_file in [temp_txt, temp_scaled_txt]:
            if os.path.exists(temp_file):
                try:
                    #print('Not deleting')
                    os.remove(temp_file)
                except Exception as e:
                    print(f"Warning: Could not remove temporary file {temp_file}: {e}")

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
    