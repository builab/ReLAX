import mrcfile
import numpy as np

def extract_subtomograms(tomogram_path, centers, size_d):
    """
    Extract multiple cubic subtomograms of size D centered at the provided coordinates
    
    Parameters:
    -----------
    tomogram_path : str
        Path to the MRC file containing the tomogram
    centers : list of tuples
        List of (x, y, z) center coordinates for each subtomogram to extract
    size_d : int
        Size of the cubic subtomogram (D×D×D)
        
    Returns:
    --------
    subtomograms : list of numpy.ndarray
        List of extracted subtomograms
    """
    # Open the MRC file
    with mrcfile.open(tomogram_path) as mrc:
        # Get the tomogram data
        tomogram = mrc.data
        
        subtomograms = []
        
        for center in centers:
            center_x, center_y, center_z = center
            
            # Calculate half size (rounded down)
            half_d = size_d // 2
            
            # Calculate the corner coordinates
            x_start = max(0, center_x - half_d)
            y_start = max(0, center_y - half_d)
            z_start = max(0, center_z - half_d)
            
            x_end = min(tomogram.shape[0], center_x + half_d + (size_d % 2))  # Add 1 if size_d is odd
            y_end = min(tomogram.shape[1], center_y + half_d + (size_d % 2))
            z_end = min(tomogram.shape[2], center_z + half_d + (size_d % 2))
            
            # Extract the subtomogram
            subtomo = tomogram[x_start:x_end, y_start:y_end, z_start:z_end]
            
            # Handle edge case where subtomogram is smaller than requested size
            # (if center is near the edge of the tomogram)
            if subtomo.shape != (size_d, size_d, size_d):
                # Create a volume of zeros with the requested size
                padded_subtomo = np.zeros((size_d, size_d, size_d), dtype=subtomo.dtype)
                
                # Calculate where to place the extracted subtomogram
                pad_x_start = half_d - (center_x - x_start)
                pad_y_start = half_d - (center_y - y_start)
                pad_z_start = half_d - (center_z - z_start)
                
                # Place the extracted subtomogram in the zero volume
                padded_subtomo[
                    pad_x_start:pad_x_start + subtomo.shape[0],
                    pad_y_start:pad_y_start + subtomo.shape[1],
                    pad_z_start:pad_z_start + subtomo.shape[2]
                ] = subtomo
                
                subtomo = padded_subtomo
                
            subtomograms.append(subtomo)
            
        return subtomograms

def average_z_sections(subtomograms, z_slices):
    """
    Average z_slices around the center of each subtomogram
    
    Parameters:
    -----------
    subtomograms : list of numpy.ndarray
        List of 3D subtomogram arrays
    z_slices : int
        Number of Z slices to average (must be even). 
        Will take z_slices/2 above and below the center.
        
    Returns:
    --------
    averaged_slices : list of numpy.ndarray
        List of 2D arrays, each the Z-average of a subtomogram
    """
    if z_slices % 2 != 0:
        raise ValueError("z_slices must be an even number")
    
    half_z = z_slices // 2
    averaged_slices = []
    
    for subtomo in subtomograms:
        # Find the center Z index
        center_z = subtomo.shape[2] // 2
        
        # Calculate start and end z indices, ensuring they're within bounds
        z_start = max(0, center_z - half_z)
        z_end = min(subtomo.shape[2], center_z + half_z)
        
        # Extract the relevant Z sections
        z_sections = subtomo[:, :, z_start:z_end]
        
        # Average along Z axis
        avg_slice = np.mean(z_sections, axis=2)
        
        averaged_slices.append(avg_slice)
    
    return averaged_slices

# Example usage
if __name__ == "__main__":
    tomogram_file = "path/to/your/tomogram.mrc"
    
    # Define multiple centers for extraction
    centers = [
        (150, 200, 75),
        (200, 250, 80),
        (100, 150, 60)
    ]
    
    # Size of each cubic subtomogram
    size = 64
    
    # Extract multiple subtomograms
    subtomograms = extract_subtomograms(tomogram_file, centers, size)
    print(f"Extracted {len(subtomograms)} subtomograms of shape {subtomograms[0].shape}")
    
    # Specify number of Z slices to average (must be even)
    z_slices_to_avg = 10
    
    # Average Z sections for each subtomogram
    averaged_slices = average_z_sections(subtomograms, z_slices_to_avg)
    print(f"Created {len(averaged_slices)} averaged slices of shape {averaged_slices[0].shape}")
    
    # Example: Save the first averaged slice
    with mrcfile.new("averaged_slice_0.mrc", overwrite=True) as mrc:
        mrc.set_data(averaged_slices[0].astype(np.float32))
        print("First averaged slice saved to 'averaged_slice_0.mrc'")
    
    # Example: Save all subtomograms
    for i, subtomo in enumerate(subtomograms):
        with mrcfile.new(f"subtomogram_{i}.mrc", overwrite=True) as mrc:
            mrc.set_data(subtomo.astype(np.float32))
            print(f"Subtomogram {i} saved to 'subtomogram_{i}.mrc'")
    
    # Example: Save all averaged slices
    for i, avg_slice in enumerate(averaged_slices):
        with mrcfile.new(f"averaged_slice_{i}.mrc", overwrite=True) as mrc:
            mrc.set_data(avg_slice.astype(np.float32))
            print(f"Averaged slice {i} saved to 'averaged_slice_{i}.mrc'")