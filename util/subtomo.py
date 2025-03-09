"""
Written by Claude 3.7 Sonnet, Modified by Huy Bui, McGill.
Using rotation convention of Relion
https://relion.readthedocs.io/en/release-3.1/Reference/Conventions.html
Apparently, in the mrcfile reading, z is the first dimension. So z, y, x. 
"""

import mrcfile
import starfile
import numpy as np
from scipy.ndimage import rotate, affine_transform, ndimage
from scipy.fft import fftn, ifftn, fftshift, ifftshift


def extract_subtomograms(tomogram_path, centers, size_d):
    """
    Extract multiple subtomograms of size (size_x × size_y × size_z) centered at the provided coordinates
    in the mrcfile reading, z is the first dimension. So z, y, x. 
    Therefore in reading we have to switch things.
    
    Parameters:
    -----------
    tomogram_path : str
        Path to the MRC file containing the tomogram
    centers : list of tuples
        List of (x, y, z) center coordinates for each subtomogram to extract
    size_d : list of int
        Size of the subtomogram as [size_x, size_y, size_z]
        
    Returns:
    --------
    subtomograms : list of numpy.ndarray
        List of extracted subtomograms
    """
    # Open the MRC file
    if all(x % 2 == 0 for x in size_d):
        print("Check: Subtomo dimensions are even.")
    else:
        print("ERROR: subtomogram size must be even.")
        return
    
    with mrcfile.open(tomogram_path) as mrc:
        # Get the tomogram data
        tomogram = mrc.data
        dim_z, dim_y, dim_x = tomogram.shape
        print(f"Tomogram dimension: {dim_x}, {dim_y}, {dim_z}")
        
        subtomograms = []
        
        for center in centers:
            center_x, center_y, center_z = center
            size_x, size_y, size_z = size_d
            
            # Calculate half sizes (rounded down)
            half_x = size_x // 2
            half_y = size_y // 2
            half_z = size_z // 2
            
            # Calculate the corner coordinates
            x_start = max(0, center_x - half_x)
            y_start = max(0, center_y - half_y)
            z_start = max(0, center_z - half_z)
            
            x_end = min(tomogram.shape[2], center_x + half_x + (size_x % 2))  # Add 1 if size is odd
            y_end = min(tomogram.shape[1], center_y + half_y + (size_y % 2))
            z_end = min(tomogram.shape[0], center_z + half_z + (size_z % 2))
            
            # Extract the subtomogram
            subtomo = tomogram[z_start:z_end, y_start:y_end, x_start:x_end]
            
            # Handle edge case where subtomogram is smaller than requested size
            # (if center is near the edge of the tomogram)
            if subtomo.shape != (size_z, size_y, size_x):
                # Create a volume of zeros with the requested size
                padded_subtomo = np.zeros((size_z, size_y, size_x), dtype=subtomo.dtype)
                
                # Calculate where to place the extracted subtomogram
                pad_x_start = half_x - (center_x - x_start)
                pad_y_start = half_y - (center_y - y_start)
                pad_z_start = half_z - (center_z - z_start)
                
                # Place the extracted subtomogram in the zero volume
                padded_subtomo[
                    pad_z_start:pad_z_start + subtomo.shape[0],
                    pad_y_start:pad_y_start + subtomo.shape[1],
                    pad_x_start:pad_x_start + subtomo.shape[2]
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
        center_z = subtomo.shape[0] // 2
        
        # Calculate start and end z indices, ensuring they're within bounds
        z_start = max(0, center_z - half_z)
        z_end = min(subtomo.shape[0], center_z + half_z)
        
        # Extract the relevant Z sections
        z_sections = subtomo[z_start:z_end, :, :]
        
        # Average along Z axis
        avg_slice = np.mean(z_sections, axis=0)
        
        averaged_slices.append(np.squeeze(avg_slice))
    
    return averaged_slices

    
def rotate_subtomogram_zyz(subtomo, alpha, beta, gamma):
    """
    Rotate a subtomogram using Euler angles in ZYZ convention
    for data in MRC convention (Z, Y, X)
    
    Parameters:
    -----------
    subtomo : numpy.ndarray
        3D subtomogram array in (Z, Y, X) order
    alpha : float
        First rotation angle around Z axis (in degrees)
    beta : float
        Second rotation angle around Y axis (in degrees)
    gamma : float
        Third rotation angle around Z axis (in degrees)
        
    Returns:
    --------
    rotated_subtomo : numpy.ndarray
        Rotated subtomogram
    """
    # Make a copy to avoid modifying the original, need float32 as float16 will error
    rotated = subtomo.copy().astype(np.float32)
        
    # First rotation around Z axis (alpha)
    rotated = rotate(rotated, angle=alpha, axes=(1, 2), reshape=False, mode='constant')
    
    # Second rotation around Y axis (beta). Modified due to right-hand rule for rotations in scipy.ndimage
    rotated = rotate(rotated, angle=-beta, axes=(0, 2), reshape=False, mode='constant')
    
    # Third rotation around Z axis (gamma)
    rotated = rotate(rotated, angle=gamma, axes=(1, 2), reshape=False, mode='constant')
    
    return rotated
    
def rotate_subtomogram_zyz_single_interpolation(subtomo, alpha, beta, gamma):
    """
    Rotate a subtomogram using Euler angles in ZYZ convention
    using a single interpolation step via a combined rotation matrix.
    Specifically handles MRC convention where Z is axis 0.
    
    Parameters:
    -----------
    subtomo : numpy.ndarray
        3D subtomogram array in MRC convention (Z, Y, X) order
    alpha : float
        First rotation angle around Z axis (in degrees)
    beta : float
        Second rotation angle around Y axis (in degrees)
    gamma : float
        Third rotation angle around Z axis (in degrees)
        
    Returns:
    --------
    rotated_subtomo : numpy.ndarray
        Rotated subtomogram
    """
    # Convert to float32 to avoid data type issues with affine_transform
    # This handles float16 input arrays that scipy can't process directly
    subtomo_float32 = subtomo.astype(np.float32)

    # Convert angles to radians
    alpha_rad = np.deg2rad(alpha)
    beta_rad = np.deg2rad(beta)
    gamma_rad = np.deg2rad(gamma)
    
    # Create rotation matrices for each axis
    # Remember that for MRC, the axes are: Z=0, Y=1, X=2
    # But scipy's affine_transform works in reversed order (X,Y,Z)
    
    # Build the rotation matrices for the ZYZ convention
    # First rotation around Z axis (alpha)
    Rz_alpha = np.array([
        [np.cos(alpha_rad), -np.sin(alpha_rad), 0],
        [np.sin(alpha_rad), np.cos(alpha_rad), 0],
        [0, 0, 1]
    ])
    
    # Second rotation around Y axis (beta)
    Ry_beta = np.array([
        [np.cos(beta_rad), 0, np.sin(beta_rad)],
        [0, 1, 0],
        [-np.sin(beta_rad), 0, np.cos(beta_rad)]
    ])
    
    # Third rotation around Z axis (gamma)
    Rz_gamma = np.array([
        [np.cos(gamma_rad), -np.sin(gamma_rad), 0],
        [np.sin(gamma_rad), np.cos(gamma_rad), 0],
        [0, 0, 1]
    ])
    
    # Calculate the combined rotation matrix (R = Rz(alpha) * Ry(beta) * Rz(gamma))
    R = np.dot(Rz_alpha, np.dot(Ry_beta, Rz_gamma))
    
    # For MRC convention (Z,Y,X), we need to swap axes for affine_transform
    # which expects (X,Y,Z), so we reverse the rotation matrix
    R_mrc = R[::-1, ::-1]
    
    # Get the center of the volume
    center = np.array(subtomo.shape) // 2
    
    # For MRC convention, we reverse the center coordinates
    center_mrc = center[::-1]
    
    # Calculate the offset for the affine transform
    offset = center_mrc - np.dot(R_mrc, center_mrc)
    
    # Apply the rotation with a single interpolation
    rotated = affine_transform(
        subtomo_float32,
        matrix=R_mrc,
        offset=offset,
        order=3,  # cubic interpolation
        mode='constant',
        cval=0.0
    )
    
    return rotated


def rotate_subtomograms_zyz(subtomograms, euler_angles):
    """
    Rotate multiple subtomograms using Euler angles in ZYZ convention
    
    Parameters:
    -----------
    subtomograms : list of numpy.ndarray
        List of 3D subtomogram arrays
    euler_angles : list of tuples
        List of (alpha, beta, gamma) angles in degrees for each subtomogram
        
    Returns:
    --------
    rotated_subtomograms : list of numpy.ndarray
        List of rotated subtomograms
    """
    if len(subtomograms) != len(euler_angles):
        raise ValueError("Number of subtomograms must match number of Euler angle sets")
    
    rotated_subtomograms = []
    
    for subtomo, angles in zip(subtomograms, euler_angles):
        alpha, beta, gamma = angles
        rotated = rotate_subtomogram_zyz_single_interpolation(subtomo, alpha, beta, gamma)
        rotated_subtomograms.append(rotated)
    
    return rotated_subtomograms
    

def generate_subtomogram_cross_section(subtomogram, eulers, z_slices):
    """
    Generate a cross-section of a subtomogram by rotating according to Euler angles
    and averaging specified Z slices. Subtomogram must be a cube.
    
    First determine the center through cross section
    Then extract a subtomogram with dim_z^3, rotate psi & tilt, then generate the average
    
    Parameters:
    -----------
    subtomogram : numpy.ndarray
        3D tomogram array in (Z, Y, X) format in cube
    eulers : tuple or list
        Euler angles (alpha, beta, gamma) in degrees for ZYZ convention
    z_slices : list or numpy.ndarray
        Indices of Z slices to average after rotation
        
    Returns:
    --------
    cross_section_2D : numpy.ndarray
        2D cross-section image
    """
    # Extract Euler angles
    alpha, beta, gamma = eulers
    
    # Rotate the tomogram
    rotated_subtomogram = rotate_subtomogram_zyz_single_interpolation(subtomogram, alpha, beta, gamma)
    
    with mrcfile.new('tmp.mrc', overwrite=True) as mrc:
        mrc.set_data(rotated_subtomogram)
    
    # Average the specified Z slices
    cross_section_2D = average_z_sections([rotated_subtomogram], z_slices)
    
    return cross_section_2D
    
def generate_tomogram_cross_section(tomogram_file, cross_section, z_slices):
    """
    Generate a cross-section of tomogram by rotating according to euler angles in the cross section
    star file format (defined in util.geom) and then averaging specified Z slices. 
        
    First determine the center through cross section.
    Then extract a subtomogram with dim_z^3, rotate psi & tilt, then generate the average
    
    Parameters:
    -----------
    tomogram_file : tomogram in mrc format
    cross_section : dataframe
        Starfile format
    z_slices : list or numpy.ndarray
        Indices of Z slices to average after rotation
        
    Returns:
    --------
    cross_section_2D : numpy.ndarray
        2D cross-section image
    """
    with mrcfile.open(tomogram_file) as mrc:
        # Get the tomogram data
        tomogram = mrc.data
        dim_z = len(tomogram)
        
    center_cs = avg_values = [
        int(cross_section['rlnCoordinateX'].mean()),
        int(cross_section['rlnCoordinateY'].mean()),
        int(cross_section['rlnCoordinateZ'].mean())
    ]
    #dim_z = len(tomogram)
    phi = cross_section['rlnAnglePsi'].median() 
    tilt = cross_section['rlnAngleTilt'].median()
   
    center_subtomogram = extract_subtomograms(tomogram_file, [center_cs], [dim_z, dim_z, dim_z])
    #with mrcfile.new('tmp.mrc', overwrite=True) as mrc:
    #    mrc.set_data(center_subtomogram[0])
        
    cross_section_2D = generate_subtomogram_cross_section(center_subtomogram[0], [-phi, -tilt, 0], z_slices)
    return cross_section_2D

def low_pass_3D(volume, resolution_in_Angstrom, pixel_size_in_Angstrom, filter_edge_width=0.1):
    """
    Apply a low-pass filter to a 3D tomogram with a smooth edge to avoid artifacts
    
    Parameters:
    -----------
    volume : numpy.ndarray
        3D tomogram array
    resolution_in_Angstrom : float
        Target resolution in Angstrom
    pixel_size_in_Angstrom : float
        Pixel size in Angstrom
    filter_edge_width : float, optional
        Width of the Gaussian edge as a fraction of the cutoff frequency (default: 0.1)
        Higher values create a smoother transition
        
    Returns:
    --------
    filtered_volume : numpy.ndarray
        Filtered tomogram
    """
    # Get volume dimensions
    sz, sy, sx = volume.shape
    
    # Create frequency grid
    kz = np.fft.fftfreq(sz)
    ky = np.fft.fftfreq(sy)
    kx = np.fft.fftfreq(sx)
    
    # Create meshgrid
    kz_grid, ky_grid, kx_grid = np.meshgrid(kz, ky, kx, indexing='ij')
    
    # Calculate distance from origin in frequency space
    k_distance = np.sqrt(kz_grid**2 + ky_grid**2 + kx_grid**2)
    
    # Convert resolution to frequency cutoff
    # freq_cutoff = 1.0 / resolution_in_Angstrom
    # Normalize by pixel size and Nyquist frequency
    nyquist_freq = 0.5 / pixel_size_in_Angstrom
    freq_cutoff = pixel_size_in_Angstrom / resolution_in_Angstrom * 0.5
    
    # Create a smooth filter with Gaussian falloff
    sigma = filter_edge_width * freq_cutoff
    filter_mask = np.exp(-0.5 * ((k_distance - freq_cutoff) / sigma)**2)
    filter_mask[k_distance < freq_cutoff] = 1.0
    
    # Apply FFT
    volume_fft = fftn(volume)
    
    # Apply filter
    filtered_fft = volume_fft * filter_mask
    
    # Inverse FFT
    filtered_volume = np.real(ifftn(filtered_fft))
    
    return filtered_volume

def low_pass_2D(slice_2d, resolution_in_Angstrom, pixel_size_in_Angstrom, filter_edge_width=0.1):
    """
    Apply a low-pass filter to a 2D slice with a smooth edge to avoid artifacts
    
    Parameters:
    -----------
    slice_2d : numpy.ndarray
        2D image array
    resolution_in_Angstrom : float
        Target resolution in Angstrom
    pixel_size_in_Angstrom : float
        Pixel size in Angstrom
    filter_edge_width : float, optional
        Width of the Gaussian edge as a fraction of the cutoff frequency (default: 0.1)
        Higher values create a smoother transition
        
    Returns:
    --------
    filtered_slice : numpy.ndarray
        Filtered 2D slice
    """
    # Get slice dimensions
    sy, sx = slice_2d.shape
    
    # Create frequency grid
    ky = np.fft.fftfreq(sy)
    kx = np.fft.fftfreq(sx)
    
    # Create meshgrid
    ky_grid, kx_grid = np.meshgrid(ky, kx, indexing='ij')
    
    # Calculate distance from origin in frequency space
    k_distance = np.sqrt(ky_grid**2 + kx_grid**2)
    
    # Convert resolution to frequency cutoff
    # Normalize by pixel size and Nyquist frequency
    nyquist_freq = 0.5 / pixel_size_in_Angstrom
    freq_cutoff = pixel_size_in_Angstrom / resolution_in_Angstrom * 0.5
    
    # Create a smooth filter with Gaussian falloff
    sigma = filter_edge_width * freq_cutoff
    filter_mask = np.exp(-0.5 * ((k_distance - freq_cutoff) / sigma)**2)
    filter_mask[k_distance < freq_cutoff] = 1.0
    
    # Apply FFT
    slice_fft = fftn(slice_2d)
    
    # Apply filter
    filtered_fft = slice_fft * filter_mask
    
    # Inverse FFT
    filtered_slice = np.real(ifftn(filtered_fft))
    
    return filtered_slice

def filter_mrc_file(mrc_path, output_path, resolution_in_Angstrom, pixel_size_in_Angstrom, filter_edge_width=0.1):
    """
    Apply a low-pass filter to an entire MRC file
    
    Parameters:
    -----------
    mrc_path : str
        Path to input MRC file
    output_path : str
        Path to save filtered MRC file
    resolution_in_Angstrom : float
        Target resolution in Angstrom
    pixel_size_in_Angstrom : float
        Pixel size in Angstrom
    filter_edge_width : float, optional
        Width of the Gaussian edge as a fraction of the cutoff frequency (default: 0.1)
        
    Returns:
    --------
    None
    """
    with mrcfile.open(mrc_path) as mrc:
        data = mrc.data   
        # Check if it's a 3D volume or a 2D slice
        if len(data.shape) == 3:
            filtered_data = low_pass_3D(data, resolution_in_Angstrom, pixel_size_in_Angstrom, filter_edge_width)
        else:
            filtered_data = low_pass_2D(data, resolution_in_Angstrom, pixel_size_in_Angstrom, filter_edge_width)
        # Create a new MRC file with the filtered data
        with mrcfile.new(output_path, overwrite=True) as new_mrc:
            new_mrc.set_data(filtered_data.astype(np.float32))
            # Copy header information
            if hasattr(mrc, 'header'):
                header_dict = {k: getattr(mrc.header, k) for k in mrc.header._array_fields}
                for key, value in header_dict.items():
                    try:
                        setattr(new_mrc.header, key, value)
                    except AttributeError:
                        pass

def write_subtomograms_to_mrc(subtomograms, file_prefix='subtomogram_', dtype='float32'):
    """
    Save a list of subtomograms as .mrc files with a specified file name prefix and data type.

    Args:
        subtomograms (list of numpy arrays): List of subtomograms (3D arrays) to save.
        file_prefix (str): Prefix for the output .mrc filenames. Default is 'subtomogram_'.
        dtype (str): Data type for the output .mrc files. Supported values: 'float16', 'float32', 'int16'.
                     Default is 'float32'.
    """
    # Map the dtype string to the corresponding NumPy dtype
    dtype_map = {
        'float16': np.float16,
        'float32': np.float32,
        'int16': np.int16,
    }

    # Validate the dtype
    if dtype not in dtype_map:
        raise ValueError(f"Unsupported dtype: {dtype}. Supported values are 'float16', 'float32', 'int16'.")

    # Get the NumPy dtype
    np_dtype = dtype_map[dtype]

    for i, subtomo in enumerate(subtomograms):
        # Construct the output filename
        output_filename = f"{file_prefix}{i}.mrc"
        # Convert the subtomogram to the specified dtype
        subtomo_converted = subtomo.astype(np_dtype)     
        # Save the subtomogram as an .mrc file
        with mrcfile.new(output_filename, overwrite=True) as mrc:
            mrc.set_data(subtomo_converted)
            print(f"Subtomogram {i} saved to '{output_filename}' with dtype {dtype}")

def get_median_row(group):
    median_value = group['rlnTomoParticleId'].median()
    closest_row = group.iloc[(group['rlnTomoParticleId'] - median_value).abs().argmin()]
    return closest_row
 
def generate_2d_stack(tomogram_file, df_star, box_size, z_slices_to_avg):
    """
    Generate the 2d stack of all the doublets,     df_star is for 1 cilia only
    Args:
        tomogram_file: tomogram mrc file
        df_star: dataframe for star (1 cilia only)
        box_size: box size
        z_slices_to_avg
    Returns:
        Output stack
    """
    # Need to sort by filamentID
    result_df = (
        df_star.groupby('rlnHelicalTubeID', group_keys=False)
        .apply(get_median_row)
        .reset_index(drop=True)  # Reset the index to avoid ambiguity
        .sort_values('rlnHelicalTubeID', ascending=True)
    )
    #result_df = df_star.groupby('rlnHelicalTubeID', group_keys=False).apply(get_median_row).sort_values('rlnHelicalTubeID', ascending=True)
    centers = result_df[['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']].astype(int).values.tolist()

    print("Extracting subtomograms")
    subtomograms = extract_subtomograms(tomogram_file, centers, [box_size, box_size, box_size])
    #write_subtomograms_to_mrc(subtomograms, 'subtomo_')
        
    eulers = np.array(result_df[['rlnAngleRot', 'rlnAngleTilt', 'rlnAnglePsi']])
    #print(eulers)
    # Reverse
    euler_angles = (-eulers[:, [2, 1, 0]]).tolist()
    rotated_subtomograms = rotate_subtomograms_zyz(subtomograms, euler_angles)
    #write_subtomograms_to_mrc(rotated_subtomograms, 'rotated_subtomo_')

    # Average Z sections for each rotated subtomogram
    averaged_slices = average_z_sections(rotated_subtomograms, z_slices_to_avg)
    print(f"Created {len(averaged_slices)} averaged slices of shape {averaged_slices[0].shape}")
    #write_subtomograms_to_mrc(averaged_slices, 'rotated_avg_slice_')
    
    stack = np.stack(averaged_slices, axis=0)       
    return stack
    
def write_cross_section_to_mrc(tomogram_file, star_file, output_file, angpix, tomo_z_slices_to_avg=20, lowpass=40):
    objects = read_starfile_into_cilia_object(star_file)

    for i, obj_data in enumerate(objects):
        cross_section = process_cross_section(obj_data)
        cross_section_2D = generate_tomogram_cross_section(tomogram_file, cross_section, tomo_z_slices_to_avg)
        with mrcfile.new(f"{output_file}", overwrite=True) as mrc:
            mrc.set_data(low_pass_2D(cross_section_2D[0], lowpass, angpix).astype(np.float32))
            print(f"Cross section saved to {output_file}")
        continue
        
# Example usage
if __name__ == "__main__":
    tomogram_file = "/Users/kbui2/Desktop/Sessions/CU428lowmag_11_14.00Apx_refined.mrc"
    
    # Define multiple centers for extraction
    centers = [
        (150, 200, 75),
        (200, 250, 80),
        (100, 150, 60)
    ]
    #print(centers)
    # Size of each cubic subtomogram
    box_size = 64
    
    # Extract multiple subtomograms
    subtomograms = extract_subtomograms(tomogram_file, centers, [box_size, box_size, box_size])
    print(f"Extracted {len(subtomograms)} subtomograms of shape {subtomograms[0].shape}")
    
    # Define Euler angles for rotation (ZYZ convention) in degrees
    euler_angles = [
        (30, 45, 15),  # For the first subtomogram
        (0, 0, 0),    # For the second subtomogram
        (10, 20, 30)   # For the third subtomogram
    ]
    
    # Rotate the subtomograms
    rotated_subtomograms = rotate_subtomograms_zyz(subtomograms, euler_angles)
    print(f"Rotated {len(rotated_subtomograms)} subtomograms using ZYZ Euler angles")
    
    # Specify number of Z slices to average (must be even)
    z_slices_to_avg = 10
    
    # Average Z sections for each rotated subtomogram
    averaged_slices = average_z_sections(rotated_subtomograms, z_slices_to_avg)
    print(f"Created {len(averaged_slices)} averaged slices of shape {averaged_slices[0].shape}")
    
    # Example: Save the rotated subtomograms
    write_subtomograms_to_mrc(rotated_subtomograms, 'rotated_subtomo_')
    
    # Example: Save the averaged slices from rotated subtomograms
    write_subtomograms_to_mrc(averaged_slices, 'rotated_avg_slice_')