"""
Geom package for ReLAX
Written by Molly Yu & Huy Bui, McGill 
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import eig, inv, lstsq
from scipy.optimize import leastsq, minimize
from scipy.interpolate import splprep, splev
import itertools
import warnings


def normalize_angle(angle):
    """
    Normalize angle to range -180 to 180 in Relion
    """
    return (angle + 180) % 360 - 180
    
def robust_interpolate_spline(points, angpix, spacing):
    """
    Interpolate points along a line with a specified spacing using splines.
    Includes robust handling of edge cases like too few points, collinear points, etc.
    
    Parameters:
    -----------
    points : array-like
        List of (x, y, z) coordinate points to fit a spline through
    angpix : float
        Angstroms per pixel conversion factor
    spacing : float
        Desired spacing between interpolated points in Angstroms
        
    Returns:
    --------
    interpolated_pts : numpy.ndarray
        Array of interpolated points at the specified spacing
    cum_distances_angst : numpy.ndarray
        Cumulative distances along the path in Angstroms
    """
    points = np.array(points)
    
    # Check if points are valid
    if points.size == 0 or points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Input must be a non-empty list of 3D points with shape (n, 3)")
    
    # Handle the case with very few points
    n_points = points.shape[0]
    if n_points == 1:
        # Only one point - can't create a spline, return the point and zero distance
        return points, np.array([0.0])
    
    # Check for non-finite values
    if not np.all(np.isfinite(points)):
        raise ValueError("Input contains NaN or Inf values")
    
    # Check for duplicate points and remove them if found
    _, unique_indices = np.unique(points, axis=0, return_index=True)
    unique_indices = np.sort(unique_indices)  # Keep original order
    if len(unique_indices) < n_points:
        points = points[unique_indices]
        n_points = len(points)
        if n_points == 1:
            # After removing duplicates, only one point remains
            return points, np.array([0.0])
    
    # Determine appropriate spline order based on number of points
    if n_points < 4:
        k = 1  # Linear spline for 2-3 points
    else:
        k = 3  # Cubic spline when enough points are available
    
    # Check for collinearity if we have enough points
    if n_points >= 3 and k > 1:
        # Calculate vectors between consecutive points
        vectors = np.diff(points, axis=0)
        # Normalize vectors
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        valid_vectors = norms > 1e-10
        
        if np.all(valid_vectors):
            unit_vectors = vectors / np.where(norms > 1e-10, norms, 1.0)
            # Check if all unit vectors are nearly parallel
            dot_products = np.abs(np.sum(unit_vectors[:-1] * unit_vectors[1:], axis=1))
            if np.all(dot_products > 0.999):
                k = 1  # Use linear spline for collinear points
    
    # Helper function for cumulative distance
    def cumulative_distance(pts):
        """Calculate cumulative distance along a path of points"""
        diffs = np.diff(pts, axis=0)
        distances = np.sqrt(np.sum(diffs**2, axis=1))
        cum_dist = np.zeros(len(pts))
        cum_dist[1:] = np.cumsum(distances)
        return cum_dist
    
    # Try to fit the spline with multiple fallback options
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tck, _ = splprep(points.T, s=0, k=k)
    except Exception as e1:
        try:
            # First fallback: try with some smoothing
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tck, _ = splprep(points.T, s=1.0, k=k)
        except Exception as e2:
            try:
                # Second fallback: try with linear interpolation
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    tck, _ = splprep(points.T, s=0, k=1)
            except Exception as e3:
                # Last resort: use direct linear interpolation without splines
                # Create a parametric representation based on cumulative distance
                cum_dist = cumulative_distance(points)
                total_dist = cum_dist[-1]
                if total_dist == 0:
                    return points, np.array([0.0])
                
                # Normalize distances to [0, 1]
                normalized_dist = cum_dist / total_dist
                
                # Create dense samples
                dense_distances = np.linspace(0, 1, max(int(n_points * 100), 1000))
                
                # Interpolate each coordinate
                dense_points = np.zeros((len(dense_distances), 3))
                for i in range(3):
                    dense_points[:, i] = np.interp(dense_distances, normalized_dist, points[:, i])
                
                # Calculate distances and resample
                dense_cum_dist = cumulative_distance(dense_points * angpix)
                resampled_distances = np.arange(0, dense_cum_dist[-1], spacing)
                
                # Interpolate at equally spaced intervals
                interpolated_pts = np.zeros((len(resampled_distances), 3))
                for i in range(3):
                    interpolated_pts[:, i] = np.interp(resampled_distances, dense_cum_dist, dense_points[:, i])
                
                cum_distances_angst = np.arange(0, dense_cum_dist[-1], spacing)
                if len(cum_distances_angst) != len(interpolated_pts):
                    cum_distances_angst = np.linspace(0, dense_cum_dist[-1], len(interpolated_pts))
                
                return interpolated_pts, cum_distances_angst
    
    # Calculate dense samples along the spline
    num_dense_samples = max(int(n_points * 100), 1000)  # At least 1000 samples
    distances = np.linspace(0, 1, num_dense_samples)
    
    # Evaluate the spline at dense samples
    dense_points = np.array(splev(distances, tck)).T
    
    # Calculate cumulative distances
    cum_distances_all = cumulative_distance(dense_points * angpix)
    
    # Check if the total distance is zero or very small
    if cum_distances_all[-1] < spacing * 0.1:
        # Path is too short, return the original points
        return points, cumulative_distance(points * angpix)
    
    # Resample at equal intervals
    resampled_distances = np.arange(0, cum_distances_all[-1], spacing)
    
    # Interpolate at equally spaced intervals
    interpolated_pts = np.zeros((len(resampled_distances), 3))
    for i in range(3):
        interpolated_pts[:, i] = np.interp(resampled_distances, cum_distances_all, dense_points[:, i])
    
    # Calculate final cumulative distances
    cum_distances_angst = np.arange(0, cum_distances_all[-1], spacing)
    
    # Ensure the length matches (in case of numerical issues)
    if len(cum_distances_angst) != len(interpolated_pts):
        cum_distances_angst = np.linspace(0, cum_distances_all[-1], len(interpolated_pts))
    
    return interpolated_pts, cum_distances_angst
    
    
def interpolate_spline(points, angpix, spacing):
    """
    Interpolate points along a line with a specified spacing using splines.
    TODO: Need to deal with error if this is too short? Min 5 particles?
    """
    points = np.array(points)
    tck, _ = splprep(points.T, s=0)  # Create spline representation
    distances = np.linspace(0, 1, int(np.ceil(points.shape[0] * 100)))  # Dense samples. 100 times the points?
    
    # Evaluate the spline at dense samples
    dense_points = np.array(splev(distances, tck)).T
    
    cum_distances_all = cumulative_distance(dense_points*angpix)
    
    # Resample at equal intervals
    resampled_distances = np.arange(0, cum_distances_all[-1], spacing)
    interpolated_pts = np.array([np.interp(resampled_distances, cum_distances_all, dense_points[:, i]) for i in range(3)]).T
    cum_distances_angst = cumulative_distance(interpolated_pts*angpix)
    return interpolated_pts, cum_distances_angst
    
def cumulative_distance(points):
    # Compute the Euclidean distance between consecutive points
    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    # Compute the cumulative sum, adding an initial zero for the first point
    return np.concatenate(([0], np.cumsum(distances)))

def calculate_tilt_psi_angles(v):
    """
    Calculate the ZYZ Euler angles (Rot, Tilt, Psi) for a vector v.
    Rot is not calculated in this function yet
    These angles rotate the vector to align with the Z-axis.
    """
    v = v / np.linalg.norm(v)
    rot = 0
    tilt = np.arccos(-v[2])
    psi = np.arctan2(-v[1], v[0])
    psi_degree = normalize_angle(np.degrees(psi))
    
    return np.degrees(rot), np.degrees(tilt), psi_degree

def define_plane(normal_vector, reference_point):
    return normal_vector, reference_point
    
def calculate_perpendicular_distance(point, plane_normal, reference_point):
    return np.abs(np.dot(plane_normal, point - reference_point)) / np.linalg.norm(plane_normal)

def find_cross_section_points(data, plane_normal, reference_point):
    """
    Find the points on each filament closest to the cross-sectional plane,
    """
    cross_section = []
    grouped_data = data.groupby('rlnHelicalTubeID')
    for filament_id, group in grouped_data:
        points = group[['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']].values
        distances = np.array([calculate_perpendicular_distance(point, plane_normal, reference_point) for point in points])
        closest_point = group.iloc[np.argmin(distances)]
        cross_section.append(closest_point)
    return pd.DataFrame(cross_section, columns=data.columns)

def find_shortest_filament(data):
    shortest_length, shortest_midpoint, shortest_filament_id = float('inf'), None, None
    for filament_id, group in data.groupby('rlnHelicalTubeID'):
        filament_points = group[['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']].values
        min_point, max_point = filament_points.min(axis=0), filament_points.max(axis=0)
        length = np.linalg.norm(max_point - min_point)
        if length < shortest_length:
            shortest_length, shortest_midpoint, shortest_filament_id = length, (min_point + max_point) / 2, filament_id
    return shortest_filament_id, shortest_midpoint

def calculate_normal_vector(filament_points):
    vectors = np.diff(filament_points, axis=0)
    #print(vectors)
    normal_vector = np.sum(vectors, axis=0)
    return normal_vector / np.linalg.norm(normal_vector)

def process_cross_section(data):
    """ Even if the cross section doesn't have every filament, it can still project it from the shorter filament """
    shortest_filament_id, shortest_midpoint = find_shortest_filament(data)
    print(f"{shortest_filament_id}, {shortest_midpoint}")
    filament_points = data[data['rlnHelicalTubeID'] == shortest_filament_id][['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']].values
    #print(filament_points)
    normal_vector = calculate_normal_vector(filament_points)
    plane_normal, plane_point = define_plane(normal_vector, shortest_midpoint)
    return find_cross_section_points(data, plane_normal, plane_point)

def rotate_cross_section(cross_section):
    """
    Rotates cross-section by Psi and Tilt angles to transform into Z plane
    """
    rotated_cross_section = cross_section
    psi = 90 - cross_section['rlnAnglePsi'].median()
    #psi = -90 + cross_section['rlnAnglePsi'].median()
    tilt = cross_section['rlnAngleTilt'].median()
    
    for index, row in rotated_cross_section.iterrows():
        x, y, z = row['rlnCoordinateX'], row['rlnCoordinateY'], row['rlnCoordinateZ']
        
        # Rotation around Z-axis by Psi
        psi_rad = np.radians(psi)
        Rz = np.array([
            [np.cos(-psi_rad), -np.sin(-psi_rad), 0],
            [np.sin(-psi_rad),  np.cos(-psi_rad), 0],
            [0,                0,               1]
        ])
        rotated_point = Rz @ np.array([x, y, z])
        
        # Rotation around Y-axis by Tilt
        tilt_rad = np.radians(tilt)
        Ry = np.array([
            [1, 0,              0             ],
            [0, np.cos(-tilt_rad), -np.sin(-tilt_rad)],
            [0, np.sin(-tilt_rad),  np.cos(-tilt_rad)]
        ])
        rotated_point = Ry @ rotated_point
        rotated_cross_section.at[index, 'rlnCoordinateX'] = rotated_point[0]
        rotated_cross_section.at[index, 'rlnCoordinateY'] = rotated_point[1]
        rotated_cross_section.at[index, 'rlnCoordinateZ'] = rotated_point[2]
    
    # Replace rlnAngleTilt, rlnAnglePsi with 0
    rotated_cross_section[['rlnAngleTilt', 'rlnAnglePsi']] = 0
    return rotated_cross_section
    
def calculate_rot_angles(rotated_cross_section, fit_method):
    """ Calculate the rotation angle in a cross section """    
    if fit_method == 'simple':
        updated_cross_section = calculate_rot_angles_simple(rotated_cross_section)
    elif fit_method == 'ellipse':
        # Fitting using ellipse requires at least 5 points
        updated_cross_section = calculate_rot_angles_ellipse(rotated_cross_section)
    else:
        print("Unknown option. Using simple method")
        updated_cross_section = calculate_rot_angle_simple(rotated_cross_section)
    
    return updated_cross_section 
       
def calculate_rot_angles_simple(rotated_cross_section):
    """ 
    Calculate the rotation angle in a cross section
    Tested to work very well with polarity 1
    Seems to be good with polarity 0 as well
    """
    updated_cross_section = rotated_cross_section
    rot_angles = np.zeros(len(rotated_cross_section))
    coords = rotated_cross_section[['rlnCoordinateX', 'rlnCoordinateY']].values
    for i in range(len(rotated_cross_section)):
        delta_x, delta_y = coords[(i + 1) % len(rotated_cross_section)] - coords[(i - 1) % len(rotated_cross_section)]
        rot_angles[i] = np.degrees(np.arctan2(delta_y, delta_x)) - 180
    updated_cross_section['rlnAngleRot'] = np.vectorize(normalize_angle)(rot_angles)
    return updated_cross_section
    
    
def calculate_rot_angles_ellipse(rotated_cross_section):
    """ 
    Calculate the rotation angle in a cross section using ellipse method

    """
    updated_cross_section = rotated_cross_section
    points = rotated_cross_section[['rlnCoordinateX', 'rlnCoordinateY']].to_numpy()
    x = points[:, 0]
    y = points[:, 1]
    # Fit an ellipse to these points
    ellipse_params = fit_ellipse(x, y, axis_handle=None)
    center = [ellipse_params['X0'], ellipse_params['Y0']]
    axes = [ellipse_params['a'], ellipse_params['b']]
    angle = ellipse_params['phi']

    print(f"Fitted center: {center}")
    print(f"Fitted axes: {axes}")
    print(f"Fitted rotation (radians): {angle}")
    elliptical_distortion = ellipse_params['a']/ellipse_params['b']
    print(f"Elliptical distortion: {elliptical_distortion :.2f}")
    
    fitted_ellipse_pts = ellipse_points(center, axes, angle)

    # Order the original points along the ellipse:
    angles = angle_along_ellipse(center, axes, angle, points)
    # Empirical correction by -270 degrees
    angles = np.degrees(angles) - 270
    
    #sort_order = np.argsort(angles)
    #print(sort_order)
    #updated_cross_section['NewOrder'] = sort_order
    
    updated_cross_section['rlnAngleRot'] = angles
    updated_cross_section['rlnAngleRot'] = updated_cross_section['rlnAngleRot'].apply(normalize_angle) 
    #print(updated_cross_section['rlnAngleRot'])
    return updated_cross_section
 
def get_filament_order_from_rot(rotated_cross_section):
    """
    Reorder the doublet number
    Perhaps include new column with old & new number.
    Not working yet
    """
    # Sort the DataFrame by 'rlnAngleRot' in decreasing order
    sorted_df = rotated_cross_section.sort_values(by='rlnAngleRot', ascending=False)
    
    # Extract the 'rlnHelicalTubeID' values in the new order
    sorted_filament_ids = sorted_df['rlnHelicalTubeID'].tolist()
    return sorted_filament_ids
    
# UPDATE: Sorting based on shortest filament length and return sorted filament ids like previously

def path_length(points, path):
    """Compute the total length of a given path."""
    return sum(np.linalg.norm(points[path[i]] - points[path[i-1]]) for i in range(len(path)))

def polygon_signed_area(points, path):
    """Compute the signed area of the polygon formed by the path to determine its orientation."""
    area = 0
    for i in range(len(path)):
        x1, y1 = points[path[i]]
        x2, y2 = points[path[(i+1) % len(path)]]
        area += (x2 - x1) * (y2 + y1)
    return area  # Negative area means clockwise

def find_best_circular_paths(points, top_n=1):
    """Find the top N shortest clockwise circular paths for a given set of points."""
    n = len(points)
    indices = list(range(n))
    best_paths = []
    
    # Try all permutations, fixing the first point to avoid redundancy
    for perm in itertools.permutations(indices[1:]):
        path = [indices[0]] + list(perm)  # Start from a fixed point
        length = path_length(points, path + [path[0]])  # Complete the cycle
        area = polygon_signed_area(points, path)  # Determine path orientation
        
        if area < 0:  # Only keep clockwise paths
            best_paths.append((length, path))
    
    # Sort by path length
    best_paths.sort(key=lambda x: x[0])
    return best_paths[:top_n]
    
def plot_paths(points, paths, output_file=None):
    """Plot the top clockwise paths.
    
    Args:
        points (list or np.array): List of points to plot.
        paths (list): List of tuples containing (length, path).
        output_file (str, optional): If provided, saves the plot to this file instead of showing it.
    """
    fig, axes = plt.subplots(1, len(paths), figsize=(5 * len(paths), 5))
    if len(paths) == 1:
        axes = [axes]
    
    for ax, (length, path) in zip(axes, paths):
        ordered_points = np.array([points[i] for i in path + [path[0]]])
        ax.plot(ordered_points[:, 0], ordered_points[:, 1], 'bo-', markersize=8)
        
        for i, idx in enumerate(path):
            x, y = points[idx]
            ax.text(x, y, str(i), fontsize=12, ha='right', color='red')
        
        ax.set_title(f"Path Length: {length:.2f}")
    
    # Save or show the plot
    if output_file is not None:
        plt.savefig(output_file)  # Save the plot to the specified file
        plt.close()  # Close the figure to free up memory
    else:
        plt.show()  # Display the plot


def renumber_filament_ids(df, best_paths, updated_cross_section):  # Testing cross section sorting
    """
    Renumber the 'rlnHelicalTubeID' column in the DataFrame based on the new order.
    
    Args:
        df (pd.DataFrame): Original DataFrame with 'rlnHelicalTubeID'.
        sorted_tube_ids (list): Sorted list of 'rlnHelicalTubeID' values.
    
    Returns:
        pd.DataFrame: DataFrame with renumbered 'rlnHelicalTubeID'.
    """
    # UPDATE
    sorted_filament_ids = best_paths[0][1]
    sorted_filament_ids = [x + 1 for x in sorted_filament_ids]
    print(sorted_filament_ids)
    
    # Create a mapping from the original IDs to the new order
    id_mapping = {new_id + 1: original_id for new_id, original_id in enumerate(sorted_filament_ids, start=0)}
    print("ID Mapping:", id_mapping)
    updated_cross_section['rlnHelicalTubeID'] = updated_cross_section['rlnHelicalTubeID'].map(id_mapping)
    #print("Updated Cross Section with Renumbered Filament IDs:", updated_cross_section)
    
    df['rlnHelicalTubeID'] = df['rlnHelicalTubeID'].map(id_mapping)
    #print("DataFrame with Renumbered Filament IDs:", df)
    
    return df, updated_cross_section
    

def propagate_rot_to_entire_cilia(cross_section, original_data):
    # Create mappings for adjusted values
    rot_mapping = cross_section.set_index('rlnHelicalTubeID')['rlnAngleRot'].to_dict()

    # Propagate the values to the entire original datasetcr
    original_data['rlnAngleRot'] = original_data['rlnHelicalTubeID'].map(rot_mapping)
    
    return original_data
    
def plot_cs(cross_section, output_png=None):
    """Plot the cross section with points and circular connecting lines.
    
    Args:
        cross_section (pd.DataFrame): DataFrame containing columns 'rlnCoordinateX', 'rlnCoordinateY', and 'rlnHelicalTubeID'.
        output_png (str, optional): If provided, saves the plot to this file instead of showing it.
        The 'rlnCoordinateX/Y/Z' should be in nm for absolute plot
        
    Returns:
        plt: The matplotlib.pyplot object for further customization.
    """
    # Reset the index to ensure it's a simple integer range
    cross_section = cross_section.reset_index(drop=True)
    plt.figure(figsize=(10, 6))

    # Scatter plot for X vs Y
    scatter = plt.scatter(cross_section['rlnCoordinateX'], cross_section['rlnCoordinateY'], 
                          c=cross_section['rlnHelicalTubeID'], cmap='viridis', s=100, edgecolors='k')

    # Plot circular lines connecting the points
    x_coords = cross_section['rlnCoordinateX'].tolist()
    y_coords = cross_section['rlnCoordinateY'].tolist()
    
    # Add the first point at the end to close the loop
    x_coords.append(x_coords[0])
    y_coords.append(y_coords[0])
    
    plt.plot(x_coords, y_coords, 'k-', alpha=0.5)

    # Annotate the points with the filament IDs (rlnHelicalTubeID)
    for i in range(len(cross_section)):
        plt.annotate(cross_section['rlnHelicalTubeID'][i], 
                     (cross_section['rlnCoordinateX'][i], cross_section['rlnCoordinateY'][i]),
                     textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)

    # Set labels and title
    plt.xlabel('X (nm)')
    plt.ylabel('Y (nm)')
    plt.title('Filament Plot: X vs Y with Filament IDs and Circular Connecting Lines')
    
    # Add a color bar to show the filament ID
    plt.colorbar(scatter, label='Filament ID')

    # Add a grid
    plt.grid(True)

    # Save or show the plot
    if output_png is not None:
        plt.savefig(output_png)  # Save the plot to the specified file
        plt.close()  # Close the figure to free up memory
    else:
        return plt  # Return the plt object for further customization
        

def plot_ellipse_cs(cross_section, output_png):
    """
    Plotting the cross section with ellipse fitting.
    
    Args:
        cross_section (pd.DataFrame): DataFrame containing columns 'rlnCoordinateX' and 'rlnCoordinateY'.
        output_png (str): Path to save the output plot.
        
    Returns:
        elliptical_distortion (float): Estimated elliptical distortion, or -1 if ellipse fitting fails.
    """
    points = cross_section[['rlnCoordinateX', 'rlnCoordinateY']].to_numpy()
    x = points[:, 0]
    y = points[:, 1]

    try:
        # Fit an ellipse to these points
        ellipse_params = fit_ellipse(x, y, axis_handle=None)
        print('After ellipse fit')
        print(ellipse_params)
        
        # Check if ellipse parameters are valid
        if ellipse_params['a'] is None or ellipse_params['b'] is None:
            raise ValueError("Ellipse fitting failed: Invalid parameters.")
        
        # Extract ellipse parameters
        center = [ellipse_params['X0'], ellipse_params['Y0']]
        axes = [ellipse_params['a'], ellipse_params['b']]
        angle = ellipse_params['phi']
        
        # Calculate elliptical distortion
        elliptical_distortion = ellipse_params['a'] / ellipse_params['b']
        
        # Generate points for the fitted ellipse
        fitted_ellipse_pts = ellipse_points(center, axes, angle)
        
        # Order the original points along the ellipse
        angles = angle_along_ellipse(center, axes, angle, points)
        angles = angles / np.pi * 180
        
        # Create the base plot using plot_cs
        plt = plot_cs(cross_section)
        
        # Add the fitted ellipse to the plot
        plt.plot(fitted_ellipse_pts[0], fitted_ellipse_pts[1], 'r--', label='Fitted Ellipse')
        
        # Add text annotation for elliptical distortion
        plt.text(np.mean(x), np.mean(y), f"Elliptical distortion: {elliptical_distortion:.2f}", 
                 fontsize=9, ha='center', va='center')
        
        # Add legend and adjust plot
        plt.legend()
        plt.xlabel('X (nm)')
        plt.ylabel('Y (nm)')
        plt.title("Ellipse Fit of Cross section")
        plt.axis('equal')
        
        # Save the plot
        plt.savefig(output_png, dpi=300, bbox_inches='tight')
        plt.close()
        
        return elliptical_distortion
    
    except Exception as e:
        # Handle ellipse fitting errors
        print(f"WARNING: {e}")
        
        # Create the base plot using plot_cs
        plt = plot_cs(cross_section)
        
        # Save the plot without the ellipse
        plt.savefig(output_png, dpi=300, bbox_inches='tight')
        plt.close()
        
        return -1  # Return -1 to indicate failure

def fit_ellipse(x, y, axis_handle=None):
    """
    Fit an ellipse to the given x, y points using the least squares method.
    Returns a dictionary containing ellipse parameters. Converted to Python from fit_ellipse
    matlab script https://www.mathworks.com/matlabcentral/fileexchange/3215-fit_ellipse

    Parameters:
        x, y (array-like): Coordinates of the points.
        axis_handle (matplotlib axis, optional): Axis to plot the ellipse. Default is None.

    Returns:
        dict: Ellipse parameters including center, axes lengths, orientation, etc.
    """
    # Ensure inputs are numpy arrays
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    
    # ERROR CHECK
    if len(x) < 5:
    	print("WARNING: Not enough points to fit an ellipse!")


    # Remove bias (mean) to improve numerical stability
    mean_x, mean_y = np.mean(x), np.mean(y)
    x -= mean_x
    y -= mean_y

    # Build the design matrix
    X = np.column_stack([x**2, x*y, y**2, x, y])

    # Solve the least squares problem
    a, _, _, _ = lstsq(X, -np.ones_like(x), lapack_driver='gelsy')

    # Extract parameters from the solution
    A, B, C, D, E = a

    # Check if the conic equation represents an ellipse
    discriminant = B**2 - 4*A*C
    if discriminant >= 0:
        warnings.warn("The points do not form a valid ellipse (discriminant >= 0).")
        return {
            'a': None,
            'b': None,
            'phi': None,
            'X0': None,
            'Y0': None,
            'X0_in': None,
            'Y0_in': None,
            'long_axis': None,
            'short_axis': None,
            'status': 'Invalid ellipse (discriminant >= 0)'
        }

    # Remove tilt (orientation) from the ellipse
    orientation_rad = 0.5 * np.arctan2(B, (A - C))
    cos_phi, sin_phi = np.cos(orientation_rad), np.sin(orientation_rad)

    # Rotate the ellipse to remove tilt
    A_rot = A * cos_phi**2 - B * cos_phi * sin_phi + C * sin_phi**2
    C_rot = A * sin_phi**2 + B * cos_phi * sin_phi + C * cos_phi**2
    D_rot = D * cos_phi - E * sin_phi
    E_rot = D * sin_phi + E * cos_phi

    # Ensure A_rot and C_rot are positive
    if A_rot < 0 or C_rot < 0:
        A_rot, C_rot, D_rot, E_rot = -A_rot, -C_rot, -D_rot, -E_rot

    # Compute ellipse parameters
    X0 = (mean_x - D_rot / (2 * A_rot))
    Y0 = (mean_y - E_rot / (2 * C_rot))
    F = 1 + (D_rot**2) / (4 * A_rot) + (E_rot**2) / (4 * C_rot)
    a = np.sqrt(F / A_rot)
    b = np.sqrt(F / C_rot)

    # Rotate the center back to the original coordinate system
    R = np.array([[cos_phi, sin_phi], [-sin_phi, cos_phi]])
    X0_in, Y0_in = R @ np.array([X0, Y0])

    # Compute long and short axes
    long_axis = 2 * max(a, b)
    short_axis = 2 * min(a, b)

    # Pack ellipse parameters into a dictionary
    ellipse_t = {
        'a': a,
        'b': b,
        'phi': orientation_rad,
        'X0': X0,
        'Y0': Y0,
        'X0_in': X0_in,
        'Y0_in': Y0_in,
        'long_axis': long_axis,
        'short_axis': short_axis,
        'status': 'Success'
    }

    # Plot the ellipse if axis_handle is provided
    if axis_handle is not None:
        theta = np.linspace(0, 2 * np.pi, 100)
        ellipse_x = X0 + a * np.cos(theta)
        ellipse_y = Y0 + b * np.sin(theta)
        rotated_ellipse = R @ np.vstack([ellipse_x, ellipse_y])

        axis_handle.plot(rotated_ellipse[0, :], rotated_ellipse[1, :], 'r', label='Fitted Ellipse')
        axis_handle.scatter(x + mean_x, y + mean_y, color='blue', label='Points')
        axis_handle.set_aspect('equal', adjustable='box')
        axis_handle.legend()

    return ellipse_t


def ellipse_points(center, axes, angle, num_points=100):
    """
    Generate points on an ellipse.
    center: (x0, y0)
    axes: (a, b) lengths
    angle: rotation angle in radians
    """
    t = np.linspace(0, 2*np.pi, num_points)
    ellipse = np.array([axes[0]*np.cos(t), axes[1]*np.sin(t)])
    # Rotate
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    ellipse_rot = np.dot(R, ellipse)
    ellipse_rot[0] += center[0]
    ellipse_rot[1] += center[1]
    return ellipse_rot


def angle_along_ellipse(center, axes, angle, points):
    """
    Calculate the angle parameters 't' along an ellipse for a list of given points.
    center: (x0, y0) - center of the ellipse
    axes: (a, b) - lengths of the major and minor axes
    angle: rotation angle of the ellipse in radians
    points: list of (x, y) - points on the ellipse
    """
    cos_angle = np.cos(-angle)
    sin_angle = np.sin(-angle)
    a, b = axes
    t_rot_list = []
    
    for point in points:
        # Translate the point and center to the origin
        x_trans = point[0] - center[0]
        y_trans = point[1] - center[1]
        
        # Rotate the point to align with the ellipse's axes
        x_rot = x_trans * cos_angle - y_trans * sin_angle
        y_rot = x_trans * sin_angle + y_trans * cos_angle
        
        # Calculate the angle parameter t
        t = np.arctan2(y_rot / b, x_rot / a)
        
        # Adjust for the rotation angle of the ellipse
        t_rot = t + angle
        t_rot_list.append(t_rot)
    
    return np.array(t_rot_list)