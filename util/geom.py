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
    
    #print(dense_points)
    
    # Calculate cumulative distances
    cum_distances_all = cumulative_distance(dense_points * angpix)
    
    print(cum_distances_all[-1])
    print(f"{spacing}")
        
    # Check if the total distance is zero or very small
    if cum_distances_all[-1] < spacing * 0.1:
        # Path is too short, return the original points
        print(cum_distance_alls[-1])
        print(spacing)
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

# Redundant function, remove
def define_plane(normal_vector, reference_point):
    return normal_vector, reference_point
    
def calculate_perpendicular_distance(point, plane_normal, reference_point):
    return np.abs(np.dot(plane_normal, point - reference_point)) / np.linalg.norm(plane_normal)

def find_cross_section_points(data, plane_normal, reference_point):
    """
    Find the points on each filament closest to the cross-sectional plane,
    and record their distances.
    """
    # UPDATE: keep track of the max distance of the point that make up the cross section
    cross_section = []
    max_distance = 0
    grouped_data = data.groupby('rlnHelicalTubeID')
    
    for filament_id, group in grouped_data:
        points = group[['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']].values
        distances = np.array([calculate_perpendicular_distance(point, plane_normal, reference_point) for point in points])
        min_distance = np.min(distances)
        max_distance = max(max_distance, min_distance)  # track global max distance
        
        closest_point = group.iloc[np.argmin(distances)].copy()
        closest_point['distance_to_plane'] = min_distance  # add distance info
        cross_section.append(closest_point)

    df_cross_section = pd.DataFrame(cross_section)
    return df_cross_section, max_distance

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
    normal_vector = normal_vector / np.linalg.norm(normal_vector)
    # UPDATE
    if normal_vector[2] < 0:
        normal_vector = -normal_vector

    return normal_vector

# UPDATE: obtain normal vector with local averaging 
def calculate_normal_vector(filament_points, window_size=3):
    """
    Calculate normal vector using average of vectors near midpoint.
    Args:
        filament_points (np.ndarray): Nx3 array of filament points
        window_size (int): How many points before and after midpoint to use
    Returns:
        np.ndarray: normalized normal vector
    """
    n_points = filament_points.shape[0]
    mid_idx = n_points // 2  # midpoint index

    # Define the start and end index to avoid out-of-bounds
    start_idx = max(mid_idx - window_size, 0)
    end_idx = min(mid_idx + window_size, n_points - 1)

    # Collect vectors between consecutive points
    vectors = []
    for i in range(start_idx, end_idx):
        v = filament_points[i + 1] - filament_points[i]
        vectors.append(v)

    vectors = np.array(vectors)

    # Average vector
    avg_vector = np.mean(vectors, axis=0)

    # Normalize
    normal_vector = avg_vector / np.linalg.norm(avg_vector)
    
    return normal_vector

def process_cross_section(data):
    """ Even if the cross section doesn't have every filament, it can still project it from the shorter filament 
        Find automatically all the cross section points from middle of the shortest filament
    """
    shortest_filament_id, shortest_midpoint = find_shortest_filament(data)
    #print(f"{shortest_filament_id}, {shortest_midpoint}")
    filament_points = data[data['rlnHelicalTubeID'] == shortest_filament_id][['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']].values
    #print(filament_points)
    # UPDATE: normal vector obtained from local normal vector function
    normal_vector = calculate_normal_vector(filament_points)
    #plane_normal, plane_point = define_plane(normal_vector, shortest_midpoint)
    # UPDATE: removed define_plane redundancy
    return find_cross_section_points(data, normal_vector, shortest_midpoint)
    
    
def process_specific_cross_section(pts_idx, data):
    """ Modify to calculate the cross section points using a specific point
    """
    filament_id = data.loc[pts_idx, 'rlnHelicalTubeID']
    pts = data.loc[pts_idx, ['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']].values
    #print(f"{pts}, {filament_id}")
    filament_points = data[data['rlnHelicalTubeID'] == filament_id][['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']].values
    #print(filament_points)
    normal_vector = calculate_normal_vector(filament_points)
    return find_cross_section_points(data, normal_vector, pts)

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

def enforce_consistent_cross_section_orientation(df):
    """
    Make sure rotated cross section is clockwise and right-facing.
    """
    points = df[['rlnCoordinateX', 'rlnCoordinateY']].to_numpy()

    # Check polygon signed area
    area = polygon_signed_area(points)
    if area > 0:
        # Counterclockwise → Flip Y
        df['rlnCoordinateY'] *= -1

    # Check mean X direction
    if df['rlnCoordinateX'].mean() < 0:
        # Mostly on left → Flip X
        df['rlnCoordinateX'] *= -1

    return df

    
def calculate_rot_angles(rotated_cross_section, fit_method):
    """ Calculate the rotation angle in a cross section """    
    fit_dispatch = {
        'simple': calculate_rot_angles_simple,
        'ellipse': calculate_rot_angles_ellipse,
        # add more methods here if needed
    }

    func = fit_dispatch.get(fit_method)

    if func is None:
        print(f"Unknown fit_method '{fit_method}'. Falling back to 'simple'.")
        func = calculate_rot_angles_simple

    return func(rotated_cross_section)
           
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


# UPDATE: Helper function to handle rot angle calculation missing filament case
def detect_multiple_missing_points(rot_angles, gap_threshold=50):
    """
    Detect one or more large gaps in rot angles → indicate multiple missing filaments.

    Returns:
        List of (gap_start_angle, gap_end_angle)
    """
    sorted_angles = np.sort(rot_angles)
    gaps = []
    gap_pairs = []

    for i in range(len(sorted_angles)):
        next_angle = sorted_angles[(i + 1) % len(sorted_angles)]
        delta = (next_angle - sorted_angles[i]) % 360
        gaps.append(delta)

    for idx, delta in enumerate(gaps):
        if delta > gap_threshold:
            gap_start_angle = sorted_angles[idx]
            gap_end_angle = sorted_angles[(idx + 1) % len(sorted_angles)]
            gap_pairs.append((gap_start_angle, gap_end_angle))

    return gap_pairs

def calculate_ellipse_point(ellipse_params, theta_deg):
    """
    Calculate (x, y) on ellipse at given theta.

    Args:
        ellipse_params (dict): Ellipse parameters (X0, Y0, a, b, phi).
        theta_deg (float): Angle (deg) along ellipse.

    Returns:
        (x, y): Coordinates on ellipse.
    """
    theta = np.radians(theta_deg)
    phi = ellipse_params['phi']

    x0 = ellipse_params['X0']
    y0 = ellipse_params['Y0']
    a = ellipse_params['a']
    b = ellipse_params['b']

    x = x0 + a * np.cos(theta) * np.cos(phi) - b * np.sin(theta) * np.sin(phi)
    y = y0 + a * np.cos(theta) * np.sin(phi) + b * np.sin(theta) * np.cos(phi)

    return x, y    

    
def calculate_rot_angles_ellipse(rotated_cross_section):
    """ 
    Calculate rotation angle in a cross section using ellipse method,
    and handle missing filament if filament count < 9.
    Returns both the final cross section and the version with virtual point.
    """
    updated_cross_section = rotated_cross_section.copy()
    points = updated_cross_section[['rlnCoordinateX', 'rlnCoordinateY']].to_numpy()
    x = points[:, 0]
    y = points[:, 1]

    # Fit an ellipse to these points
    ellipse_params = fit_ellipse(x, y, axis_handle=None)
    center = [ellipse_params['X0'], ellipse_params['Y0']]
    axes = [ellipse_params['a'], ellipse_params['b']]
    phi = ellipse_params['phi']

    print(f"Fitted center: {center}")
    print(f"Fitted axes: {axes}")
    print(f"Fitted rotation (radians): {phi}")
    elliptical_distortion = ellipse_params['a'] / ellipse_params['b']
    print(f"Elliptical distortion: {elliptical_distortion :.2f}")

    # Calculate base rot angles
    angles = angle_along_ellipse(center, axes, phi, points)
    angles = np.degrees(angles) - 270
    angles = np.vectorize(normalize_angle)(angles)
    updated_cross_section['rlnAngleRot'] = angles

    cross_section_with_virtual = updated_cross_section.copy()

    # --- If filament count is incomplete ---
    num_filaments = updated_cross_section['rlnHelicalTubeID'].nunique()
    if num_filaments < 9:
        # === UPDATED FOR MULTIPLE MISSING POINTS ===
        gap_infos = detect_multiple_missing_points(updated_cross_section['rlnAngleRot'])

        for missing_idx, (gap_start_angle, gap_end_angle) in enumerate(gap_infos):
            print(f"Missing point detected! Gap between {gap_start_angle:.1f}° and {gap_end_angle:.1f}°")

            # Virtual point at +40 degrees from gap_start
            virtual_theta = normalize_angle(gap_start_angle - 40)
            virtual_x, virtual_y = calculate_ellipse_point(ellipse_params, virtual_theta)

            print(f"\n Virtual point {missing_idx+1} (θ={virtual_theta:.1f}°) at ({virtual_x:.2f}, {virtual_y:.2f})")

            # Create a dummy virtual point
            dummy = updated_cross_section.iloc[0].copy()
            dummy['rlnCoordinateX'] = virtual_x
            dummy['rlnCoordinateY'] = virtual_y
            dummy['rlnCoordinateZ'] = updated_cross_section['rlnCoordinateZ'].mean()
            dummy['rlnHelicalTubeID'] = 999 + missing_idx  # ensure unique ID
            dummy['rlnAngleRot'] = virtual_theta

            # Add dummy to cross_section_with_virtual
            cross_section_with_virtual = pd.concat([cross_section_with_virtual, pd.DataFrame([dummy])], ignore_index=True)

            # Recalculate angles on temp set with dummy
            temp_angles = angle_along_ellipse(center, axes, phi, cross_section_with_virtual[['rlnCoordinateX', 'rlnCoordinateY']].values)
            temp_angles = np.degrees(temp_angles) - 270
            temp_angles = np.vectorize(normalize_angle)(temp_angles)
            cross_section_with_virtual['rlnAngleRot'] = temp_angles

            # Find neighbors to virtual point
            neighbors = updated_cross_section.copy()
            neighbors['diff_start'] = np.abs(neighbors['rlnAngleRot'] - gap_start_angle)
            neighbors['diff_end'] = np.abs(neighbors['rlnAngleRot'] - gap_end_angle)

            neighbor_start_idx = neighbors['diff_start'].idxmin()
            neighbor_end_idx = neighbors['diff_end'].idxmin()

            for neighbor_idx in [neighbor_start_idx, neighbor_end_idx]:
                point = updated_cross_section.loc[neighbor_idx]
                nx, ny = point['rlnCoordinateX'], point['rlnCoordinateY']

                new_rot = np.degrees(np.arctan2(virtual_y - ny, virtual_x - nx))
                new_rot = normalize_angle(new_rot)
                updated_cross_section.loc[neighbor_idx, 'rlnAngleRot'] = new_rot

                print(f"Updated neighbor idx {neighbor_idx} rot angle to {new_rot:.1f}°")

    return updated_cross_section, cross_section_with_virtual

def calculate_rot_angle_twolines(rotated_cross_section, tube_id):
    """ 
    Calculate the rotation angle in a cross section
    This return only rot angles, not the entire dataframe
    """
    updated_cross_section = rotated_cross_section
    coords = rotated_cross_section[['rlnHelicalTubeID', 'rlnCoordinateX', 'rlnCoordinateY']].values
    # If the line line or 2nd line is calculated
    #print(coords[0, 1:2])
    delta_x = coords[1, 1] - coords[0, 1]  
    delta_y = coords[1, 2] - coords[0, 2]
    if coords[0, 0] == tube_id:
        rot_angle = np.degrees(np.arctan2(delta_y, delta_x))
    else:
        rot_angle = np.degrees(np.arctan2(delta_y, delta_x)) - 180

    return rot_angle
    
def get_filament_order(cs, fit_method):
    """
    Wrapper function to calculate the filament_order based on either rot (ellipse) or length (simple)
    """
    if fit_method == 'ellipse':
        return get_filament_order_from_rot(cs)
    else:
        return get_filament_order_from_length(cs)

def get_filament_order_from_rot(rotated_cross_section):
    """
    Reorder the doublet number using ellipse method
    """
    # Sort the DataFrame by 'rlnAngleRot' in decreasing order
    sorted_df = rotated_cross_section.sort_values(by='rlnAngleRot', ascending=False)
    
    # Extract the 'rlnHelicalTubeID' values in the new order
    sorted_filament_ids = sorted_df['rlnHelicalTubeID'].tolist()
    return sorted_filament_ids
    
# UPDATE: Sorting based on shortest filament length and return sorted filament ids like previously
def get_filament_order_from_length(rotated_cross_section):
    """
    Reorder the doublet number using length method
    """
    points = rotated_cross_section[['rlnCoordinateX', 'rlnCoordinateY']].values
    best_paths = find_best_circular_paths(points)

    # Extract the 'rlnHelicalTubeID' values in the new order
    sorted_filament_ids = best_paths[0][1]
    sorted_filament_ids = [x + 1 for x in sorted_filament_ids]  # Renumber filament IDs
    return sorted_filament_ids

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


def renumber_filament_ids(df, sorted_filament_ids, updated_cross_section):  # Testing cross section sorting
    """
    Renumber the 'rlnHelicalTubeID' column in the DataFrame based on the new order.
    
    Args:
        df (pd.DataFrame): Original DataFrame with 'rlnHelicalTubeID'.
        sorted_tube_ids (list): Sorted list of 'rlnHelicalTubeID' values.
    
    Returns:
        pd.DataFrame: DataFrame with renumbered 'rlnHelicalTubeID'.
    """
    print("Sorted Filament IDs (from rotation):", sorted_filament_ids)
    
    # Create a mapping from the original IDs to the new order
    id_mapping = {original_id: new_id + 1 for new_id, original_id in enumerate(sorted_filament_ids, start=0)}
    print("ID Mapping:", id_mapping)
    
    # Map the 'rlnHelicalTubeID' in both df and updated_cross_section using the mapping
    updated_cross_section['rlnHelicalTubeID'] = updated_cross_section['rlnHelicalTubeID'].map(id_mapping)
    df['rlnHelicalTubeID'] = df['rlnHelicalTubeID'].map(id_mapping)
    
    print("RENUMBER: ", updated_cross_section)
    print("COMPLETE DF: ", df)
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
        

def plot_ellipse_cs(cross_section, output_png, full_star_data=None):
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
        #print('After ellipse fit')
        #print(ellipse_params)
        
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

        # If full_star_data is provided, extract rot angles for matching HelicalTubeIDs
        if full_star_data is not None and 'rlnHelicalTubeID' in cross_section.columns:
            rot_lookup = (
                full_star_data[['rlnHelicalTubeID', 'rlnAngleRot']]
                .drop_duplicates(subset='rlnHelicalTubeID')
                .set_index('rlnHelicalTubeID')['rlnAngleRot']
                .to_dict()
            )
            # Add rot angle to each row if available
            cross_section['rlnAngleRot'] = cross_section['rlnHelicalTubeID'].map(rot_lookup)

            # Plot rotation vectors
            for _, row in cross_section.iterrows():
                if not np.isnan(row.get('rlnAngleRot', np.nan)):
                    x0, y0 = row['rlnCoordinateX'], row['rlnCoordinateY']
                    theta = np.deg2rad(row['rlnAngleRot'])
                    dx, dy = np.cos(theta) * 10, np.sin(theta) * 10  # arrow scale
                    plt.arrow(x0, y0, dx, dy, head_width=3, head_length=5, fc='blue', ec='blue', alpha=0.7)

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
