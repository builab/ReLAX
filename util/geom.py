"""
Geom package for ReLAX
Written by Molly Yu & Huy Bui, McGill 
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import eig, inv
from scipy.optimize import leastsq
from scipy.linalg import lstsq
from scipy.optimize import minimize
from scipy.interpolate import splprep, splev
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
The X, Y, Z should be calculated using unbinned pixel
"""
def normalize_angle(angle):
    """
    Normalize angle to range -180 to 180 in Relion
    """
    return (angle + 180) % 360 - 180
    
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
    add 90 degrees to the Psi angle, and return them.
    """
    cross_section = []

    for filament_id, group in data.groupby('rlnHelicalTubeID'):
        points = group[['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']].values
        distances = np.array([calculate_perpendicular_distance(point, plane_normal, reference_point) for point in points])
        closest_idx = np.argmin(distances)
        closest_point = group.iloc[closest_idx]
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
    normal_vector = np.sum(vectors, axis=0)
    return normal_vector / np.linalg.norm(normal_vector)

def process_cross_section(data):
    """ Even if the cross section doesn't have every filament, it can still project it from the shorter filament """
    shortest_filament_id, shortest_midpoint = find_shortest_filament(data)
    filament_points = data[data['rlnHelicalTubeID'] == shortest_filament_id][['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']].values
    normal_vector = calculate_normal_vector(filament_points)
    plane_normal, plane_point = define_plane(normal_vector, shortest_midpoint)
    return find_cross_section_points(data, plane_normal, plane_point)

def rotate_cross_section(cross_section):
    """
    Rotates cross-section by Psi and Tilt angles to transform into Z plane
    """
    rotated_cross_section = cross_section
    psi = 90 - cross_section['rlnAnglePsi'].median()
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
        
        # Rotation around X-axis by Tilt
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
    rot_angles = []
    n = len(rotated_cross_section)
    for i in range(n):
        prev_idx, next_idx = (i - 1) % n, (i + 1) % n
        x_prev, y_prev = rotated_cross_section.iloc[prev_idx][['rlnCoordinateX', 'rlnCoordinateY']]
        x_next, y_next = rotated_cross_section.iloc[next_idx][['rlnCoordinateX', 'rlnCoordinateY']]
        delta_x, delta_y = x_next - x_prev, y_next - y_prev
        rot = np.degrees(np.arctan2(delta_y, delta_x)) - 180
        rot_angles.append(rot)
    
    updated_cross_section['rlnAngleRot'] = rot_angles
    updated_cross_section['rlnAngleRot'] = updated_cross_section['rlnAngleRot'].apply(normalize_angle) 
    #print(updated_cross_section['rlnAngleRot'])
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

def renumber_filament_ids(df, sorted_filament_ids):
    """
    Renumber the 'rlnHelicalTubeID' column in the DataFrame based on the new order.
    
    Args:
        df (pd.DataFrame): Original DataFrame with 'rlnHelicalTubeID'.
        sorted_tube_ids (list): Sorted list of 'rlnHelicalTubeID' values.
    
    Returns:
        pd.DataFrame: DataFrame with renumbered 'rlnHelicalTubeID'.
    """
    # Create a mapping from the original IDs to the new order
    id_mapping = {original_id: new_id for new_id, original_id in enumerate(sorted_filament_ids, start=1)}
    
    # Apply the mapping to the 'rlnHelicalTubeID' column
    df['rlnHelicalTubeID'] = df['rlnHelicalTubeID'].map(id_mapping)
    
    return df
    
def propagate_rot_to_entire_cilia(cross_section, original_data):
    # Create mappings for adjusted values
    rot_mapping = cross_section.set_index('rlnHelicalTubeID')['rlnAngleRot'].to_dict()

    # Propagate the values to the entire original dataset
    original_data['rlnAngleRot'] = original_data['rlnHelicalTubeID'].map(rot_mapping)
    
    return original_data

def plot_ellipse_cs(cross_section, output_png):
    """
    Plotting the cross section
    """
    points = cross_section[['rlnCoordinateX', 'rlnCoordinateY']].to_numpy()
    x = points[:, 0]
    y = points[:, 1]

    # Fit an ellipse to these points
    ellipse_params = fit_ellipse(x, y, axis_handle=None)
    center = [ellipse_params['X0'], ellipse_params['Y0']]
    axes = [ellipse_params['a'], ellipse_params['b']]
    angle = ellipse_params['phi']

    elliptical_distortion = ellipse_params['a']/ellipse_params['b']
    fitted_ellipse_pts = ellipse_points(center, axes, angle)

    # Order the original points along the ellipse:
    angles = angle_along_ellipse(center, axes, angle, points)
    angles = angles/np.pi*180

    # Plotting the results
    plt.figure(figsize=(8, 6))
    plt.plot(fitted_ellipse_pts[0], fitted_ellipse_pts[1], 'r--', label='Fitted Ellipse')
    plt.scatter(x, y, c='b', label='Doublet Number')
    for i, pt in enumerate(points):
        plt.text(x[i]+0.5, y[i]+0.5, str(i), fontsize=10)
        
    plt.text(np.mean(x), np.mean(y), f"Elliptical distortion: {elliptical_distortion:.2f}", fontsize=9, ha='center', va='center')
    plt.legend()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Ellipse Fit of Cross section")
    plt.axis('equal')
    plt.savefig(output_png, dpi=300, bbox_inches='tight')

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

