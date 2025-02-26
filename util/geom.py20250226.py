import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import eig, inv
from scipy.optimize import leastsq
from scipy.optimize import minimize
from scipy.interpolate import splprep, splev

"""
The X, Y, Z should be calculated using unbinned pixel
"""

def interpolate_spline(points, angpix, spacing):
    """
    Interpolate points along a line with a specified spacing using splines.
    TODO: Need to deal with error if this is too short? Min 5 particles?
    """
    points = np.array(points)
    #print(points.T)
    tck, _ = splprep(points.T, s=0)  # Create spline representation
    distances = np.linspace(0, 1, int(np.ceil(points.shape[0] * 100)))  # Dense samples. 100 times the points?
    
    # Evaluate the spline at dense samples
    dense_points = np.array(splev(distances, tck)).T
    
    # Calculate cumulative distances along the spline
    diffs = np.diff(dense_points, axis=0)
    cumulative_distances = np.insert(np.cumsum(np.sqrt((diffs**2).sum(axis=1))), 0, 0)*angpix
    #print(cumulative_distances)
    
    # Resample at equal intervals
    resampled_distances = np.arange(0, cumulative_distances[-1], spacing)
    #print(resampled_distances)
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

def fit_ellipse_cs(cross_section, dodraw):
    """
    Using fit_ellipse to fit on cross section
    """
    points = cross_section[['rlnCoordinateX', 'rlnCoordinateY']].to_numpy()
    #print(points)
    x = points[:, 0]
    y = points[:, 1]
    # Centralize points
    x = np.array(x) - np.mean(x)
    y = np.array(y) - np.mean(y)
    points = np.column_stack((x, y))
    #print(points)

    # Fit an ellipse to these points
    ellipse_params_fit = fit_ellipse(x, y)
    x0, y0, axis1, axis2, angle = ellipse_params(ellipse_params_fit)
     
    print(f"Fitted center: {x0}, {y0}")
    print(f"Fitted axes: {axis1}, {axis2}")
    print(f"Fitted rotation (radians): {angle}")

    fitted_ellipse_pts = ellipse_points([x0, y0], [axis1, axis2], angle)

    # Order the original points along the ellipse:
    angles = angle_along_ellipse([x0, y0], [axis1, axis2], angle, points)
    angles = angles/np.pi*180
    print(f"Angles {angles}")
    sort_order = np.argsort(angles)
    ordered_points = points[sort_order]

    # Plotting the results
    if dodraw:
        plt.figure(figsize=(8, 6))
        plt.plot(fitted_ellipse_pts[0], fitted_ellipse_pts[1], 'r--', label='Fitted Ellipse')
        plt.scatter(x, y, c='b', label='Original Points')
        plt.scatter(ordered_points[:,0], ordered_points[:,1], 
        c=angles[sort_order], cmap='viridis', s=80, label='Ordered Points')
        for i, pt in enumerate(ordered_points):
            plt.text(pt[0]+0.1, pt[1]+0.1, str(i), fontsize=9)
        plt.legend()
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Ellipse Fit and Point Ordering")
        plt.axis('equal')
        plt.show()


def fit_ellipse(x, y):
    
    D1 = np.vstack([x**2, x*y, y**2]).T
    D2 = np.vstack([x, y, np.ones_like(x)]).T
    S1 = np.dot(D1.T, D1)
    S2 = np.dot(D1.T, D2)
    S3 = np.dot(D2.T, D2)
    
    T = -np.linalg.inv(S3).dot(S2.T)
    M = S1 + S2.dot(T)
    M = np.array(M)
    
    C = np.zeros((3, 3))
    C[0, 2] = C[2, 0] = 2
    C[1, 1] = -1

    E, V = np.linalg.eig(np.linalg.inv(M).dot(C))
    n = np.argmax(np.abs(E))
    a = V[:, n]

    def ellipse_residuals(params, x, y):
        a, b, c, d, e, f = params
        return ((a * x**2 + b * x * y + c * y**2 + d * x + e * y + f)**2).sum()
    
    params_initial = [a[0], a[1], a[2], 0, 0, -1]
    result = minimize(ellipse_residuals, params_initial, args=(x, y))
    return result.x
    
def ellipse_params(params, threshold=1e-10):
    """ Convert ellipse coefficients to center, axes, and rotation angle """
    a, b, c, d, e, f = params
    num = b**2 - 4*a*c

    if abs(num) < threshold:
        # Handle the special case of a circle or near-circle
        x0 = -d / (2*a)
        y0 = -e / (2*c)
        axis1 = axis2 = np.sqrt(d**2 + e**2 - 4*a*f) / (2*a)
        angle = 0  # No rotation angle for a circle
    else:
        x0 = (2*c*d - b*e) / num
        y0 = (2*a*e - b*d) / num
        angle = 0.5 * np.arctan2(b, a - c)

        up = 2 * (a*e**2 + c*d**2 - b*d*e + num*f)
        down1 = (b**2 - 4*a*c) * ((c-a) + np.sqrt((a-c)**2 + b**2))
        down2 = (b**2 - 4*a*c) * ((c-a) - np.sqrt((a-c)**2 + b**2))

        axis1 = np.sqrt(up / down1)
        axis2 = np.sqrt(up / down2)

    return x0, y0, axis1, axis2, angle
    

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

