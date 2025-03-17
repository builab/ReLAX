import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import sys

def load_points_from_file(filename):
    """
    Load 3D points from a space-separated text file.
    
    Parameters:
    -----------
    filename : str
        Path to the input file
        
    Returns:
    --------
    points : ndarray of shape (n, 3)
        The loaded 3D points
    """
    try:
        points = np.loadtxt(filename, dtype=float)
        
        # Check if the file has the right format
        if points.ndim == 1 and len(points) == 3:
            # Handle the case of a single point
            points = points.reshape(1, 3)
        elif points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("Input file must contain three columns for X, Y, Z coordinates")
            
        print(f"Successfully loaded {points.shape[0]} points from {filename}")
        return points
    except Exception as e:
        print(f"Error loading points from file: {e}")
        sys.exit(1)

def rotate_pc1_to_z(points):
    """
    Rotate a cluster of points such that the first principal component (PC1)
    aligns with the Z-axis.
    
    Parameters:
    -----------
    points : ndarray of shape (n, 3)
        The input 3D points
        
    Returns:
    --------
    rotated_points : ndarray of shape (n, 3)
        The rotated points
    angles : tuple
        (theta_z, theta_y) rotation angles in radians
    pca : sklearn.decomposition.PCA
        The fitted PCA object
    rotation_matrices : tuple
        (rotation_z, rotation_y) rotation matrices
    """
    # Center the points
    center = np.mean(points, axis=0)
    points_centered = points - center
    
    # Perform PCA to find principal components
    pca = PCA(n_components=3)
    pca.fit(points_centered)
    
    # Get the first principal component (PC1)
    pc1 = pca.components_[0]
    
    # Ensure PC1 points in the positive direction (arbitrary choice)
    if pc1[2] < 0:
        pc1 = -pc1
    
    print(f"PC1 (first principal component): {pc1}")
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    
    # Calculate the angle around Z-axis to align PC1's projection onto XY plane with X-axis
    # Project PC1 onto the XY plane
    projection_xy = np.array([pc1[0], pc1[1], 0])
    norm_xy = np.linalg.norm(projection_xy)
    
    if norm_xy < 1e-6:
        # If PC1 is already aligned with Z-axis (or very close)
        theta_z = 0
        print("PC1 is already nearly parallel to Z-axis, skipping Z rotation")
    else:
        # Normalize the projection
        projection_xy = projection_xy / norm_xy
        # Calculate the angle between the projection and the X-axis
        cos_theta_z = np.dot(projection_xy, np.array([1, 0, 0]))
        theta_z = np.arccos(np.clip(cos_theta_z, -1.0, 1.0))
        # Determine the sign of the angle
        if projection_xy[1] < 0:
            theta_z = -theta_z
    
    # Create rotation matrix around Z-axis
    rotation_z = np.array([
        [np.cos(theta_z), -np.sin(theta_z), 0],
        [np.sin(theta_z), np.cos(theta_z), 0],
        [0, 0, 1]
    ])
    
    # Apply first rotation to align PC1's projection with X-axis
    intermediate_points = np.dot(points_centered, rotation_z.T)
    
    # Apply the same rotation to PC1
    pc1_rotated_z = np.dot(pc1, rotation_z.T)
    print(f"PC1 after Z rotation: {pc1_rotated_z}")
    
    # Calculate the angle around Y-axis to align PC1 with Z-axis
    # Project PC1 onto XZ plane after Z rotation
    projection_xz = np.array([pc1_rotated_z[0], 0, pc1_rotated_z[2]])
    norm_xz = np.linalg.norm(projection_xz)
    
    if norm_xz < 1e-6:
        # If PC1 is already aligned with Y-axis
        theta_y = 0
        print("After Z rotation, PC1 is already aligned with Y-axis, skipping Y rotation")
    else:
        # Normalize the projection
        projection_xz = projection_xz / norm_xz
        # Calculate the angle between the projection and the Z-axis
        cos_theta_y = np.dot(projection_xz, np.array([0, 0, 1]))
        theta_y = np.arccos(np.clip(cos_theta_y, -1.0, 1.0))
        # Determine the sign of the angle (negative if X component is positive)
        if projection_xz[0] > 0:
            theta_y = -theta_y
    
    # Create rotation matrix around Y-axis
    rotation_y = np.array([
        [np.cos(theta_y), 0, np.sin(theta_y)],
        [0, 1, 0],
        [-np.sin(theta_y), 0, np.cos(theta_y)]
    ])
    
    # Apply second rotation to align PC1 with Z-axis
    final_points_centered = np.dot(intermediate_points, rotation_y.T)
    
    # Apply the same rotation to PC1 to verify alignment with Z-axis
    pc1_final = np.dot(pc1_rotated_z, rotation_y.T)
    print(f"PC1 after both rotations: {pc1_final}")
    
    # Verify PC1 is now aligned with Z-axis
    z_axis = np.array([0, 0, 1])
    alignment = np.abs(np.dot(pc1_final, z_axis))
    print(f"Alignment of PC1 with Z-axis: {alignment:.6f} (should be close to 1.0)")
    
    # Add back the center to get the final rotated points
    final_points = final_points_centered + center
    
    return final_points, (theta_z, theta_y), pca, (rotation_z, rotation_y)

def extract_middle_slab(points, fraction=1/3):
    """
    Extract a middle slab of points in the Z direction.
    
    Parameters:
    -----------
    points : ndarray of shape (n, 3)
        The input 3D points
    fraction : float
        The fraction of the Z range to include in the slab
        
    Returns:
    --------
    slab_points : ndarray of shape (m, 3)
        The points in the middle slab
    slab_indices : ndarray
        The indices of the slab points in the original array
    z_min : float
        The minimum Z value of the slab
    z_max : float
        The maximum Z value of the slab
    """
    z_min = np.min(points[:, 2])
    z_max = np.max(points[:, 2])
    z_range = z_max - z_min
    
    # Calculate the middle slab boundaries
    slab_height = z_range * fraction
    slab_center = (z_max + z_min) / 2
    slab_min = slab_center - slab_height / 2
    slab_max = slab_center + slab_height / 2
    
    # Extract points in the middle slab
    slab_indices = np.where((points[:, 2] >= slab_min) & (points[:, 2] <= slab_max))[0]
    slab_points = points[slab_indices]
    
    print(f"Full point cloud Z range: {z_min:.6f} to {z_max:.6f}")
    print(f"Middle slab Z range: {slab_min:.6f} to {slab_max:.6f} ({fraction*100:.1f}% of total)")
    print(f"Selected {len(slab_points)} points out of {len(points)} ({len(slab_points)/len(points)*100:.1f}%)")
    
    return slab_points, slab_indices, slab_min, slab_max

def visualize_points(points, slab_points=None, slab_indices=None, 
                     slab_min=None, slab_max=None, title="Point Cloud"):
    """
    Visualize points with optional slab highlighting.
    
    Parameters:
    -----------
    points : ndarray of shape (n, 3)
        The input 3D points
    slab_points : ndarray of shape (m, 3), optional
        The points in the middle slab
    slab_indices : ndarray, optional
        The indices of the slab points in the original array
    slab_min : float, optional
        The minimum Z value of the slab
    slab_max : float, optional 
        The maximum Z value of the slab
    title : str
        The title of the plot
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    if slab_points is not None:
        # Plot non-slab points in light gray
        non_slab_indices = np.setdiff1d(np.arange(len(points)), slab_indices)
        ax.scatter(points[non_slab_indices, 0], points[non_slab_indices, 1], points[non_slab_indices, 2], 
                  c='lightgray', marker='o', alpha=0.3, label='Non-slab points')
        
        # Plot slab points in blue
        ax.scatter(slab_points[:, 0], slab_points[:, 1], slab_points[:, 2], 
                  c='blue', marker='o', alpha=0.8, label='Slab points')
        
        # Add slab boundaries
        if slab_min is not None and slab_max is not None:
            x_min, x_max = ax.get_xlim()
            y_min, y_max = ax.get_ylim()
            xx, yy = np.meshgrid([x_min, x_max], [y_min, y_max])
            
            # Create lower and upper slab planes
            z_lower = np.ones_like(xx) * slab_min
            z_upper = np.ones_like(xx) * slab_max
            
            # Plot transparent planes at slab boundaries
            ax.plot_surface(xx, yy, z_lower, alpha=0.2, color='blue')
            ax.plot_surface(xx, yy, z_upper, alpha=0.2, color='blue')
    else:
        # Plot all points in blue
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                  c='blue', marker='o', alpha=0.5)
    
    # Perform PCA to show principal directions
    pca = PCA(n_components=3)
    points_centered = points - np.mean(points, axis=0)
    pca.fit(points_centered)
    
    # Plot principal directions
    center = np.mean(points, axis=0)
    colors = ['red', 'green', 'blue']
    labels = ['PC1', 'PC2', 'PC3']
    
    # Get the range of the data for scaling
    max_range = np.max([
        np.max(points[:, 0]) - np.min(points[:, 0]),
        np.max(points[:, 1]) - np.min(points[:, 1]),
        np.max(points[:, 2]) - np.min(points[:, 2])
    ]) / 4
    
    # Plot principal directions
    for i, (ev, c, label) in enumerate(zip(pca.components_, colors, labels)):
        scale = np.sqrt(pca.explained_variance_[i]) * 2  # Scale by explained variance
        ax.quiver(center[0], center[1], center[2], 
                 ev[0], ev[1], ev[2], 
                 length=scale, color=c, label=label)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()
    
    # Create equal aspect ratio
    set_axes_equal(ax)
    
    plt.tight_layout()
    plt.show()

def visualize_rotation_steps(original_points, first_rotation_points, 
                             slab_points=None, slab_rotation_points=None):
    """
    Visualize original points, first rotation, and slab rotation points.
    
    Parameters:
    -----------
    original_points : ndarray of shape (n, 3)
        The original points
    first_rotation_points : ndarray of shape (n, 3)
        The points after first PC1 to Z rotation
    slab_points : ndarray of shape (m, 3), optional
        The slab points
    slab_rotation_points : ndarray of shape (n, 3), optional
        The points after second rotation based on slab
    """
    fig = plt.figure(figsize=(18, 10))
    
    # Original points
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(original_points[:, 0], original_points[:, 1], original_points[:, 2], 
               c='blue', marker='o', alpha=0.5)
    
    pca = PCA(n_components=3)
    points_centered = original_points - np.mean(original_points, axis=0)
    pca.fit(points_centered)
    
    # Plot original principal components
    center = np.mean(original_points, axis=0)
    colors = ['red', 'green', 'blue']
    labels = ['PC1', 'PC2', 'PC3']
    
    # Get the range of the data for scaling
    max_range = np.max([
        np.max(original_points[:, 0]) - np.min(original_points[:, 0]),
        np.max(original_points[:, 1]) - np.min(original_points[:, 1]),
        np.max(original_points[:, 2]) - np.min(original_points[:, 2])
    ]) / 4
    
    # Plot principal directions
    for i, (ev, c, label) in enumerate(zip(pca.components_, colors, labels)):
        scale = np.sqrt(pca.explained_variance_[i])  # Scale by explained variance
        ax1.quiver(center[0], center[1], center[2], 
                  ev[0], ev[1], ev[2], 
                  length=scale, color=c, label=label)
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Original Points')
    ax1.legend()
    
    # Create equal aspect ratio
    set_axes_equal(ax1)
    
    # First rotation points
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(first_rotation_points[:, 0], first_rotation_points[:, 1], first_rotation_points[:, 2], 
               c='green', marker='o', alpha=0.5)
    
    pca2 = PCA(n_components=3)
    points_centered2 = first_rotation_points - np.mean(first_rotation_points, axis=0)
    pca2.fit(points_centered2)
    
    # Plot rotated principal components
    center2 = np.mean(first_rotation_points, axis=0)
    
    # Plot principal directions
    for i, (ev, c, label) in enumerate(zip(pca2.components_, colors, labels)):
        scale = np.sqrt(pca2.explained_variance_[i])  # Scale by explained variance
        ax2.quiver(center2[0], center2[1], center2[2], 
                  ev[0], ev[1], ev[2], 
                  length=scale, color=c, label=label)
    
    # Add coordinate axes
    axis_length = max_range * 0.3
    ax2.quiver(center2[0], center2[1], center2[2], 
               axis_length, 0, 0, color='gray', label='X-axis')
    ax2.quiver(center2[0], center2[1], center2[2], 
               0, axis_length, 0, color='gray', label='Y-axis')
    ax2.quiver(center2[0], center2[1], center2[2], 
               0, 0, axis_length, color='gray', label='Z-axis')
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('First Rotation (PC1 to Z)')
    ax2.legend()
    
    # Create equal aspect ratio
    set_axes_equal(ax2)
    
    # Second rotation points (slab-based)
    if slab_rotation_points is not None:
        ax3 = fig.add_subplot(133, projection='3d')
        ax3.scatter(slab_rotation_points[:, 0], slab_rotation_points[:, 1], slab_rotation_points[:, 2],
                   c='red', marker='o', alpha=0.5)
        
        # Perform PCA on the final rotated points
        pca3 = PCA(n_components=3)
        points_centered3 = slab_rotation_points - np.mean(slab_rotation_points, axis=0)
        pca3.fit(points_centered3)
        
        # Plot final rotated principal components
        center3 = np.mean(slab_rotation_points, axis=0)
        
        # Plot principal directions
        for i, (ev, c, label) in enumerate(zip(pca3.components_, colors, labels)):
            scale = np.sqrt(pca3.explained_variance_[i])  # Scale by explained variance
            ax3.quiver(center3[0], center3[1], center3[2], 
                      ev[0], ev[1], ev[2], 
                      length=scale, color=c, label=label)
        
        # Add coordinate axes
        ax3.quiver(center3[0], center3[1], center3[2], 
                   axis_length, 0, 0, color='gray', label='X-axis')
        ax3.quiver(center3[0], center3[1], center3[2], 
                   0, axis_length, 0, color='gray', label='Y-axis')
        ax3.quiver(center3[0], center3[1], center3[2], 
                   0, 0, axis_length, color='gray', label='Z-axis')
        
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')
        ax3.set_title('Final Rotation (Slab PC1 to Z)')
        ax3.legend()
        
        # Create equal aspect ratio
        set_axes_equal(ax3)
    
    plt.tight_layout()
    plt.show()

def save_rotated_points(rotated_points, output_filename="rotated_points.txt"):
    """
    Save rotated points to a space-separated text file.
    
    Parameters:
    -----------
    rotated_points : ndarray of shape (n, 3)
        The rotated points to save
    output_filename : str
        Path to the output file
    """
    try:
        np.savetxt(output_filename, rotated_points, fmt='%.6f', delimiter=' ')
        print(f"Successfully saved {rotated_points.shape[0]} rotated points to {output_filename}")
    except Exception as e:
        print(f"Error saving rotated points: {e}")

def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.
    
    Parameters:
    -----------
    ax : matplotlib 3D axis object
        The axis to adjust
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def main():
    """Main function to execute the program."""
    # Check if filename is provided as command line argument
    if len(sys.argv) > 1:
        input_filename = sys.argv[1]
    else:
        input_filename = input("Enter the path to your point cloud file: ")
    
    # Load points from file
    original_points = load_points_from_file(input_filename)
    
    # Step 1: Perform the first rotation to align overall PC1 with Z-axis
    first_rotation_points, angles1, pca1, rotation_matrices1 = rotate_pc1_to_z(original_points)
    print(f"First rotation angles: theta_z = {np.degrees(angles1[0]):.2f}°, theta_y = {np.degrees(angles1[1]):.2f}°")
    
    # Step 2: Extract middle slab from the rotated points
    slab_points, slab_indices, slab_min, slab_max = extract_middle_slab(first_rotation_points, fraction=1/3)
    
    # Visualize the slab selection
    visualize_points(first_rotation_points, slab_points, slab_indices, slab_min, slab_max, 
                     title="First Rotation with Middle Slab Highlighted")
    
    # Step 3: Perform PCA on the slab points
    slab_rotation_points, angles2, pca2, rotation_matrices2 = rotate_pc1_to_z(slab_points)
    print(f"Slab rotation angles: theta_z = {np.degrees(angles2[0]):.2f}°, theta_y = {np.degrees(angles2[1]):.2f}°")
    
    # Apply the slab-based rotation to all points
    rotation_z2, rotation_y2 = rotation_matrices2
    
    # Center points before rotation
    center = np.mean(first_rotation_points, axis=0)
    centered_points = first_rotation_points - center
    
    # Apply rotations
    intermediate_points = np.dot(centered_points, rotation_z2.T)
    final_centered_points = np.dot(intermediate_points, rotation_y2.T)
    
    # Add back the center
    final_points = final_centered_points + center
    
    # Visualize all steps
    visualize_rotation_steps(original_points, first_rotation_points, slab_points, final_points)
    
    # Save the final rotated points
    save_rotated_points(final_points, "final_rotated_points.txt")
    
    print("Processing complete!")

if __name__ == "__main__":
    main()