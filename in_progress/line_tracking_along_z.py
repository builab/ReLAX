import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.cm as cm
from scipy.interpolate import splprep, splev
from scipy.spatial import distance
from mpl_toolkits.mplot3d import Axes3D
from scipy import linalg
import matplotlib.colors as mcolors

""" 
Tracking line part
"""
def fit_line_to_points(points):
    """
    Fit a 3D line to points using least squares.
    Returns a point on the line and the direction vector.
    """
    # Calculate the mean point (centroid)
    centroid = np.mean(points, axis=0)
    
    # Shift points to have zero mean
    shifted_points = points - centroid
    
    # Create the covariance matrix
    cov_matrix = np.dot(shifted_points.T, shifted_points)
    
    # Find the eigenvectors and eigenvalues of the covariance matrix
    eigenvalues, eigenvectors = linalg.eigh(cov_matrix)
    
    # The direction of the line is the eigenvector with the largest eigenvalue
    direction = eigenvectors[:, np.argmax(eigenvalues)]
    
    # Normalize the direction vector
    direction = direction / np.linalg.norm(direction)
    
    return centroid, direction

def distance_point_to_line(point, line_point, line_direction):
    """Calculate the perpendicular distance from a point to a line."""
    # Vector from line point to the point
    vec = point - line_point
    
    # Project this vector onto the line direction
    projection = np.dot(vec, line_direction)
    
    # The closest point on the line
    closest_point = line_point + projection * line_direction
    
    # Return the distance
    return np.linalg.norm(point - closest_point)

def cluster_points_bidirectional(points, initial_indices, bin_size=50, distance_threshold=10):
    """
    Cluster points that are close to a line fitted to the data, growing in both Z directions.
    
    Args:
        points: Numpy array of shape (n, 3) containing the 3D points
        bin_size: Size of bins along Z-axis
        distance_threshold: Maximum distance for a point to be included in the cluster
        initial_indices: Indices of initial points to fit the line
        
    Returns:
        Indices of points in the cluster, distances to the fitted line
    """
    # Select initial points and fit a line
    initial_points = points[initial_indices]
    distances = np.zeros(len(initial_indices)).tolist()  # Convert to list for appending
    initial_line_point, initial_line_direction = fit_line_to_points(initial_points)
    
    # Sort points by Z-coordinate
    z_sorted_indices = np.argsort(points[:, 2])
    sorted_points = points[z_sorted_indices]
    
    # Find min and max Z values to create bins
    min_z = np.min(sorted_points[:, 2])
    max_z = np.max(sorted_points[:, 2])
    
    # Initialize cluster with initial points
    cluster_indices = list(initial_indices)
    
    # Find Z value of the center of initial points to split into increasing and decreasing directions
    initial_center_z = np.mean(initial_points[:, 2])
    
    # Create bins for increasing Z direction
    z_bins_increasing = np.arange(initial_center_z, max_z + bin_size, bin_size)
    
    # Create bins for decreasing Z direction
    z_bins_decreasing = np.arange(initial_center_z, min_z - bin_size, -bin_size)
    
    # Initialize line parameters for increasing and decreasing directions
    increasing_line_point, increasing_line_direction = initial_line_point.copy(), initial_line_direction.copy()
    decreasing_line_point, decreasing_line_direction = initial_line_point.copy(), initial_line_direction.copy()
    
    # Process bins in increasing Z direction
    print(f"Processing {len(z_bins_increasing)-1} bins in increasing Z direction")
    for i in range(len(z_bins_increasing) - 1):
        bin_start = z_bins_increasing[i]
        bin_end = z_bins_increasing[i+1]
        
        # Find points in this Z bin
        bin_mask = (points[:, 2] >= bin_start) & (points[:, 2] < bin_end)
        bin_points = points[bin_mask]
        bin_original_indices = np.where(bin_mask)[0]
        
        # Find points within distance threshold of the current line
        for j, (point, orig_idx) in enumerate(zip(bin_points, bin_original_indices)):
            distance = distance_point_to_line(point, increasing_line_point, increasing_line_direction)
            
            if distance <= distance_threshold and orig_idx not in cluster_indices:
                cluster_indices.append(orig_idx)
                distances.append(distance)  # Append the corresponding distance

        # Update the line fit using the most recent points in the cluster
        if cluster_indices:
            cluster_points = points[cluster_indices]
            z_sorted_cluster = cluster_points[np.argsort(cluster_points[:, 2])]
            
            # Use the last 5 points (highest Z values) to update the line fit
            fit_points = z_sorted_cluster[-5:] if len(z_sorted_cluster) >= 5 else z_sorted_cluster
            increasing_line_point, increasing_line_direction = fit_line_to_points(fit_points)
    
    # Process bins in decreasing Z direction
    print(f"Processing {len(z_bins_decreasing)-1} bins in decreasing Z direction")
    for i in range(len(z_bins_decreasing) - 1):
        bin_start = z_bins_decreasing[i]
        bin_end = z_bins_decreasing[i+1]
        
        # Find points in this Z bin
        bin_mask = (points[:, 2] <= bin_start) & (points[:, 2] > bin_end)
        bin_points = points[bin_mask]
        bin_original_indices = np.where(bin_mask)[0]
        
        # Find points within distance threshold of the current line
        for j, (point, orig_idx) in enumerate(zip(bin_points, bin_original_indices)):
            distance = distance_point_to_line(point, decreasing_line_point, decreasing_line_direction)
            
            if distance <= distance_threshold and orig_idx not in cluster_indices:
                cluster_indices.append(orig_idx)
                distances.append(distance)  # Append the corresponding distance

        # Update the line fit using the most recent points in the cluster
        if cluster_indices:
            cluster_points = points[cluster_indices]
            z_sorted_cluster = cluster_points[np.argsort(cluster_points[:, 2])]
            
            # Use the first 5 points (lowest Z values) to update the line fit
            fit_points = z_sorted_cluster[:5] if len(z_sorted_cluster) >= 5 else z_sorted_cluster
            decreasing_line_point, decreasing_line_direction = fit_line_to_points(fit_points)
    
    return cluster_indices, np.array(distances)  # Convert distances back to a NumPy array


def visualize_bidirectional_cluster(points, initial_indices, cluster_indices):
    """
    Create a 3D visualization of the original points, the clustered points, and the fitted lines in both directions.
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract clustered points
    clustered_points = points[cluster_indices]
    
    # Extract non-clustered points
    all_indices = set(range(len(points)))
    non_cluster_indices = list(all_indices - set(cluster_indices))
    non_clustered_points = points[non_cluster_indices]
    
    # Plot non-clustered points in light gray
    ax.scatter(non_clustered_points[:, 0], non_clustered_points[:, 1], non_clustered_points[:, 2], 
               color='lightgray', alpha=0.3, s=10, label='Non-clustered Points')
    
    # Plot clustered points in blue
    ax.scatter(clustered_points[:, 0], clustered_points[:, 1], clustered_points[:, 2], 
               color='blue', alpha=0.7, s=20, label='Clustered Points')
    
    # Plot initial points used for line fitting in red
    initial_points = points[initial_indices]
    ax.scatter(initial_points[:, 0], initial_points[:, 1], initial_points[:, 2], 
               color='red', s=50, label='Initial Points')
    
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Visualization of Bidirectional Point Cluster and Fitted Lines')
    
    # Add legend
    ax.legend()
    
    # Adjust view angle for better visibility
    ax.view_init(elev=20, azim=30)
    ax.set_box_aspect([1, 1, 1]) 
    
    plt.tight_layout()
    plt.savefig('bidirectional_cluster_visualization.png', dpi=300)
    plt.show()
    
    return fig

"""----------------------------
Clustering part
----------------------------"""

# Step 1: Read the file
def read_points(file_path):
    # Read space-separated XYZ coordinates
    points = np.loadtxt(file_path)
    return points

# Step 2: Select points in the middle 1/6 of Z
def select_middle_points(points):
    z_values = points[:, 2]
    z_min, z_max = np.min(z_values), np.max(z_values)
    z_range = z_max - z_min
    
    # Calculate the bounds for the middle 1/6
    z_middle_lower = z_min + (5/12) * z_range  # 5/12 is the start of the middle 1/6
    z_middle_upper = z_min + (7/12) * z_range  # 7/12 is the end of the middle 1/6
    
    # Create mask for middle points
    middle_mask = (z_values >= z_middle_lower) & (z_values <= z_middle_upper)
    middle_points = points[middle_mask]
    
    print(f"Total points: {len(points)}")
    print(f"Points in middle 1/6 of Z: {len(middle_points)}")
    print(f"Z range: [{z_min:.2f}, {z_max:.2f}]")
    print(f"Middle 1/6 Z range: [{z_middle_lower:.2f}, {z_middle_upper:.2f}]")
    
    return middle_points, middle_mask

# Step 3: Project points to X,Y plane (for clustering only)
def project_to_xy(points):
    return points[:, :2]

# Step 4: Find optimal number of clusters between 7-9
def find_optimal_clusters(points_xy, min_clusters=7, max_clusters=9):
    silhouette_scores = []
    
    for n_clusters in range(min_clusters, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(points_xy)
        silhouette_avg = silhouette_score(points_xy, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        print(f"For n_clusters = {n_clusters}, the silhouette score is {silhouette_avg:.3f}")
    
    optimal_clusters = range(min_clusters, max_clusters + 1)[np.argmax(silhouette_scores)]
    print(f"Optimal number of clusters: {optimal_clusters}")
    
    return optimal_clusters

# Step 5: Perform clustering with optimal number of clusters
def perform_clustering(points_xy, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(points_xy)
    cluster_centers = kmeans.cluster_centers_
    
    return cluster_labels, cluster_centers

# New Step 6: Sort points by Z for spline fitting
def sort_points_by_z(points):
    """Sort points by Z coordinate for spline fitting"""
    z_sorted_indices = np.argsort(points[:, 2])
    return points[z_sorted_indices]

# Step 7: Fit 3D spline curves to each cluster using Z-sorted points
def fit_3d_splines(points_3d, cluster_labels, n_clusters):
    splines = []
    
    for i in range(n_clusters):
        # Get 3D points in this cluster
        cluster_points = points_3d[cluster_labels == i]
        
        # Need at least 4 points for cubic spline
        if len(cluster_points) < 4:
            print(f"Cluster {i+1} has {len(cluster_points)} points - too few for spline fitting")
            splines.append(None)
            continue
            
        # Sort points by Z coordinate
        ordered_points = sort_points_by_z(cluster_points)
        
        try:
            # Fit parametric spline in 3D
            tck, u = splprep([ordered_points[:, 0], ordered_points[:, 1], ordered_points[:, 2]], s=0.0, k=3)
            
            # Generate points along spline
            u_new = np.linspace(0, 1, 100)
            x_new, y_new, z_new = splev(u_new, tck)
            
            splines.append((x_new, y_new, z_new))
            print(f"Fitted 3D spline to cluster {i+1} with {len(cluster_points)} points (Z-sorted)")
        except Exception as e:
            print(f"Error fitting spline to cluster {i+1}: {e}")
            splines.append(None)
    
    return splines

# Step 8: Visualize 3D points and splines
def visualize_3d_clusters_with_splines(all_points, middle_mask, middle_points, cluster_labels, splines):
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot all points (not in the middle) in light gray
    ax.scatter(
        all_points[~middle_mask, 0], 
        all_points[~middle_mask, 1], 
        all_points[~middle_mask, 2],
        color='lightgray', alpha=0.3, s=10, label='Other Points'
    )
    
    # Create a colormap for clusters
    unique_labels = np.unique(cluster_labels)
    colors = cm.nipy_spectral(np.linspace(0, 1, len(unique_labels)))
    
    # Plot clustered points from middle section with colors
    for i, color in zip(unique_labels, colors):
        cluster_points = middle_points[cluster_labels == i]
        ax.scatter(
            cluster_points[:, 0], 
            cluster_points[:, 1], 
            cluster_points[:, 2],
            color=color, alpha=0.7, s=40, 
            label=f'Cluster {i+1} ({len(cluster_points)} points)'
        )
    
    # Plot 3D splines with increased line width for visibility
    for i, spline in enumerate(splines):
        if spline is not None:
            x_new, y_new, z_new = spline
            ax.plot(x_new, y_new, z_new, color=colors[i], linewidth=4, label=f'Spline {i+1}')
    
    # Set labels and title
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    ax.set_title('3D Visualization with Z-Sorted Splines', fontsize=16)
    
    # Add legend with smaller font and outside plot
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=8)
    
    plt.tight_layout()
    plt.savefig('3d_z_sorted_splines.png', dpi=300)
    plt.show()
    
    # Also create a top-down (XY) view
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111)
    
    # Plot all XY points in light gray
    ax.scatter(
        all_points[~middle_mask, 0], 
        all_points[~middle_mask, 1],
        color='lightgray', alpha=0.3, s=10, label='Other Points'
    )
    
    # Plot clustered points from middle section with colors
    for i, color in zip(unique_labels, colors):
        cluster_points = middle_points[cluster_labels == i]
        ax.scatter(
            cluster_points[:, 0], 
            cluster_points[:, 1],
            color=color, alpha=0.7, s=40, 
            label=f'Cluster {i+1} ({len(cluster_points)} points)'
        )
    
    # Plot XY projection of splines
    for i, spline in enumerate(splines):
        if spline is not None:
            x_new, y_new, _ = spline
            ax.plot(x_new, y_new, color=colors[i], linewidth=4)
    
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_title('Top-Down (XY) View with Z-Sorted Splines', fontsize=16)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=8)
    
    plt.tight_layout()
    plt.savefig('xy_z_sorted_splines.png', dpi=300)
    plt.show()

def old_main(all_points):
    middle_points, middle_mask = select_middle_points(all_points)
    points_xy = project_to_xy(middle_points)
    
    print(all_points)
    # Find optimal number of clusters and perform clustering
    optimal_clusters = find_optimal_clusters(points_xy)
    cluster_labels, cluster_centers = perform_clustering(points_xy, optimal_clusters)
    
    # Perhaps, make sure this cluster is relatively smooth with 2d polynominal?
    
    # NOT USE
    # Fit 3D splines to each cluster using Z-sorted points
    #splines = fit_3d_splines(middle_points, cluster_labels, optimal_clusters)
    
    # Visualize the results
    #visualize_3d_clusters_with_splines(all_points, middle_mask, middle_points, cluster_labels, splines)
    
    # Return additional cluster statistics
    #cluster_stats = {}
    #for i in range(optimal_clusters):
    #    cluster_points = middle_points[cluster_labels == i]
    #    z_min = np.min(cluster_points[:, 2])
    #    z_max = np.max(cluster_points[:, 2])
    #    z_range = z_max - z_min
        
    #    cluster_stats[f'Cluster {i+1}'] = {
    #        'points_count': len(cluster_points),
    #        'percentage': len(cluster_points) / len(middle_points) * 100,
    #        'centroid_3d': np.mean(cluster_points, axis=0).tolist(),
    #        'z_range': [float(z_min), float(z_max)],
    #        'z_span': float(z_range),
    #        'has_spline': splines[i] is not None
    #    }
    
    return cluster_labels, cluster_centers
    
def find_cluster_seeds(all_points):
    middle_points, middle_mask = select_middle_points(all_points)
    points_xy = project_to_xy(middle_points)
    
    # Find optimal number of clusters and perform clustering
    optimal_clusters = find_optimal_clusters(points_xy)
    cluster_labels, cluster_centers = perform_clustering(points_xy, optimal_clusters)

    # Convert middle_mask (boolean mask) to original indices in all_points
    all_indices = np.where(middle_mask)[0]

    # Create list of lists for cluster indices
    clustered_indices = [[] for _ in range(optimal_clusters)]
    for i, label in enumerate(cluster_labels):
        clustered_indices[label].append(all_indices[i])

    return clustered_indices  # List of lists: clustered_indices[cluster_id] contains indices from all_points



if __name__ == "__main__":
    # Run the analysis
    bin_size=50
    distance_threshold=10
    
    points = read_points('final_rotated_points.txt')
    clustered_indices = find_cluster_seeds(points)

    print("Now tracking line")
    # Create an array to store cluster assignments for all points
    # Initialize with -1 (no cluster) for all points
    all_points_cluster_ids = np.full(len(points), -1, dtype=int)
    all_points_distances = np.full(len(points), float('inf'))  # Initialize with infinity

    for cluster_id, indices in enumerate(clustered_indices):
        # Set distance to 0 for initial seed points of this cluster
        for idx in indices:
            all_points_cluster_ids[idx] = cluster_id
            all_points_distances[idx] = 0  # Set distance to 0 for seed points
    
        print(f"Cluster {cluster_id}: {indices}")
        # Cluster points bidirectionally
        new_cluster_indices, distances = cluster_points_bidirectional(points, indices, bin_size, distance_threshold)
    
        # Visualize the results
        fig = visualize_bidirectional_cluster(points, indices, new_cluster_indices)
        print("Created visualization and saved as 'bidirectional_cluster_visualization.png'")
        
        # Assign cluster_id to all points in this cluster
        for i, idx in enumerate(new_cluster_indices):
            # Skip seed points as they've already been assigned with distance 0
            if idx not in indices:
                distance = distances[i]
                # If this point hasn't been assigned yet, or if this assignment has a smaller distance
                if all_points_cluster_ids[idx] == -1 or distance < all_points_distances[idx]:
                    all_points_cluster_ids[idx] = cluster_id
                    all_points_distances[idx] = distance

    
        clustered_points = points[new_cluster_indices]
        print(f"Found {len(clustered_points)} points in the bidirectional cluster")

        # Save clustered points
        with open(f"bidirectional_clustered_points_group{cluster_id}.txt", 'w') as f:
            for point in clustered_points:
                f.write(f"{point[0]} {point[1]} {point[2]}\n")
        print(f"Saved clustered points to bidirectional_clustered_points_{cluster_id}.txt")

    # Save the cluster assignments and distances together
    with open("point_cluster_assignments_with_distances.txt", 'w') as f:
        for i in range(len(points)):
            cluster_id = all_points_cluster_ids[i]
            distance = all_points_distances[i] if cluster_id != -1 else -1
            f.write(f"{i} {cluster_id} {distance}\n")

    # Print statistics about cluster assignments
    print(f"Total points: {len(points)}")
    print(f"Points assigned to clusters: {np.sum(all_points_cluster_ids >= 0)}")
    print(f"Points not assigned to any cluster: {np.sum(all_points_cluster_ids == -1)}")
    
