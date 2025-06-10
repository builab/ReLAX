#!/usr/bin/env python3
# Script to analyse CP distance from a star file for Current Biology paper, 2025
# Authors: HB, 05/2025
# Script assume helicalTubeID==1 is C1 but the star file is not correctly label
# Something not right in term of X, Y and Z shift
# Fix the angle switching

import argparse
import numpy as np
import pandas as pd
import starfile
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.transform import Rotation as R
from scipy.stats import gaussian_kde  # For KDE plot

def correct_angles_for_linearity(angles):
    """
    Corrects a sequence of angles (in degrees) to minimize discontinuities
    when plotting for linearity, assuming the angles are expected to
    form a relatively continuous progression.

    Args:
        angles (numpy.ndarray): A 1D NumPy array of angles in degrees,
                                 typically in the range of -180 to 180.

    Returns:
        numpy.ndarray: A 1D NumPy array of corrected angles.
    """
    corrected_angles = angles.copy()  # Create a copy to avoid modifying the original array
    for i in range(1, len(corrected_angles)):
        diff = corrected_angles[i] - corrected_angles[i - 1]
        if diff > 180:
            corrected_angles[i:] -= 360
        elif diff < -180:
            corrected_angles[i:] += 360
    return corrected_angles

def pts2csv(origins1, origins2, out_csv):
    """
    Write csv for checking
    """
    # Create DataFrames
    origins1_df = pd.DataFrame(origins1, columns=['X1', 'Y1', 'Z1'])
    origins2_df = pd.DataFrame(origins2, columns=['X2', 'Y2', 'Z2'])

    # Determine the number of rows in each DataFrame
    combined_df = pd.concat([origins1_df, origins2_df], axis=1)

    # Write the combined DataFrame to a single CSV file
    combined_df.to_csv(out_csv, index=False, float_format='%.2f')
    print(f"Written {out_csv}")

def find_nearest_neighbours(origins1, origins2):
    """
    Find the nearest neighbours of points in vector 1 in vector 2
    """
    nn = NearestNeighbors(n_neighbors=1, algorithm='auto')

    # Fit the model to the origins2 data (the data we're searching within)
    nn.fit(origins2)

    # Find the nearest neighbors in origins2 for each point in origins1
    distances, indices = nn.kneighbors(origins1)

    # 'indices' will contain the indices of the nearest neighbors in origins2
    # 'distances' will contain the distances to those nearest neighbors

    # Create the origins1_nearest array by selecting the nearest points from origins2
    origins1_nearest = origins2[indices.flatten()]
    
    return origins1_nearest

# main
def main(args):
    """
    Main function to calculate and optionally plot distance modulo,
    taking parameters from the command line.
    """
    do_plot = args.do_plot
    angpix = args.angpix
    tomo_angpix = args.tomo_angpix
    distance_period = args.dist_period  # Periodicity to calculate the modulo
    distance_threshold = args.dist_threshold  # Make it large
    input_star_file = args.input

    print(f"Parameters:")
    print(f"  do_plot: {do_plot}")
    print(f"  angpix: {angpix}")
    print(f"  tomo_angpix: {tomo_angpix}")
    print(f"  distance_period: {distance_period}")
    print(f"  distance_threshold: {distance_threshold}")
    print(f"  input_star_file: {input_star_file}")
    
    star = starfile.read(input_star_file, always_dict=True)
    print(f"{input_star_file} read")
    
    if not all(key in star for key in ('particles', 'optics')):
        print("expected RELION 3.1+ style STAR file containing particles and optics blocks")

    if 'optics' in star:
        df = star['particles'].merge(star['optics'], on='rlnOpticsGroup')
        print("optics table merged")
    else:
        df = star['particles'].copy()
        print("no optics table")

    # Total all diff
    total_origins_diff = []

    for tomo_id, tomo_group in df.groupby('rlnTomoName', sort=True):
        if tomo_id.endswith(".tomostar"):
            tomo_name = tomo_id[:-len(".tomostar")]
        else:
            tomo_name = tomo_id
                
        group1 = tomo_group[tomo_group['rlnHelicalTubeID'] == 1]
        group2 = tomo_group[tomo_group['rlnHelicalTubeID'] == 2]
        xyz1 = group1[['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']].to_numpy()
        shift1 = group1[['rlnOriginXAngst', 'rlnOriginYAngst', 'rlnOriginZAngst']].to_numpy()
        euler_angles = group1[['rlnAngleRot', 'rlnAngleTilt', 'rlnAnglePsi']].to_numpy()
        xyz2 = group2[['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']].to_numpy()
        shift2 = group2[['rlnOriginXAngst', 'rlnOriginYAngst', 'rlnOriginZAngst']].to_numpy()

        origins1 = xyz1 * angpix - shift1
        origins2 = xyz2 * angpix - shift2
        #print(origins1)
        origins1_nearest = find_nearest_neighbours(origins1, origins2)
        print("Finding nearest neighbours of C1")
        rotation_matrices = R.from_euler(angles=euler_angles, seq='ZYZ', degrees=True).inv().as_matrix()
        #print(rotation_matrices)
        print("calculated rotation matrices from euler angles")
        # Reshape origins1 to (num_points, 3, 1) to allow for batched matrix multiplication
        origins1_reshaped = origins1[:, :, np.newaxis]
        origins1_nearest_reshaped = origins1_nearest[:, :, np.newaxis]
        
        pts2csv(origins1, origins1_nearest, f'{tomo_name}_original_pts.csv')
      
        # Perform batched matrix multiplication
        origins1_diff = origins1_nearest - origins1
        origins1_diff_reshaped = origins1_diff[:, :, np.newaxis]
        
        xform_origins1_diff = rotation_matrices @ origins1_diff_reshaped
        # The result will be (num_points, 3, 1). Remove the last dimension to get (num_points, 3)
        xform_origins1_diff = xform_origins1_diff.squeeze(axis=-1)
        
        # Filter rows based on the Z-component of the transformed difference
        kept_indices = np.where(np.abs(xform_origins1_diff[:, 2]) <= distance_threshold)[0]
        filtered_xform_origins1_diff = xform_origins1_diff[kept_indices]
        filtered_origins1 = origins1[kept_indices]
        filtered_origins1_nearest = origins1_nearest[kept_indices]


        print(f"Calculated transformed points for {tomo_name} (filtered by transformed Z)")
        pts2csv(origins1_diff[np.abs(xform_origins1_diff[:, 2]) <= distance_threshold], filtered_xform_origins1_diff, f'{tomo_name}_xform_filtered_z_pts.csv')

        # Append the filtered transformed origin differences to the total list
        total_origins_diff.extend(filtered_xform_origins1_diff)
        

        if do_plot:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

            # Plot origins1 as a line (red)
            ax.plot(origins1[:, 0], origins1[:, 1], origins1[:, 2], c='r', marker='o', linestyle='-', label='Origins1')

		    # Plot origins2 as a line (blue)
            ax.plot(origins2[:, 0], origins2[:, 1], origins2[:, 2], c='b', marker='x', linestyle='--', label='Origins2')
        
            # Plot connecting lines
            for i in range(filtered_origins1.shape[0]):
                x_coords = [filtered_origins1[i, 0], filtered_origins1_nearest[i, 0]]
                y_coords = [filtered_origins1[i, 1], filtered_origins1_nearest[i, 1]]
                z_coords = [filtered_origins1[i, 2], filtered_origins1_nearest[i, 2]]
                ax.plot(x_coords, y_coords, z_coords, c='gray', linestyle='--', linewidth=0.5)


            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title('3D Lines of Origins1 and Origins2')
            ax.legend()
            #plt.show()
            
            
            plt.savefig(f"{tomo_name}.png")
            plt.close(fig)
            
            # Plot the 2nd plot for distribution
            # Subplot 1: X vs Y with color based on index
            fig, axes = plt.subplots(1, 3, figsize=(21, 6)) # Changed to 1 row, 3 columns

            x_diff = filtered_xform_origins1_diff[:, 0]
            y_diff = filtered_xform_origins1_diff[:, 1]
            z_diff = filtered_xform_origins1_diff[:, 2] % distance_period
            euler_angle_rot_all = euler_angles[:, 0] # Get all Euler angle rot values

            # Subplot 1: X vs Y
            # Get the indices of the points
            indices = np.arange(len(x_diff))

            # Use a colormap to map indices to colors
            cmap = plt.cm.viridis  # You can choose other colormaps like 'plasma', 'magma', 'cividis', etc.
            norm = plt.Normalize(indices.min(), indices.max())  # Normalize indices to the range [0, 1] for the colormap


            scatter = axes[0].scatter(x_diff, y_diff, s=20, alpha=0.6, c=indices, cmap=cmap, norm=norm)
            axes[0].set_xlabel('Transformed Difference in X')
            axes[0].set_ylabel('Transformed Difference in Y')
            axes[0].set_title('XY Distribution of Transformed Origin Differences')
            axes[0].grid(True)
            
            # Set the x and y axis limits for the first subplot
            axes[0].set_xlim([-500, 200])
            axes[0].set_ylim([-350, 350])
            
            # Add a colorbar to show the mapping of indices to colors
            cbar = fig.colorbar(scatter, ax=axes[0], label='Point Index')
            
            # Subplot 2: Distribution of Z
            axes[1].hist(z_diff, bins=30, density=True, alpha=0.7, color='skyblue')
            kde = gaussian_kde(z_diff)
            x_eval = np.linspace(min(z_diff), max(z_diff), 200)
            axes[1].plot(x_eval, kde(x_eval), color='red', label='KDE')
            axes[1].set_xlabel('Transformed Difference in Z')
            axes[1].set_ylabel('Density')
            axes[1].set_title('Distribution of Transformed Z Differences')
            axes[1].legend()
            
            # Subplot 3: Euler Angle 1 vs Index of Kept Points
            indices_all = np.arange(len(euler_angles))

            axes[2].plot(indices_all, correct_angles_for_linearity(euler_angle_rot_all), marker='o', linestyle='-', markersize=4)
            axes[2].set_xlabel('Index of Points')
            axes[2].set_ylabel('Euler Angle Rot (degrees)')
            axes[2].set_title('Euler Angle Rot vs. Index')
            axes[2].grid(True)
            
            plt.tight_layout()
            
            plt.savefig(f"{tomo_name}_shifts.png")
            plt.close(fig)
   

    # Convert the list of NumPy arrays into a single NumPy array
    total_origins_diff = np.array(total_origins_diff)
    
    if total_origins_diff.size > 0 and total_origins_diff.shape[1] == 3:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        x_diff = total_origins_diff[:, 0]
        y_diff = total_origins_diff[:, 1]
        z_diff = total_origins_diff[:, 2] % distance_period

        # Subplot 1: X vs Y
        axes[0].scatter(x_diff, y_diff, s=20, alpha=0.6)
        axes[0].set_xlabel('Transformed Difference in X')
        axes[0].set_ylabel('Transformed Difference in Y')
        axes[0].set_title('XY Distribution of Transformed Origin Differences')
        axes[0].grid(True)
        
        # Set the x and y axis limits for the first subplot
        axes[0].set_xlim([-500, 200])
        axes[0].set_ylim([-350, 350])

        # Subplot 2: Distribution of Z
        axes[1].hist(z_diff, bins=30, density=True, alpha=0.7, color='skyblue')
        kde = gaussian_kde(z_diff)
        x_eval = np.linspace(min(z_diff), max(z_diff), 200)
        axes[1].plot(x_eval, kde(x_eval), color='red', label='KDE')
        axes[1].set_xlabel('Transformed Difference in Z')
        axes[1].set_ylabel('Density')
        axes[1].set_title('Distribution of Transformed Z Differences')
        axes[1].legend()
        plt.savefig(input_star_file.replace('.star', '.eps'))
        plt.tight_layout()
        plt.show()
    else:
        print("\nCannot plot total_origins_diff. Ensure it has data and 3 columns (x, y, z).")

    # You can still save this total_origins_diff to a CSV file if needed
    total_origins_diff_df = pd.DataFrame(total_origins_diff, columns=['diff_x_transformed', 'diff_y_transformed', 'diff_z_transformed'])
    total_origins_diff_df.to_csv('total_xform_origins_diff.csv', index=False)
    print("\nTotal transformed origin differences saved to 'total_xform_origins_diff.csv'")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate distance modulo and optionally plot.")
    parser.add_argument("--do_plot", action="store_true", default=True, help="Enable plotting.")
    parser.add_argument("--angpix", type=float, default=14.00, help="Pixel size (angstroms).")
    parser.add_argument("--tomo_angpix", type=float, default=3.37, help="Tomogram pixel size (angstroms).")
    parser.add_argument("--dist_period", type=float, default=83.2, help="Periodicity for modulo calculation.")
    parser.add_argument("--dist_threshold", type=float, default=500.0, help="Distance threshold.")
    parser.add_argument("--input", type=str, required=True, help="Input STAR file.")

    args = parser.parse_args()
    main(args)
