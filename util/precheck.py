"""
Precheck package to fix IMOD model for axoneme
Written by Jerry Gao, McGill
"""

import os, glob, sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from util.io import run_model2point, run_point2model
import shutil

def conv2bool(input):
    if input == "False" or input == "false" or not input:
        return False
    return True

def make_dir(dir_path, clear_dir=False):
    if clear_dir and os.path.exists(dir_path):
        for item in glob.glob(os.path.join(dir_path, "*")):
            shutil.rmtree(item) if os.path.isdir(item) else os.unlink(item)
    os.makedirs(dir_path, exist_ok=True)
    
def densify_contour(contour, num_points_per_segment=10):
    """
    Interpolate additional points along each segment of the contour.
    contour: an array of shape (N, 3)
    Returns an array of shape (M, 3) with more points.
    """
    densified = []
    for i in range(len(contour) - 1):
        # Interpolate num_points_per_segment points between contour[i] and contour[i+1]
        interp = np.linspace(contour[i], contour[i+1], num=num_points_per_segment, endpoint=False)
        densified.append(interp)
    # Append the final point
    densified.append(contour[-1:])
    return np.concatenate(densified, axis=0)

def compute_overlap_percentage_dense(contour1, contour2, threshold, num_points_per_segment=10):
    """
    Densify contour1, then for each densified point check if it's within the
    threshold of any point on densified contour2. Estimate the fraction of contour1's
    length that is "overlapping" based on these samples.
    
    Parameters:
      contour1, contour2: arrays of shape (N, 3) and (M, 3)
      threshold: distance threshold (in the same units as your coordinates)
      num_points_per_segment: how many points to interpolate between each pair of original points
    
    Returns:
      overlap_percentage: fraction of contour1's length that is overlapping
    """
    # Densify both contours
    dense1 = densify_contour(contour1, num_points_per_segment)
    dense2 = densify_contour(contour2, num_points_per_segment)
    
    # Build a KDTree for dense2 for fast nearest-neighbor queries
    tree = cKDTree(dense2)
    distances, _ = tree.query(dense1)
    is_overlap = distances < threshold
    
    # Compute cumulative arc-length along dense1
    seg_lengths = np.linalg.norm(np.diff(dense1, axis=0), axis=1)
    cumulative = np.concatenate(([0], np.cumsum(seg_lengths)))
    
    total_length = cumulative[-1]
    overlap_length = 0.0
    
    # For each segment between consecutive densified points,
    # estimate its contribution to the overlapping length.
    for i in range(len(is_overlap) - 1):
        seg_len = cumulative[i+1] - cumulative[i]
        # If both endpoints are overlapping, count the full segment
        if is_overlap[i] and is_overlap[i+1]:
            overlap_length += seg_len
        # If only one endpoint is overlapping, count half the segment length
        elif is_overlap[i] or is_overlap[i+1]:
            overlap_length += 0.5 * seg_len
    
    return overlap_length / total_length if total_length > 0 else 0.0

def get_objname(file, stringToRemove):
    file = file.replace("_modified", "")
    objname = os.path.basename(file.removesuffix(stringToRemove))
    return objname

def run_model_check(modDir, prjPath, modFileDelimiter, stringToBeRemoved, angpix, min_separation, min_angle, min_len, overlap_radius, overlap_threshold,
         tomo_x_size, tomo_y_size, tomo_z_size,
         skip_graph, print_length, print_overlap, print_angle, 
         delete_duplicate, delete_overlap,
         absolute_graph):
    
    mod_files = sorted([os.path.basename(f) for f in glob.glob(os.path.join(modDir, modFileDelimiter))])

    if not mod_files:
        print ("No model files found!")
        sys.exit()

    txt_files = []
    length_warnings = []
    overlap_warnings = []
    duplicate_warnings = []
    angle_warnings = []

    chkDir = f"{prjPath}/precheck"
    txtDir = f"{prjPath}/precheck/model_txt"
    pngDir = f"{prjPath}/precheck/graphs"
    newTxtDir = f"{prjPath}/precheck/new_txt"
    newModDir = f"{prjPath}/precheck/new_mod"
    make_dir(chkDir)
    make_dir(txtDir, True)
    make_dir(pngDir, not skip_graph)
    make_dir(newTxtDir, True)
    make_dir(newModDir, True)
    
    for file in mod_files:
        basename = os.path.splitext(file)[0]
        run_model2point(f"{modDir}/{basename}.mod", f"{txtDir}/{basename}.txt")

    txt_files = sorted(glob.glob(os.path.join(txtDir, "*.txt")))
    
    # Visualizes and saves plotted points as graph
    if not skip_graph:
        for txt_file in txt_files:
            print (f"Creating graph for {txt_file}")
            data = np.loadtxt(txt_file)
            contours = np.unique(data[:, 1])
            
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')
            
            for c in contours:
                indices = data[:, 1] == c
                x = data[indices, 2]
                y = data[indices, 3]
                z = data[indices, 4]
                ax.plot(x, y, z, marker='o', markersize=2, linewidth=0.5, label=f"{int(c)}")

            if absolute_graph:
                ax.set_xlim([0, tomo_x_size])
                ax.set_ylim([0, tomo_y_size])
                ax.set_zlim([0, tomo_z_size])
                ax.set_box_aspect((tomo_x_size, tomo_y_size, tomo_z_size))
            
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")

            ax.tick_params(labelsize=8)
            ax.legend(fontsize=6, loc="upper right")
            plt.tight_layout()

            base_name = os.path.splitext(os.path.basename(txt_file))[0]
            png_file = os.path.join(pngDir, base_name + ".png")
            plt.savefig(png_file)
            plt.close(fig)

    #### Calculates prechecks ####
    nm_pix = round(float(angpix)/10,3)
    txtStringToBeRemoved = os.path.splitext(stringToBeRemoved)[0] + ".txt"

    ## Removes duplicate points ##
    print("-------------------------------------------------------------------------------")
    print(f"\nCHECKING FOR DUPLICATE POINTS with minimum separation of {min_separation}nm.")
    print(f"Delete duplicates: {delete_duplicate}")

    ang_separation = min_separation/nm_pix
    for t, txt_file in enumerate(txt_files):
        data = np.loadtxt(txt_file)
        filename = get_objname(txt_file, txtStringToBeRemoved)
        duplicate_rows = set()
        contours = np.unique(data[:, 1])
        for c in contours:
            # Get indices for rows corresponding to contour c
            idx = np.where(data[:, 1] == c)[0]
            points = data[idx, 2:5]
            if points.shape[0] < 2:
                continue
            tree = cKDTree(points)
            # Reset the local counter for each contour
            for local_i, pt in enumerate(points):
                neighbor_local_indices = tree.query_ball_point(pt, r=ang_separation)
                for local_j in neighbor_local_indices:
                    # Only mark duplicate if the neighbor has a higher local index than the current point
                    if local_j > local_i:
                        dist = np.linalg.norm(pt - points[local_j]) * nm_pix
                        w = f"Distance between {filename}_{int(c)} point {local_i+1} and point {local_j+1} is {dist:.2f}nm."
                        print(f"WARNING: {w}")
                        duplicate_warnings.append(w)
                        if delete_duplicate:
                            print(f"Deleting point {local_j+1} from {filename}_{int(c)}")
                            duplicate_rows.add(idx[local_j])
        
        if delete_duplicate: # Remove duplicate rows from the data array
            #print(duplicate_rows)
            if duplicate_rows:
                new_data = np.delete(data, list(duplicate_rows), axis=0)
                new_file = os.path.join(newTxtDir, os.path.splitext(os.path.basename(txt_file))[0] + "_modified" + os.path.splitext(txt_file)[1])
                fmt = ("%d", "%d", "%.2f", "%.2f", "%.2f")
                np.savetxt(new_file, new_data, fmt=fmt)
                txt_files[t] = new_file # Update txt_files to point to the new cleaned file

    if not duplicate_warnings:
        print("No duplicate points detected.")

    ## Checks for sharp angles ##
    print("-------------------------------------------------------------------------------")    
    print(f"\nCHECKING FOR SHARP TURN ANGLE with minimum angle threshold of {min_angle}°.")

    for txt_file in txt_files:
        data = np.loadtxt(txt_file)
        filename = get_objname(txt_file, txtStringToBeRemoved)
        contours = np.unique(data[:, 1])
        for c in contours:
            points = data[data[:, 1] == c, 2:5]
            if points.shape[0] < 3:
                continue
            # Iterate over three consecutive points
            for i in range(1, len(points) - 1):
                # Vectors from the middle point to its neighbors
                v1 = points[i] - points[i - 1]
                v2 = points[i + 1] - points[i]
                norm1 = np.linalg.norm(v1)
                norm2 = np.linalg.norm(v2)
                if norm1 == 0 or norm2 == 0:
                    continue  # Avoid division by zero
                # Compute the angle (in degrees) at the middle point
                cos_angle = np.dot(v1, v2) / (norm1 * norm2)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle_deg = abs(180-np.degrees(np.arccos(cos_angle)))

                if print_angle:
                    print (f"{filename}_{int(c)} point {i+1} has angle {angle_deg:.2f}°")

                if angle_deg < min_angle:
                    w = f"{filename}_{int(c)} point {i+1} has angle {angle_deg:.2f}°"
                    print(f"WARNING: {w}")
                    angle_warnings.append(w)

    if not angle_warnings:
        print("No aberrant angles detected.")

    ## Determines if any lengths are too short ##
    print(f"\nEvaluating length at {angpix}Apx ({nm_pix}nm/px) with cutoff {min_len}nm.")

    for txt_file in txt_files:
        data = np.loadtxt(txt_file)
        contours = np.unique(data[:, 1])
        filename = get_objname(txt_file, txtStringToBeRemoved)
        
        for c in contours:
            points = data[data[:, 1] == c, 2:5]
            
            # Calculate the distances between consecutive points
            diffs = np.diff(points, axis=0)
            seg_lengths = np.sqrt((diffs**2).sum(axis=1))
            contour_length = seg_lengths.sum()*nm_pix

            if print_length:
                print (f"Length of {filename}_{int(c)} is {contour_length:.2f}nm")

            if contour_length < min_len:
                w = f"Length of {filename}_{int(c)} is {contour_length:.2f}nm"
                print (f"WARNING: {w}")
                length_warnings.append(w)

    if not length_warnings:
        print(f"Lengths nominal.")

    ## Determines if any contours overlap ##
    print(f"\nEvaluating overlap with radius {overlap_radius}nm and threshold {overlap_threshold}%.")
    print(f"Delete overlaps: {delete_overlap}")
    
    for t, txt_file in enumerate(txt_files):
        data = np.loadtxt(txt_file)
        contours = np.unique(data[:, 1])
        filename = get_objname(txt_file, txtStringToBeRemoved)
        
        # Build a dictionary mapping contour number to its (N,3) points.
        contour_dict = {}
        for c in contours:
            contour_dict[int(c)] = data[data[:, 1] == c, 2:5]
        
        contour_keys = sorted(contour_dict.keys())

        # Compute the length for each contour.
        contour_lengths = {}
        for c in contour_keys:
            pts = contour_dict[c]
            if pts.shape[0] < 2:
                length = 0.0
            else:
                diffs = np.diff(pts, axis=0)
                seg_lengths = np.linalg.norm(diffs, axis=1)
                length = seg_lengths.sum() * nm_pix
            contour_lengths[c] = length

        # Initialize union-find structure.
        parent = {c: c for c in contour_keys}
        def find(x):
            while parent[x] != x:
                x = parent[x]
            return x
        def union(x, y):
            root_x = find(x)
            root_y = find(y)
            if root_x != root_y:
                parent[root_y] = root_x

        # Check overlaps between pairs of contours and union if they overlap.
        for i in range(len(contour_keys)):
            for j in range(i+1, len(contour_keys)):
                c1 = contour_dict[contour_keys[i]]
                c2 = contour_dict[contour_keys[j]]
                
                # Compute the overlap percentage for contour1 relative to contour2.
                perc_overlap = compute_overlap_percentage_dense(c1, c2, overlap_radius/nm_pix)

                if print_overlap:
                    print(f"{filename} contours {contour_keys[i]} and {contour_keys[j]} has overlap of {perc_overlap*100:.2f}%")

                if perc_overlap > overlap_threshold/100:
                    union(contour_keys[i], contour_keys[j])
                    w = f"Overlap of {filename}_{contour_keys[i]} and {filename}_{contour_keys[j]} is {perc_overlap*100:.2f}%"
                    print(f"WARNING: {w}")
                    overlap_warnings.append(w)

        # Group contours by their union-find root.
        groups = {}
        for c in contour_keys:
            root = find(c)
            groups.setdefault(root, []).append(c)

        # In each group, keep only the longest contour.
        deletion_set = set()
        for group in groups.values():
            if len(group) > 1:
                best = max(group, key=lambda x: contour_lengths[x])
                for c in group:
                    if c != best:
                        deletion_set.add(c)
                        w = f"Deleting contour {filename}_{c} (length: {contour_lengths[c]:.2f}nm), preserving {filename}_{best} (length: {contour_lengths[best]:.2f}nm)"
                        overlap_warnings.append(w)

        # Remove points belonging to contours marked for deletion.
        if delete_overlap:
            if deletion_set:
                new_data = np.array([row for row in data if int(row[1]) not in deletion_set])
                filename, ext = os.path.splitext(os.path.basename(txt_file))
                if not filename.endswith("_modified"):
                    filename += "_modified"
                new_file = os.path.join(newTxtDir, filename + ext)
                fmt = ("%d", "%d", "%.2f", "%.2f", "%.2f")
                np.savetxt(new_file, new_data, fmt=fmt)
                txt_files[t] = new_file
            else:
                print(f"No overlaps detected in {filename}.")

    if not overlap_warnings:
        print(f"No overlaps detected.")

    #### Cleans up modified files ####
    new_files = sorted([os.path.basename(f) for f in glob.glob(os.path.join(newTxtDir, "*_modified.txt"))])

    if new_files:
        print("\nProcesseing modified files.")
        for file in new_files: 

            # Re-numbers contours if any have been deleted
            input_filename = os.path.join(newTxtDir, file)
            with open(input_filename, 'r') as f:
                lines = f.readlines()
            data = []
            unique_contours = {}
            new_contour_id = 1
            for line in lines:
                parts = line.split()
                if len(parts) < 5:
                    continue  # Skip malformed lines
                key = int(parts[1])  # Contour number
                if key not in unique_contours:
                    unique_contours[key] = new_contour_id
                    new_contour_id += 1
                parts[1] = str(unique_contours[key])  # Update contour number
                data.append(" ".join(parts))
            with open(input_filename, 'w') as f:
                f.write("\n".join(data) + "\n")
            
            # Converts txt to mod
            basename = os.path.splitext(file)[0]
            run_point2model(f"{newTxtDir}/{basename}.txt", f"{newModDir}/{basename}.mod")
        print(f"Modified .mod files saved to {newModDir}")

    # print(warnings)
    if length_warnings or overlap_warnings:
        warning_file = f"{chkDir}/precheck_warnings.txt"
        with open(warning_file, "w") as file:
            for line in duplicate_warnings:
                file.write(line + "\n")
            file.write("\n")
            for line in angle_warnings:
                file.write(line + "\n")
            file.write("\n")
            for line in length_warnings:
                file.write(line + "\n")
            file.write("\n")
            for line in overlap_warnings:
                file.write(line + "\n")
        print(f"\nSaved warnings to {warning_file}")
    else:
        print(f"\nNo warnings detected!")


