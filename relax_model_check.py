#!/usr/bin/env python3
# Script to to check if IMOD model has problem.
# The following check should be performed
# - Check if one line is too short with a threshold
# - Check if line are duplicates
# - Check if there are two points from the same line are too close with a threshold
# python ~/Documents/GitHub/ReLAX/relax_model_check.py --mod_dir models --mod_pattern "*doublet.mod" --suffix_remove _doublet.mod --tomo_size "1024,1440,500" --angpix 8.48
# Written by Jerry Gao, McGill. Modified by Huy Bui

from util.precheck import run_model_check, conv2bool
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--mod_dir", type=str, required=True, help="Model directory")
parser.add_argument("--mod_pattern", type=str, required=True, help="Model file pattern")
parser.add_argument("--suffix_remove", type=str, required=True, help="Suffix string to be removed")
parser.add_argument("--angpix", type=float, required=True, help="Pixel size of model")
parser.add_argument("--tomo_size", type=str, default="1024,1440,500", required=True, help="Tomogram size (X, Y, Z)")
parser.add_argument("--min_separation", type=float, default=10, help="Min separation between contour points in nm")
parser.add_argument("--min_angle", type=float, default=100, help="Minimum angle between three points in a contour")
parser.add_argument("--min_len", type=float, default=50, help="Minimum length for a contour in nanometres")
parser.add_argument("--overlap_radius", type=float, default=10, help="Radius (nm) of each contour to calculate overlap")
parser.add_argument("--overlap_threshold", type=float, default=5, help="Threshold (%) overlap to detect")

parser.add_argument("--skip_graph", action="store_true", help="Skip the graphing step")
parser.add_argument("--print_length", action="store_true", help="Show length calculations")
parser.add_argument("--print_overlap", action="store_true", help="Show overlap calculations")
parser.add_argument("--print_angle", action="store_true", help="Show angle calculations")
parser.add_argument("--delete_duplicate", type=conv2bool, default=True, help="Delete duplicate points within min_separation of each other")
parser.add_argument("--delete_overlap", type=conv2bool, default=True, help="Delete overlapping contour")
parser.add_argument("--absolute-graph", action="store_true", help="Creates a graph with fixed axes")

args = parser.parse_args()
modDir = args.mod_dir
modFileDelimiter = args.mod_pattern
stringToBeRemoved = args.suffix_remove
angpix = args.angpix

tomo_x_size, tomo_y_size, tomo_z_size = map(int, args.tomo_size.split(','))
min_separation = args.min_separation
min_angle = args.min_angle
min_len = args.min_len
overlap_radius = args.overlap_radius
overlap_threshold = args.overlap_threshold

if __name__ == "__main__":
    print(f"Model dir: {modDir}")
    print(f"Model file pattern: {modFileDelimiter}")
    print(f"Suffix to be removed: {stringToBeRemoved}")
    print(f"Pixel size: {angpix} Angstrom")
    print("-------------------------------------------------------------------------------")
    run_model_check(modDir, "./", modFileDelimiter=modFileDelimiter, stringToBeRemoved=stringToBeRemoved, angpix=angpix,
         min_separation=min_separation, min_angle=min_angle, min_len=min_len, overlap_radius=overlap_radius, overlap_threshold=overlap_threshold,
         tomo_x_size=tomo_x_size, tomo_y_size=tomo_y_size, tomo_z_size=tomo_z_size,
         skip_graph=args.skip_graph, print_length=args.print_length, print_overlap=args.print_overlap, print_angle=args.print_angle, 
         delete_duplicate=args.delete_duplicate, delete_overlap=args.delete_overlap,
         absolute_graph=args.absolute_graph)