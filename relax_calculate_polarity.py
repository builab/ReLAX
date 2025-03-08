#!/usr/bin/env python3
# Script to calculate polarity based on the 2d alignment
# IDEA: instead of do average, take average of top 4
# Authors: HB, 03/2025

import os
import pandas as pd
import argparse

def calculate_top_n_average_cc(df, top_n):
    """
    Calculate the average of the top n CC values for each ObjectID in the DataFrame.
    If top_n is 0, calculate the average of all CC values.
    If top_n > number of rows, set top_n to the number of rows.
    """
    if top_n == 0:
        return df.groupby('ObjectID')['CC'].mean()
    else:
        return df.groupby('ObjectID')['CC'].apply(
            lambda x: x.nlargest(min(top_n, len(x))).mean()  # Ensure top_n does not exceed the number of rows
        )

def compare_polarity(polarity0_dir, polarity1_dir, output_file, top_n):
    """
    Compare the average of the top n CC values (or all CC values if top_n is 0)
    from polarity_0 and polarity_1 directories and generate an output CSV with the calculated polarity.
    """
    results = []

    # Get the list of CSV files in polarity_0 directory and sort them alphabetically
    csv_files = sorted([f for f in os.listdir(polarity0_dir) if f.endswith('.csv')])

    for csv_file in csv_files:
        # Construct file paths for polarity_0 and polarity_1
        polarity0_path = os.path.join(polarity0_dir, csv_file)
        polarity1_path = os.path.join(polarity1_dir, csv_file)

        # Check if the corresponding file exists in polarity_1
        if not os.path.exists(polarity1_path):
            print(f"Warning: {csv_file} not found in {polarity1_dir}. Skipping.")
            continue

        # Load CSV files
        df_polarity0 = pd.read_csv(polarity0_path)
        df_polarity1 = pd.read_csv(polarity1_path)

        # Calculate average of top n CC (or all CC if top_n is 0) for each ObjectID
        avg_cc_polarity0 = calculate_top_n_average_cc(df_polarity0, top_n)
        avg_cc_polarity1 = calculate_top_n_average_cc(df_polarity1, top_n)

        # Compare average CC values and determine polarity
        for object_id in avg_cc_polarity0.index:
            if object_id in avg_cc_polarity1.index:
                if avg_cc_polarity0[object_id] >= avg_cc_polarity1[object_id]:
                    polarity = 0
                else:
                    polarity = 1
            else:
                polarity = 0  # Default to polarity 0 if ObjectID is not in polarity_1

            tomo_name = csv_file.replace('.csv', '')
            print(f'Tomogram {tomo_name} with polarity {polarity} with CC0 {avg_cc_polarity0[object_id]:.3f} and CC1 {avg_cc_polarity1[object_id]:.3f}')
            results.append({
                'rlnTomoName': tomo_name,
                'ObjectID': object_id,
                'Polarity': polarity
            })

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)

    # Save the results to the output CSV file
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Calculate polarity based on comparison of CC values.")
    parser.add_argument("--polarity0_dir", type=str, required=True, help="Directory containing polarity_0 CSV files")
    parser.add_argument("--polarity1_dir", type=str, required=True, help="Directory containing polarity_1 CSV files")
    parser.add_argument("--output_file", type=str, default="polarity.csv", help="Output CSV file name")
    parser.add_argument("--top_n", type=int, default=0, required=True, help="Number of top CC values to average (use 0 to average all)")
    
    # Parse arguments
    args = parser.parse_args()

    # Call the comparison function
    compare_polarity(args.polarity0_dir, args.polarity1_dir, args.output_file, args.top_n)

if __name__ == "__main__":
    main()