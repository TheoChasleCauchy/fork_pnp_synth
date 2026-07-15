import argparse
import csv
import numpy as np

def process_csv(input_path: str):
    with open(input_path, mode='r') as infile:
        reader = csv.reader(infile)
        rows = list(reader)

        # Values Row
        row = rows[1]
        values = [float(x) for x in row[1:]]
        row_mean = np.mean(values)
        row_std = np.std(values)
    
    return row_mean, row_std

def main(morphing_name):

    smoothness_mean_value, smoothness_std_value = process_csv(f"generations/{morphing_name}/{morphing_name}_cdpam_values.csv")

    # Write smoothness values in a csv file
    with open(f"generations/{morphing_name}/{morphing_name}_smoothness_cdpam.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Smoothness CDPAM"])
        writer.writerow([f"{smoothness_mean_value} +- {smoothness_std_value}"])

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Load the morphing's audios.")
    parser.add_argument("morphing_name", type=str, help="Name of the morphing")

    # Parse arguments
    args = parser.parse_args()

    # Call main with the directory argument
    main(args.morphing_name)