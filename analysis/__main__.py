import argparse

import numpy as np

from cases.study_cases import StudyCases
from utils.array_file_manager import ArrayFileManager

def calculate_results(data_path, occupancy, slice_size):
    if np.mod(slice_size, 2) != 0:
        # Loading data
        filename = f"training_occupancy_{occupancy}.csv"
        training_dataset = ArrayFileManager.read_array_from_file(data_path, filename)

        filename = f"test_occupancy_{occupancy}.csv"
        test_dataset = ArrayFileManager.read_array_from_file(data_path, filename)

        # Retrieving data size
        data_size, _ = np.shape(training_dataset)

        # Discarding edge data and calculating the number of slices
        discard_size = np.mod(data_size, slice_size)
        if discard_size == 0:
            n_slices = data_size // slice_size
        else:
            n_slices = (data_size - discard_size) // slice_size
            training_dataset = training_dataset[:-discard_size, :]
            test_dataset = test_dataset[:-discard_size, :]

        # Running analysis
        analysis_cases = StudyCases(training_dataset, test_dataset, n_slices, slice_size, occupancy)
        analysis_cases.run_case_1(save_file=True, plot_results=True)
    else:
        raise ValueError("Slice size must be an odd number.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", required=False, type=str, help="Calorimetry pulse data relative path (default: data/)")
    parser.add_argument("--occupancy", required=True, type=float, help="Occupancy (0.1 | 0.3 | 0.5)")
    parser.add_argument("--slice_size", required=True, type=int, help="Slice size (integer greater than zero)")
    
    args = parser.parse_args()
    calculate_results(args.data_path, args.occupancy, args.slice_size)