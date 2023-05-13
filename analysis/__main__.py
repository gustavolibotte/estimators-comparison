import argparse

from cases.study_cases import StudyCases
from utils.array_file_manager import ArrayFileManager

def main(data_path, occupancy):
    # Defining the number of slices and slice size (TODO: change)
    n_slices = 1_000_000
    slice_size = 7

    filename = f"training_occupancy_{occupancy}_slice_{slice_size}.csv"
    training_dataset = ArrayFileManager.read_array_from_file(data_path, filename)

    filename = f"test_occupancy_{occupancy}_slice_{slice_size}.csv"
    test_dataset = ArrayFileManager.read_array_from_file(data_path, filename)

    # Running results
    analysis_cases = StudyCases(training_dataset, test_dataset, n_slices, slice_size, occupancy)
    analysis_cases.run_case_1(save_file=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, help="Calorimetry pulse data relative path (default: data/)")
    parser.add_argument("--occupancy", type=float, help="Occupancy (0.1 | 0.3 | 0.5)")
    
    args = parser.parse_args()
    if args.data_path and args.occupancy:
        main(args.data_path, args.occupancy)
    else:
        print("Please specify both the calorimetry pulse data relative path and occupancy.")