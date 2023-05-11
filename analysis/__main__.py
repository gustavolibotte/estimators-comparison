from cases.study_cases import StudyCases

def main():
    # # Saving dataset to file
    # directory = "data/"
    # filename = f"training_occupancy_{occupancy}.csv"
    # ArrayFileManager.save_array_to_file(directory, filename, dataset)

    # Reading dataset from file
    # directory = "data/"
    # filename = f"training_occupancy_{occupancy}.csv"
    # dataset = ArrayFileManager.read_array_from_file(directory, filename)

    # Setting up the pulse shape path
    pulse_shape_path = 'analysis/base/unipolar-pulse-shape.dat'

    analysis_cases = StudyCases(pulse_shape_path)
    analysis_cases.case_1()

if __name__ == '__main__':
    try:
        main()
    except FileNotFoundError as e:
        print(f'Error: {e}')
        exit(1)