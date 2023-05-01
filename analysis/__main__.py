import os
from pycps import Random, TextFilePulseShape

from datasets.setup_dataset import SetupDataset

def main():

    # Reading the pulse shape file
    path = 'base/unipolar-pulse-shape.dat'
    abs_path = os.path.abspath(os.path.join(os.getcwd(), path))
    
    if os.path.isfile(abs_path):
        pulse_shape = TextFilePulseShape(abs_path)
    else:
        raise FileNotFoundError(f'File {path} was not found.')

    # Setting up the random seed
    Random.seed(0)

    # Defining the number of events
    n_events = 1_000

    # Creating the dataset
    dataset_obj = SetupDataset(pulse_shape, n_events)
    dataset_obj.create_dataset(0.0)

    # Retrieving the quantities of interest from the dataset
    dataset_times = dataset_obj.get_dataset_times()
    dataset_samples = dataset_obj.get_dataset_samples()
    dataset_amplitudes = dataset_obj.get_dataset_amplitudes()

    print(dataset_times, dataset_samples, dataset_amplitudes)
    

if __name__ == '__main__':
    try:
        main()
    except FileNotFoundError as e:
        print(f'Error: {e}')
        exit(1)