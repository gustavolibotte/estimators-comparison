import os
import numpy as np
from numpy.random import uniform, standard_normal

from pycps import Random, TextFilePulseShape
from filters.least_squares import LS

from datasets.setup_dataset import SetupDataset

def main():

    # Reading the pulse shape file
    path = 'analysis/base/unipolar-pulse-shape.dat'
    abs_path = os.path.abspath(os.path.join(os.getcwd(), path))
    
    if os.path.isfile(abs_path):
        pulse_shape = TextFilePulseShape(abs_path)
    else:
        raise FileNotFoundError(f'File {path} was not found.')

    # Setting up the random seed
    Random.seed(0)

    # Defining the number of slices and slice size
    n_slices = 1_000
    slice_size = 7

    # Creating the dataset
    dataset_obj = SetupDataset(pulse_shape, n_slices, slice_size)
    dataset_obj.create_sliced_dataset(0.0)

    # Retrieving the quantities of interest from the dataset
    dataset_times = dataset_obj.get_dataset_times()
    dataset_samples = dataset_obj.get_dataset_samples()
    dataset_amplitudes = dataset_obj.get_dataset_amplitudes()

    print(np.shape(dataset_amplitudes))

if __name__ == '__main__':
    try:
        main()
    except FileNotFoundError as e:
        print(f'Error: {e}')
        exit(1)