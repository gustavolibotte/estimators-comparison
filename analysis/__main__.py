import os
import numpy as np
from numpy.random import uniform, standard_normal

from pycps import Random, TextFilePulseShape
from filters.least_squares import LS

from datasets.setup_dataset import SetupDataset
from utils.array_file_manager import ArrayFileManager

def main():

    # Reading the pulse shape file
    pulse_shape_path = 'analysis/base/unipolar-pulse-shape.dat'
    abs_path = os.path.abspath(os.path.join(os.getcwd(), pulse_shape_path))
    
    if os.path.isfile(abs_path):
        pulse_shape = TextFilePulseShape(abs_path)
    else:
        raise FileNotFoundError(f'File {pulse_shape_path} was not found.')

    # Setting up the random seed
    Random.seed(0)

    # Defining the number of slices and slice size
    n_slices = 1_000_000
    slice_size = 7

    # Defining the occupancy
    occupancy = 0.1

    # Creating the dataset
    # dataset_obj = SetupDataset(pulse_shape, n_slices, slice_size)
    # dataset_obj.create_sliced_dataset(occupancy)
    # dataset = dataset_obj.get_flatten_dataset()

    # Saving dataset to file
    # directory = "data/"
    # filename = f"occupancy_{occupancy}.csv"
    # ArrayFileManager.save_array_to_file(directory, filename, dataset)

    # Reading dataset from file
    directory = "data/"
    filename = f"occupancy_{occupancy}.csv"
    dataset = ArrayFileManager.read_array_from_file(directory, filename)

    times = dataset[:, 0].reshape((n_slices, slice_size))
    samples = dataset[:, 1].reshape((n_slices, slice_size))
    amplitudes = dataset[:, 2].reshape((n_slices, slice_size))

if __name__ == '__main__':
    try:
        main()
        # temp = np.array([[1, 2, 3], [4, 5, 6]])
        # print(temp)
        # temp = np.reshape(temp, (-1, 1)).flatten()
        # print(temp)
        # temp = np.reshape(temp, (2, 3))
        # print(temp)
    except FileNotFoundError as e:
        print(f'Error: {e}')
        exit(1)