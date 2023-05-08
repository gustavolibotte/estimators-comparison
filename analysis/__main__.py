import os
import numpy as np
from numpy.random import uniform, standard_normal

from pycps import Random, TextFilePulseShape
from filters.least_squares import LS

from datasets.setup_dataset import SetupDataset
from utils.array_file_manager import ArrayFileManager

import matplotlib.pyplot as plt

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
    n_slices = 100
    slice_size = 7

    # Defining the occupancy
    occupancy = 0.1

    # Creating the dataset
    dataset_obj = SetupDataset(pulse_shape, n_slices, slice_size)
    dataset_obj.create_sliced_dataset(occupancy)
    dataset = dataset_obj.get_flatten_dataset()

    # Saving dataset to file
    directory = "data/"
    filename = f"occupancy_{occupancy}.csv"
    ArrayFileManager.save_array_to_file(directory, filename, dataset)

    # Reading dataset from file
    # directory = "data/"
    # filename = f"occupancy_{occupancy}.csv"
    # dataset = ArrayFileManager.read_array_from_file(directory, filename)

    # times = dataset[:, 0].reshape((n_slices, slice_size))
    # samples = dataset[:, 1].reshape((n_slices, slice_size))
    # amplitudes = ((dataset[:, 2].reshape((n_slices, slice_size)))[:, slice_size//2])

    times = dataset_obj.get_dataset_times()
    samples = dataset_obj.get_dataset_samples()
    amplitudes = dataset_obj.get_dataset_amplitudes()

    print(f"times = {times}")
    # print(f"amplitude = {amplitudes}")
    # print(f"samples = {samples}")

    # # Asserting for data equality
    # np.testing.assert_almost_equal(times, dataset_obj.get_dataset_times(), decimal=4, err_msg='Not equals')
    # np.testing.assert_almost_equal(samples, dataset_obj.get_dataset_samples(), decimal=4, err_msg='Not equals')
    # np.testing.assert_almost_equal(amplitudes, dataset_obj.get_dataset_amplitudes(), decimal=4, err_msg='Not equals')

    # Running least squares
    n_filter = slice_size
    t_filter = np.array([-75.0, -50.0, -25.0, 0.0, 25.0, 50.0, 75.0])
    g = np.array([0.0000, 0.0172, 0.4524, 1.0000, 0.5633, 0.1493, 0.0424])
    dg = np.array([0.00004019,  0.00333578,  0.03108120, 0.00000000, -0.02434490,
                   -0.00800683, -0.00243344])
    
    temp_amplitudes = 1023 * uniform(low=0.0, high=1.0, size=n_slices)
    # print(f"amplitude = {temp_amplitudes}")
    temp_noise = standard_normal(size=(n_slices, n_filter))
    # print(f"noise = {temp_noise}")

    temp_samples = np.zeros((n_slices, n_filter))
    for i in range(n_slices):
        temp_samples[i] = temp_amplitudes[i] * g + temp_noise[i]
    # print(f"samples = {temp_samples}")

    # ls = LS(samples, amplitudes, n_filter)
    # w = ls.go_filtering()
    # print(w)

if __name__ == '__main__':
    try:
        main()
    except FileNotFoundError as e:
        print(f'Error: {e}')
        exit(1)