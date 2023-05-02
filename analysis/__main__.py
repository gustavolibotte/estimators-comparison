import os
import numpy as np
from numpy.random import uniform, standard_normal

from pycps import Random, TextFilePulseShape
from filters.of2 import OF2
from filters.least_squares import LS

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

    # Testing the filters
    n_filter = 7
    t_filter = np.array([-75.0, -50.0, -25.0, 0.0, 25.0, 50.0, 75.0])
    g = np.array([0.0000, 0.0172, 0.4524, 1.0000, 0.5633, 0.1493, 0.0424])
    dg = np.array([0.00004019,  0.00333578,  0.03108120, 0.00000000, -0.02434490,
                   -0.00800683, -0.00243344])
    
    # OF2
    of2 = OF2(n_filter, t_filter, g, dg)
    w = of2.go_filtering()
    print(w)

    # Constrained least-squares
    temp_amplitudes = 1023.0 * uniform(low=0.0, high=1.0, size=n_events)
    temp_noise = standard_normal(size=(n_events, n_filter))

    temp_samples = np.zeros((n_events, n_filter))
    for i in range(n_events):
        temp_samples[i] = temp_amplitudes[i] * g + temp_noise[i]

    ls = LS(temp_samples, temp_amplitudes, n_filter, t_filter, g, dg)
    w = ls.go_filtering()
    print(w)

if __name__ == '__main__':
    try:
        main()
    except FileNotFoundError as e:
        print(f'Error: {e}')
        exit(1)