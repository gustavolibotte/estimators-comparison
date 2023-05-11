import os
from pycps import Random, TextFilePulseShape

from datasets.setup_dataset import SetupDataset
from filters.least_squares import LS

class StudyCases:

    def __init__(self, pulse_shape_path):
        # Reading the pulse shape file
        abs_path = os.path.abspath(os.path.join(os.getcwd(), pulse_shape_path))
        if os.path.isfile(abs_path):
            self.pulse_shape = TextFilePulseShape(abs_path)
        else:
            raise FileNotFoundError(f'File {pulse_shape_path} was not found.')
    
    def case_1(self):
        # Setting up the random seed
        Random.seed(0)

        # Defining the number of slices and slice size
        self.n_slices = 100
        self.slice_size = 7

        # Defining the occupancy
        self.occupancy = 0.1

        # Creating the dataset
        _, training_samples, training_amplitudes = self.__create_dataset()

        # Taking the central amplitudes of each window
        training_amplitudes = training_amplitudes[:, self.slice_size//2]

        # Running least squares
        weights, success = self.__run_filtering(training_samples, training_amplitudes)
        if success:
            print(f"weights = {weights}")
    
    def __create_dataset(self):
        dataset_obj = SetupDataset(self.pulse_shape, self.n_slices, self.slice_size)
        dataset_obj.create_sliced_dataset(self.occupancy)
        
        times = dataset_obj.get_dataset_times()
        samples = dataset_obj.get_dataset_samples()
        amplitudes = dataset_obj.get_dataset_amplitudes()

        return (times, samples, amplitudes)
    
    def __run_filtering(self, samples, amplitudes):
        n_filter = self.slice_size
        
        ls = LS(samples, amplitudes, n_filter)
        weights, status = ls.go_filtering()

        return (weights, status)
