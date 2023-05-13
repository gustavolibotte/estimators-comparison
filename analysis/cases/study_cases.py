import numpy as np
from filters.least_squares import LS
from statistics.analysis_statistics import AnalysisStatistics
from utils.array_file_manager import ArrayFileManager
from utils.plot_manager import GraphPlotter

class StudyCases:
    """
    A class to run and analyze study cases on filtered datasets.

    Parameters
    ----------
    training_dataset : numpy.ndarray
        The training dataset containing time, samples, and amplitudes.
    test_dataset : numpy.ndarray
        The test dataset containing time, samples, and amplitudes.
    n_slices : int
        The number of slices in the datasets.
    slice_size : int
        The size of each slice in the datasets.
    occupancy : float
        The occupancy of the dataset.

    Methods
    -------
    run_case_1(save_file: bool)
        Runs study case 1 by estimating amplitudes, computing errors, and plotting results.
    __handle_dataset(dataset: numpy.ndarray)
        Handles the input dataset by reshaping the time, samples, and amplitudes arrays.
    __run_filtering(samples: numpy.ndarray, amplitudes: numpy.ndarray)
        Runs the least squares filter on the input samples and amplitudes.
    """
    
    def __init__(self, training_dataset, test_dataset, n_slices, slice_size, occupancy) -> None:
        self.training_dataset = training_dataset
        self.test_dataset = test_dataset
        self.n_slices = n_slices
        self.slice_size = slice_size
        self.occupancy = occupancy
    
    def run_case_1(self, save_file):
        """
        Runs study case 1 by estimating amplitudes, computing errors, and plotting results.
        
        Parameters
        ----------
        save_file : bool
            Whether to save the error amplitudes to a file.
        
        Returns
        -------
        None
        """

        # Organizing loaded data
        _, training_samples, training_amplitudes = self.__handle_dataset(self.training_dataset)

        # Taking the central amplitudes of each window
        training_amplitudes = training_amplitudes[:, self.slice_size//2]

        # Running least squares
        weights, success = self.__run_filtering(training_samples, training_amplitudes)
        if success:
            estimated_amplitudes = AnalysisStatistics.estimate_amplitudes(training_samples, weights,
                                                                         self.n_slices, self.slice_size)

            # Calculating the error between test dataset and estimated amplitudes
            _, _, test_amplitudes = self.__handle_dataset(self.test_dataset)
            error_amplitudes = AnalysisStatistics.compare_amplitudes(test_amplitudes, self.n_slices,
                                                                     self.slice_size, estimated_amplitudes)
            
            # Computing sample mean and standard deviation
            error_mean = np.mean(error_amplitudes)
            error_stdev = np.std(error_amplitudes)
            
            # Plotting results
            plotter = GraphPlotter(figsize=(5, 3))
            plotter.hist_plot(error_amplitudes, error_mean, error_stdev, "Error")

            if save_file:
                ArrayFileManager.save_array_to_file("results/",
                                                    f"error_occupancy_{self.occupancy}_slice_{self.slice_size}.csv",
                                                    error_amplitudes)
        else:
            print("Least squares could not find a feasible solution.")
    
    def __handle_dataset(self, dataset):
        """
        Handles the input dataset by reshaping the time, samples, and amplitudes arrays.
        
        Parameters
        ----------
        dataset : numpy.ndarray
            The dataset containing time, samples, and amplitudes.
        
        Returns
        -------
        Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
            A tuple of times, samples, and amplitudes arrays.
        """
        times = np.reshape(dataset[:, 0], (self.n_slices, self.slice_size))
        samples = np.reshape(dataset[:, 1], (self.n_slices, self.slice_size))
        amplitudes = np.reshape(dataset[:, 2], (self.n_slices, self.slice_size))

        return (times, samples, amplitudes)
    
    def __run_filtering(self, samples, amplitudes):
        """
        Runs the least squares filter on the input samples and amplitudes.
        
        Parameters
        ----------
        samples : numpy.ndarray
            The samples array.
        amplitudes : numpy.ndarray
            The amplitudes array.
        
        Returns
        -------
        Tuple[numpy.ndarray, bool]
            A tuple of weights and success status.
        """
        n_filter = self.slice_size
        
        ls = LS(samples, amplitudes, n_filter)
        weights, status = ls.go_filtering()

        return (weights, status)
