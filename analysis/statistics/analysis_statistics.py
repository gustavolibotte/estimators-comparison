import numpy as np

class AnalysisStatistics:
    """
    A collection of static methods for analyzing statistical properties of a given set of data.
    """

    @staticmethod
    def estimate_amplitudes(samples, weights, n_slices, slice_size):
        """
        Given a 2D array of data samples, a 1D array of weights, and the number of slices and slice size,
        estimate the amplitudes for each slice of the data using the given weights.

        Args:
            samples (numpy.ndarray): A 2D array of data samples.
            weights (numpy.ndarray): A 1D array of weights used to estimate amplitudes.
            n_slices (int): The number of slices to take from the data.
            slice_size (int): The size of each slice.

        Returns:
            numpy.ndarray: A 1D array of estimated amplitudes for each slice.
        """
        n_estimates = (n_slices - 1) * slice_size + 1
        amplitudes = np.zeros(n_estimates)

        x = samples.flatten()

        for i in range(n_estimates):
            x_window = x[i : i + slice_size]
            amplitudes[i] = np.sum(x_window * weights)
        
        return amplitudes
    
    @staticmethod
    def compare_amplitudes(test_amplitudes, n_slices, slice_size, estimated_amplitudes):
        """
        Given a set of test amplitudes, the number of slices, slice size, and the estimated amplitudes,
        compute the difference (error) between the estimated amplitudes and the target amplitudes.

        Args:
            test_amplitudes (numpy.ndarray): A 1D array of target amplitudes.
            n_slices (int): The number of slices to take from the data.
            slice_size (int): The size of each slice.
            estimated_amplitudes (numpy.ndarray): A 1D array of estimated amplitudes.

        Returns:
            numpy.ndarray: A 1D array of error amplitudes.
        """
        half_window = slice_size // 2 # integer floor division

        target_amplitudes = test_amplitudes.flatten()[half_window : n_slices * slice_size - half_window]
        error_amplitudes = estimated_amplitudes - target_amplitudes
        
        return error_amplitudes