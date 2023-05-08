from filters.filter import Filter

import numpy as np
from numpy.linalg import inv

class LS(Filter):
    """
    A class for Least Squares Filter, a subclass of Filter.

    Attributes
    ----------
    _samples : numpy.ndarray
        A 2D numpy array of shape (n_samples, n_filter) representing the filter samples.
    _amplitudes : numpy.ndarray
        A 2D numpy array of shape (n_samples, 1) representing the amplitudes of the filter samples.
    _vec_a : numpy.ndarray
        A 2D numpy array of shape (1, n_filter) representing the vector a used to impose constraints.
    _b : float
        A float value representing the constant term in the constraint equation.
    vec_w : numpy.ndarray
        A 2D numpy array of shape (n_filter, 1) representing the weight vector of the filter.

    Methods
    -------
    _predict_w()
        Predicts the weight vector.
    _solve_cstr_ls()
        Solves the constrained least-squares problem.
    _get_weights()
        Sets the filter weights.
    _check_solution()
        Checks the solution for feasibility.

    """

    def __init__(self, samples, amplitudes, n_filter):
        """
        Parameters
        ----------
        samples : numpy.ndarray
            A 2D numpy array of shape (n_samples, n_filter) representing the filter samples.
        amplitudes : numpy.ndarray
            A 2D numpy array of shape (n_samples, 1) representing the amplitudes of the filter samples.
        n_filter : int
            The number of filter coefficients.
        """

        self._samples = samples
        self._amplitudes = amplitudes

        self._vec_a = np.ones((1, n_filter))
        self._b = 0.0

        self.vec_w = None

        super().__init__(n_filter, None, None, None)
    
    def _predict_w(self):
        """
        Predicts the weight vector.

        Returns
        -------
        numpy.ndarray
            A 2D numpy array of shape (n_filter, 1) representing the weight vector of the filter.
        """

        return inv(self._samples.T @ self._samples) @ self._samples.T @ self._amplitudes
    
    def _solve_cstr_ls(self):
        """
        Solves the constrained least-squares problem.
        """

        mat_h_inv = inv(self._samples.T @ self._samples)
        mat_h_a = mat_h_inv @ self._vec_a.T
        mat_h_inv = inv(self._vec_a @ mat_h_inv @ self._vec_a.T)

        w = self._predict_w()
        cstr_lagr = self._vec_a @ w - self._b

        self.vec_w = w - (mat_h_a @ mat_h_inv @ cstr_lagr)

    def _get_weights(self):
        """
        Sets the filter weights.
        """

        self._weights = self.vec_w
    
    def _check_solution(self):
        """
        Checks the solution for feasibility.
        """

        if np.abs(np.sum(self._weights)) < 1e-12:
            self._status = True
        else:
            self._status = False

    def go_filtering(self):
        """
        Runs the Least Squares filter and returns the filter weights and status.

        Returns
        -------
        tuple
            A tuple containing the filter weights (ndarray of shape (n_filter,)) and the status of the filter (bool).
        """

        self._solve_cstr_ls()

        self._get_weights()
        self._check_solution()
        
        return (self._weights, self._status)