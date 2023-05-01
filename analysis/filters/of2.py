from filters.filter import Filter

import numpy as np
from numpy.linalg import solve

class OF2(Filter):
    """
    OF2 filter class that extends Filter class and implements a specific type of filter.

    Parameters:
    -----------
    n_filter: int
        The number of filters.
    t_filter: float
        The filter bandwidth.
    g: ndarray
        A numpy array of shape (n_filter,) containing the filter coefficients.
    dg: ndarray
        A numpy array of shape (n_filter,) containing the derivative of the filter coefficients.

    Attributes:
    -----------
    _mat_a: ndarray
        A numpy array of shape (n_filter + 3, n_filter + 3) that represents the matrix A of the OF2 filter.
    _vec_b: ndarray
        A numpy array of shape (n_filter + 3,) that represents the vector b of the OF2 filter.
    _vec_x: ndarray
        A numpy array of shape (n_filter + 3,) that represents the solution of the system Ax = b.
    _weights: ndarray
        A numpy array of shape (n_filter,) that represents the weights of the filter.
    _status: bool
        A boolean that represents the status of the filter. True if the filter weights are valid, False otherwise.

    Methods:
    --------
    of2() -> Tuple[ndarray, bool]:
        Runs the OF2 filter and returns the filter weights and status.
    
    Private Methods
    ---------------
    _solve_system() -> None:
        Solves the system Ax = b.
    _get_weights() -> None:
        Extracts the weights from the solution of the system Ax = b.
    _check_solution() -> None:
        Checks if the solution of the system Ax = b is valid.
    _setup_matrix_a() -> None:
        Sets up the matrix A of the OF2 filter.
    _setup_vector_b() -> None:
        Sets up the vector b of the OF2 filter.
    """

    def __init__(self, n_filter, t_filter, g, dg):
        """
        Initializes the OF2 filter object.

        Parameters
        ----------
        n_filter : int
            The number of filters.
        t_filter : float
            The filter bandwidth.
        g : ndarray
            A numpy array of shape (n_filter,) containing the filter coefficients.
        dg : ndarray
            A numpy array of shape (n_filter,) containing the derivative of the filter coefficients.
        """
        self._mat_a = np.zeros((n_filter + 3, n_filter + 3))
        self._vec_b = np.zeros(n_filter + 3)
        self._vec_x = None
        
        self._weights = None
        self._status = False

        super().__init__(n_filter, t_filter, g, dg)

    def _solve_system(self):
        """
        Solves the system Ax = b.
        """
        self._setup_matrix_a()
        self._setup_vector_b()
        self._vec_x = solve(self._mat_a, self._vec_b)

    def _get_weights(self):
        """
        Extracts the weights from the solution of the system Ax = b.
        """
        self._weights = self._vec_x[:self._n_filter]
    
    def _check_solution(self):
        """
        Checks if the solution of the system Ax = b is valid.
        """
        if np.allclose(np.dot(self._mat_a, self._vec_x), self._vec_b) and \
            np.all(-1.0 <= self._weights) and \
                np.all(self._weights <= 1.0) and \
                    np.abs(np.sum(self._weights)) < 1e-12:
            self._status = True
        else:
            self._status = False

    def _setup_matrix_a(self):
        """
        Sets up the matrix A of the OF2 filter for solving the linear system
        """
        self._mat_a[:self._n_filter, :self._n_filter] = np.identity(self._n_filter)

        self._mat_a[self._n_filter, :self._n_filter] = self._g
        self._mat_a[self._n_filter + 1, :self._n_filter] = self._dg
        self._mat_a[self._n_filter + 2, :self._n_filter] = np.ones(self._n_filter)

        self._mat_a[:self._n_filter, self._n_filter] = -self._g
        self._mat_a[:self._n_filter, self._n_filter + 1] = -self._dg
        self._mat_a[:self._n_filter, self._n_filter + 2] = -np.ones(self._n_filter)
    
    def _setup_vector_b(self):
        """
        Sets up the vector b of the OF2 filter for solving the linear system
        """
        self._vec_b[self._n_filter] = 1.0
    
    def of2(self):
        """
        Runs the OF2 filter and returns the filter weights and status.

        Returns
        -------
        tuple
            A tuple containing the filter weights (ndarray of shape (n_filter,)) and the status of the filter (bool).
        """
        self._solve_system()

        self._get_weights()
        self._check_solution()
        
        return (self._weights, self._status)
