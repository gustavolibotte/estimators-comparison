from abc import ABC, abstractmethod

class Filter(ABC):
    """
    Abstract class for filter methods.

    Attributes:
    -----------
    _n_filter: int
        The number of filters.
    _t_filter: float
        The filter bandwidth.
    _g: ndarray
        A numpy array of shape (n_filter,) containing the filter coefficients.
    _dg: ndarray
        A numpy array of shape (n_filter,) containing the derivative of the filter coefficients.
    _weights: ndarray
        A numpy array of shape (n_filter,) that represents the weights of the filter.
    _status: bool
        A boolean that represents the status of the filter. True if the filter weights are valid, False otherwise.
    """
    def __init__(self, n_filter, t_filter, g, dg):
        self._n_filter = n_filter

        if len(t_filter) != n_filter:
            raise ValueError(f"Expected vector of length {n_filter}, but got {len(t_filter)}")
        self._t_filter = t_filter

        if len(g) != n_filter:
            raise ValueError(f"Expected vector of length {n_filter}, but got {len(g)}")
        self._g = g

        if len(g) != len(dg):
            raise ValueError("The length of 'g' must match the length of 'dg'")
        self._dg = dg
        
        self._weights = None
        self._status = False

    @abstractmethod
    def go_filtering(self):
        pass