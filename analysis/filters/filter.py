from abc import ABC, abstractmethod

class Filter(ABC):
    def __init__(self, n_filter, t_filter, g, dg):
        self._status = None
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

    @abstractmethod
    def of2(self):
        pass