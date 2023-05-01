from filters.filter import Filter

import numpy as np
from numpy.linalg import solve

class LS(Filter):
    def __init__(self, n_filter, t_filter, g, dg):

        super().__init__(n_filter, t_filter, g, dg)

    def go_filtering(self):
        print("Got it!")