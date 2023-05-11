import matplotlib
from matplotlib import pyplot as plt

class GraphPlotter:

    def __init__(self, figsize, nrows=1, ncols=1):
        self.fig, self.ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    def hist_plot(self, y):
        self.ax.hist(y)
        self.fig.tight_layout()