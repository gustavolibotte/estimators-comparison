import matplotlib
from matplotlib import pyplot as plt

class GraphPlotter:
    """
    GraphPlotter class for plotting various types of figures.

    Parameters:
    -----------
    figsize : tuple
        Figure size as a tuple of (width, height).
    nrows : int, optional (default=1)
        Number of rows in the figure.
    ncols : int, optional (default=1)
        Number of columns in the figure.

    Methods:
    --------
    hist_plot(y, mean, stdev, xlabel)
        Plots histogram with mean and standard deviation.
        
        Parameters:
        -----------
        y : array-like
            Input data for the histogram.
        mean : float
            Mean value of the data.
        stdev : float
            Standard deviation of the data.
        xlabel : str
            Label for the x-axis of the plot.
    """

    def __init__(self, figsize, nrows=1, ncols=1):
        self.fig, self.ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    def hist_plot(self, y, mean, stdev, xlabel):
        self.ax.hist(y, bins=70, density=True, rwidth=0.85, color="C0")

        self.ax.set_xlabel(xlabel)
        self.ax.set_xlim((mean - 3 * stdev, mean + 3 * stdev))

        ylim = self.ax.get_ylim()
        self.ax.plot((mean, mean), ylim, "--", color="C1", label="Mean")

        self.ax.fill_between((mean - stdev, mean + stdev),
                             (ylim[0], ylim[0]), (ylim[1], ylim[1]),
                             color="C7", alpha=0.5, label="Std dev")
        self.ax.fill_between((mean - 2 * stdev, mean - stdev),
                             (ylim[0], ylim[0]), (ylim[1], ylim[1]),
                             color="C7", alpha=0.35, label="2 Std devs")
        self.ax.fill_between((mean + stdev, mean + 2 * stdev),
                             (ylim[0], ylim[0]), (ylim[1], ylim[1]),
                             color="C7", alpha=0.35)
        self.ax.fill_between((mean - 3 * stdev, mean - 2 * stdev),
                             (ylim[0], ylim[0]), (ylim[1], ylim[1]),
                             color="C7", alpha=0.2, label="3 Std devs")
        self.ax.fill_between((mean + 2 * stdev, mean + 3 * stdev),
                             (ylim[0], ylim[0]), (ylim[1], ylim[1]),
                             color="C7", alpha=0.2)
        
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['left'].set_visible(False)
        self.ax.get_yaxis().set_visible(False)

        self.ax.legend()

        self.fig.tight_layout()

        plt.show()