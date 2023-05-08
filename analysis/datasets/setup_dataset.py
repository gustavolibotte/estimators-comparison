from pycps import TextFilePulseShape, PulseGenerator, DatasetGenerator
import numpy as np

class SetupDataset:
    """
    A class for creating a dataset of simulated pulses.

    Attributes
    ----------
    pulse_shape : pycps.TextFilePulseShape
        Object of the reference pulse shape.
    n_slices : int
        The number of slices in the dataset.
    slice_size : int
        Window size.
    _pulse_generator : PulseGenerator or None
        The pulse generator object used to create the pulses.
    _dataset_generator : DatasetGenerator or None
        The dataset generator object used to create the dataset.
    _dataset : ContinuousPulseSequence or None
        The dataset of simulated pulses.

    Methods
    -------
    create_dataset(occupancy)
        Generates the dataset of simulated pulses with the specified occupancy.
    get_dataset_times()
        Returns an array of the times of the pulses in the dataset.
    get_dataset_samples()
        Returns an array of the samples of the pulses in the dataset.
    get_dataset_amplitudes()
        Returns an array of the amplitudes of the pulses in the dataset.
    get_flatten_dataset()
        Returns a flattened array of the times, samples, and amplitudes of the pulses in the dataset.

    Private Methods
    ---------------
    _setup_pulse_generator()
        Initializes the pulse generator object with the pulse shape and sets the amplitude and phase distributions,
        deformation level, and pedestal.
    _setup_dataset_generator(occupancy)
        Initializes the dataset generator object with the pulse generator, sampling rate, noise parameters,
        and occupancy.

    """
    def __init__(self, pulse_shape, n_slices, slice_size):
        """
        Parameters
        ----------
        pulse_shape : pycps.TextFilePulseShape
            Object of the reference pulse shape.
        n_slices : int
            The number of slices in the dataset.
        slice_size : int
            Window size
        """
        self.pulse_shape = pulse_shape
        self.n_slices = n_slices
        self.slice_size = slice_size

        self._pulse_generator = None
        self._dataset_generator = None
        self._dataset = None

    def _setup_pulse_generator(self):
        """
        Initializes the pulse generator object with the pulse shape and sets the amplitude and phase distributions,
        deformation level, and pedestal.
        """
        self._pulse_generator = PulseGenerator(self.pulse_shape)
        self._pulse_generator.set_amplitude_distribution(
            PulseGenerator.UNIFORM_REAL_DISTRIBUTION, [0, 1023])
        self._pulse_generator.set_phase_distribution(
            PulseGenerator.UNIFORM_INT_DISTRIBUTION, [-5, 5])
        self._pulse_generator.set_deformation_level(0.01)
        self._pulse_generator.set_pedestal(0.0)

    def _setup_dataset_generator(self, occupancy):
        """
        Initializes the dataset generator object with the pulse generator, sampling rate, noise parameters,
        and occupancy.

        Parameters
        ----------
        occupancy : float
            The occupancy of the pulses in the dataset.
        """
        self._dataset_generator = DatasetGenerator()
        self._dataset_generator.set_pulse_generator(self._pulse_generator)
        self._dataset_generator.set_sampling_rate(25.0)
        self._dataset_generator.set_noise_params(0.0, 1.5)
        self._dataset_generator.set_occupancy(occupancy)

    def create_sliced_dataset(self, occupancy):
        """
        Generates the dataset of simulated pulses with the specified occupancy.

        Parameters
        ----------
        occupancy : float
            The occupancy of the pulses in the dataset.
        """
        self._setup_pulse_generator()
        self._setup_dataset_generator(occupancy)

        self._dataset = self._dataset_generator.generate_sliced_dataset(self.n_slices, self.slice_size)

    def get_dataset_times(self):
        """
        Returns an array of the times of the pulses in the dataset.

        Returns
        -------
        ndarray
            The array of times.
        """
        return np.array(self._dataset.time)

    def get_dataset_samples(self):
        """
        Returns an array of the samples of the pulses in the dataset.

        Returns
        -------
        ndarray
            The array of samples.
        """
        return np.array(self._dataset.samples)

    def get_dataset_amplitudes(self):
        """
        Returns an array of the amplitudes of the pulses in the dataset.

        Returns
        -------
        ndarray
            The array of amplitudes.
        """
        return np.array(self._dataset.amplitudes)
    
    def get_flatten_dataset(self):
        """
        Returns a flattened version of the dataset.

        The method gets the times, samples and amplitudes of the dataset, flattens them,
        and creates a new array with shape (self.n_slices * self.slice_size, 3) that
        contains the flattened values.

        Returns:
            np.ndarray: A flattened version of the dataset with shape
            (self.n_slices * self.slice_size, 3), where each row contains the flattened
            values of the time, sample and amplitude of a slice.
        """
        times = self.get_dataset_times().flatten()
        samples = self.get_dataset_samples().flatten()
        amplitudes = self.get_dataset_amplitudes().flatten()

        flattened_dataset = np.zeros((self.n_slices * self.slice_size, 3))
        flattened_dataset[:, 0] = times
        flattened_dataset[:, 1] = samples
        flattened_dataset[:, 2] = amplitudes

        return flattened_dataset