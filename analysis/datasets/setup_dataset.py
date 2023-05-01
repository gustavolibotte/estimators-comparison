from pycps import TextFilePulseShape, PulseGenerator, DatasetGenerator
import numpy as np

class SetupDataset:
    """
    A class for creating a dataset of simulated pulses.

    Attributes
    ----------
    pulse_shape : pycps.TextFilePulseShape
        Object of the reference pulse shape.
    n_events : int
        The number of events in the dataset.
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

    Private Methods
    ---------------
    _setup_pulse_generator()
        Initializes the pulse generator object with the pulse shape and sets the amplitude and phase distributions,
        deformation level, and pedestal.
    _setup_dataset_generator(occupancy)
        Initializes the dataset generator object with the pulse generator, sampling rate, noise parameters,
        and occupancy.

    """
    def __init__(self, pulse_shape, n_events):
        """
        Parameters
        ----------
        pulse_shape : pycps.TextFilePulseShape
            Object of the reference pulse shape.
        n_events : int
            The number of events in the dataset.
        """
        self.pulse_shape = pulse_shape
        self.n_events = n_events

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
            PulseGenerator.EXPONENTIAL_DISTRIBUTION, [100])
        self._pulse_generator.set_phase_distribution(
            PulseGenerator.UNIFORM_INT_DISTRIBUTION, [-5, 5])
        self._pulse_generator.set_deformation_level(0.01)
        self._pulse_generator.set_pedestal(40)

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
        self._dataset_generator.set_noise_params(0, 1.5)
        self._dataset_generator.set_occupancy(occupancy)

    def create_dataset(self, occupancy):
        """
        Generates the dataset of simulated pulses with the specified occupancy.

        Parameters
        ----------
        occupancy : float
            The occupancy of the pulses in the dataset.
        """
        self._setup_pulse_generator()
        self._setup_dataset_generator(occupancy)

        self._dataset = self._dataset_generator.generate_continuous_dataset(self.n_events)

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
        return np.array(self._dataset.time)
