import os
import numpy as np

class ArrayFileManager:
    """
    A class for saving and reading arrays of floating point numbers using numpy.

    Usage example:

    directory = "path/to/directory/"
    filename = "example.txt"
    array = np.array([1.0, 2.0, 3.0])

    1. Writing
    ArrayFileManager.save_array_to_file(directory, filename, array)

    2. Reading
    loaded_array = ArrayFileManager.read_array_from_file(directory, filename)
    """

    @staticmethod
    def save_array_to_file(directory, filename, array):
        """
        Save the given array to a file with the given filename in the given directory.

        Parameters:
        directory (str): The directory to save the file to.
        filename (str): The name of the file to save.
        array (numpy.ndarray): The array to save.

        Raises:
        FileNotFoundError: If the directory does not exist.
        PermissionError: If the directory is not writable.
        IOError: If there is an error writing to the file.
        """
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"Directory {directory} does not exist")
        if not os.access(directory, os.W_OK):
            raise PermissionError(f"No write permission in directory {directory}")
        
        filepath = os.path.abspath(os.path.join(os.getcwd(), directory, filename))
        try:
            np.savetxt(filepath, array, fmt='%.4f', delimiter=', ', comments="")
        except IOError as e:
            raise IOError(f"Error writing to file {filepath}: {e}")
        else:
            print("File saved successfully")

    @staticmethod
    def read_array_from_file(directory, filename):
        """
        Read an array from a file with the given filename in the given directory.

        Parameters:
        directory (str): The directory to read the file from.
        filename (str): The name of the file to read.

        Returns:
        numpy.ndarray: The array read from the file.

        Raises:
        FileNotFoundError: If the file does not exist.
        IOError: If there is an error reading the file.
        """
        filepath = os.path.abspath(os.path.join(os.getcwd(), directory, filename))
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"File {filepath} does not exist")
        try:
            return np.loadtxt(filepath, delimiter=',')
        except IOError as e:
            raise IOError(f"Error reading file {filepath}: {e}")