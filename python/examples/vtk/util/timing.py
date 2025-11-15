"""
Performance timing utilities for VTK examples
"""

from collections import deque


class RollingAverage:
    """
    Track rolling average over a fixed number of samples

    Mimics the C++ pattern used across examples for performance timing.
    """

    def __init__(self, maxlen=1000):
        """
        Parameters
        ----------
        maxlen : int
            Maximum number of samples to track
        """
        self.samples = deque(maxlen=maxlen)

    def add(self, value):
        """
        Add a new sample

        Parameters
        ----------
        value : float
            New sample value
        """
        self.samples.append(value)

    def get_average(self):
        """
        Get current rolling average

        Returns
        -------
        float
            Average of all samples, or 0.0 if no samples
        """
        if len(self.samples) == 0:
            return 0.0
        return sum(self.samples) / len(self.samples)

    def __len__(self):
        return len(self.samples)


def format_time_us(seconds):
    """
    Format time in seconds as microseconds string

    Parameters
    ----------
    seconds : float
        Time in seconds

    Returns
    -------
    str
        Formatted string like "123.4 μs"
    """
    return f"{seconds * 1e6:.1f} μs"


def format_time_ms(seconds):
    """
    Format time in seconds as milliseconds string

    Parameters
    ----------
    seconds : float
        Time in seconds

    Returns
    -------
    str
        Formatted string like "12.3 ms"
    """
    return f"{seconds * 1e3:.1f} ms"
