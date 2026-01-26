"""
as_offset_blocked convenience function

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

import numpy as np
from .offset_blocked_array import OffsetBlockedArray


def as_offset_blocked(array: np.ndarray) -> OffsetBlockedArray:
    """
    Convert a uniform 2D array to OffsetBlockedArray.

    Convenience function equivalent to OffsetBlockedArray.from_uniform(array).

    Parameters
    ----------
    array : np.ndarray
        2D array of shape (N, V) where N is number of blocks and V is
        vertices per block. Must have dtype int32 or int64.

    Returns
    -------
    OffsetBlockedArray
        OffsetBlockedArray with uniform block sizes.

    Examples
    --------
    >>> import trueform as tf
    >>> import numpy as np
    >>> quads = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int32)
    >>> faces = tf.as_offset_blocked(quads)
    >>> mesh = tf.Mesh(faces, points)
    """
    return OffsetBlockedArray.from_uniform(array)
