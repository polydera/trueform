"""
Offset Blocked Array for variable-length blocks

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

import numpy as np
from .. import _trueform


class OffsetBlockedArray:
    """
    A range of variable-length blocks backed by two numpy arrays.

    The data is stored in two arrays:
    - offsets: Block boundaries (must start at 0, end at len(data))
    - data: Packed data for all blocks

    Example:
        offsets = np.array([0, 3, 7, 10], dtype=np.int32)
        data = np.array([0,1,2, 3,4,5,6, 7,8,9], dtype=np.int32)

        This creates 3 blocks:
        - Block 0: [0, 1, 2]
        - Block 1: [3, 4, 5, 6]
        - Block 2: [7, 8, 9]

    Usage:
        >>> blocks = OffsetBlockedArray(offsets, data)
        >>> len(blocks)  # Number of blocks
        3
        >>> blocks[0]    # First block (view into data)
        array([0, 1, 2])
        >>> for block in blocks:
        ...     for value in block:
        ...         process(value)
    """

    def __init__(self, offsets: np.ndarray, data: np.ndarray):
        """
        Create an offset block range view.

        Parameters
        ----------
        offsets : np.ndarray
            1D array of block boundaries. Must be int32 or int64 dtype.
            offsets[0] must be 0, offsets[-1] must equal len(data).
        data : np.ndarray
            1D array of packed data. Must be int32 or int64 dtype.

        Raises
        ------
        TypeError
            If arrays are not int32 or int64, or if dtypes don't match
        ValueError
            If offsets don't satisfy constraints
        """
        # Validate dtypes
        if offsets.dtype not in [np.int32, np.int64]:
            raise TypeError(
                f"offsets must be int32 or int64, got {offsets.dtype}")
        if data.dtype not in [np.int32, np.int64]:
            raise TypeError(f"data must be int32 or int64, got {data.dtype}")

        # Both must have same dtype
        if offsets.dtype != data.dtype:
            raise TypeError(
                f"offsets and data must have same dtype, got {offsets.dtype} and {data.dtype}")

        # Validate shapes
        if offsets.ndim != 1:
            raise ValueError(f"offsets must be 1D, got shape {offsets.shape}")
        if data.ndim != 1:
            raise ValueError(f"data must be 1D, got shape {data.shape}")

        # Store arrays and create C++ wrapper
        self._offsets = offsets
        self._data = data

        # C++ wrapper validates offset constraints (first=0, last=len(data))
        # Select appropriate wrapper based on dtype
        if offsets.dtype == np.int32:
            self._wrapper = _trueform.core.OffsetBlockedArrayWrapperIntInt(
                offsets, data)
        else:  # int64
            self._wrapper = _trueform.core.OffsetBlockedArrayWrapperInt64Int64(
                offsets, data)

    def __len__(self) -> int:
        """Number of blocks"""
        return self._wrapper.size()

    def __getitem__(self, idx: int) -> np.ndarray:
        """
        Get block at index idx as a view into the data array.

        Parameters
        ----------
        idx : int
            Block index (0 <= idx < len(self))

        Returns
        -------
        np.ndarray
            View into data array for this block
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(
                f"Block index {idx} out of range [0, {len(self)})")

        start = self._offsets[idx]
        end = self._offsets[idx + 1]
        return self._data[start:end]

    def __iter__(self):
        """Iterate over blocks"""
        for i in range(len(self)):
            yield self[i]

    def __repr__(self) -> str:
        """String representation"""
        return f"OffsetBlockedArray({len(self)} blocks, {len(self._data)} total elements)"

    @property
    def offsets(self) -> np.ndarray:
        """Get the offsets array"""
        return self._offsets

    @property
    def data(self) -> np.ndarray:
        """Get the data array"""
        return self._data

    @property
    def dtype(self) -> np.dtype:
        """Get the dtype of the data array"""
        return self._data.dtype

    @classmethod
    def from_uniform(cls, array: np.ndarray) -> "OffsetBlockedArray":
        """
        Create an OffsetBlockedArray from a uniform 2D array.

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
        >>> oba = tf.OffsetBlockedArray.from_uniform(quads)
        >>> len(oba)  # 2 blocks
        2
        """
        if not isinstance(array, np.ndarray):
            raise TypeError(f"Expected np.ndarray, got {type(array).__name__}")

        if array.ndim != 2:
            raise ValueError(
                f"Expected 2D array, got {array.ndim}D with shape {array.shape}"
            )

        if array.dtype not in (np.int32, np.int64):
            raise TypeError(
                f"dtype must be int32 or int64, got {array.dtype}. "
                f"Convert with array.astype(np.int32)"
            )

        n_blocks, block_size = array.shape
        offsets = np.arange(
            0, (n_blocks + 1) * block_size, block_size, dtype=array.dtype
        )
        data = np.ascontiguousarray(array).ravel()

        return cls(offsets, data)
