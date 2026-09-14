"""
NIfTI-1 volume IO

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

from collections import namedtuple

import numpy as np

from .. import _trueform
from .._volume import Volume
from .._volume.volume import _ROWS


NiftiHeader = namedtuple(
    "NiftiHeader",
    ["dims", "spacing", "dtype", "posed", "reflecting", "slope", "inter",
     "units"],
)

_DATATYPES = {
    0: np.dtype(np.uint8),
    1: np.dtype(np.int16),
    2: np.dtype(np.uint16),
    3: np.dtype(np.int32),
    4: np.dtype(np.float32),
    5: np.dtype(np.float64),
}

_UNITS = {0: "unknown", 1: "m", 2: "mm", 3: "um"}


def read_nifti_header(path) -> NiftiHeader:
    """
    The header facts of a NIfTI-1 file, read without its samples.

    On a gzipped file only the head is inflated, so dispatching on a large
    scan's dtype is cheap. A refusal raises ValueError naming its fact.

    Returns
    -------
    NiftiHeader
        ``(dims, spacing, dtype, posed, reflecting, slope, inter, units)``.
        ``posed`` says the file states an affine the axis-aligned grid cannot
        absorb; ``reflecting`` says that pose flips handedness.
    """
    datatype, dims, spacing, slope, inter, units, posed, reflecting = (
        _trueform.io.read_nifti_header(str(path))
    )
    return NiftiHeader(
        tuple(dims), tuple(spacing), _DATATYPES[datatype], posed, reflecting,
        slope, inter, _UNITS.get(units, "unknown"),
    )


def read_nifti(path, dtype=None) -> Volume:
    """
    Read a NIfTI-1 volume (.nii or .nii.gz) in its native dtype.

    The samples arrive as the file stores them — an int16 CT stays int16 on a
    float32 grid, with no conversion pass: native-endian bytes, and the
    header's ``scl`` scaling already applied. ``dtype=`` asks for another
    sample type and converts on read through one cast. A file whose own dtype
    the layer carries no row for (int32) reads converted to float32; an
    explicit request the layer cannot serve is refused rather than
    substituted.

    A posed file lands its affine in :attr:`Volume.transformation`, so the
    isosurface of a scan is a patient-space mesh with no further step. The
    read splits the file's one placement canonically — spacing and any
    axis-aligned translation onto the grid, the rest into the pose — so a
    file written from an origin plus a pose reads back with the origin folded
    into the grid; the composed world placement is what round-trips.

    Parameters
    ----------
    path : str or Path
        The .nii or .nii.gz file.
    dtype : numpy.dtype, optional
        The sample type to read in: float32, float64, int16, uint16 or uint8.
        Default: the file's own.

    Returns
    -------
    Volume
        The field, posed when the file is.

    Raises
    ------
    ValueError
        If `dtype` is not a sample type the layer reads in, or the file is
        refused — the reader names its fact.

    Examples
    --------
    >>> import trueform as tf
    >>> scan = tf.read_nifti("ct.nii.gz")
    >>> faces, points = tf.isosurface(scan, iso=300)   # patient space
    """
    header = read_nifti_header(path)
    if dtype is None:
        # The file's own dtype, or float32 where the layer carries no row for
        # it: nothing was asked for, so nothing is refused.
        native = header.dtype if header.dtype in _ROWS else np.dtype(np.float32)
    else:
        native = np.dtype(dtype)
        if native not in _ROWS:
            accepted = ", ".join(str(d) for d in _ROWS)
            raise ValueError(
                f"read_nifti: dtype must be one of {accepted}, got {native}"
            )
    func = getattr(_trueform.volume, f"read_nifti_{_ROWS[native].suffix}")
    flat, dims, spacing, origin, pose, _reflecting = func(str(path))
    vol = Volume._from_parts(flat, dims, spacing, origin)
    if pose is not None:
        vol.transformation = np.asarray(pose).astype(vol.coordinate_dtype)
    return vol


def write_nifti(volume: Volume, path) -> bool:
    """
    Write a volume as NIfTI-1 (.nii, or .nii.gz by extension).

    A posed volume writes its :attr:`Volume.transformation` composed with the
    grid: a translation the axis-aligned grid can hold is absorbed into the
    origin, so such a file reads back unposed with a shifted origin — the same
    world placement, split the file's way.

    Returns
    -------
    bool
        True when the file was written, False when it was not.
    """
    if not isinstance(volume, Volume):
        raise TypeError(f"Expected a Volume, got {type(volume).__name__}")
    func = getattr(_trueform.volume,
                   f"write_nifti_{_ROWS[volume.dtype].suffix}")
    return func(volume._wrapper, str(path))
