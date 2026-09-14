"""
Volume data structure for trueform

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

from typing import NamedTuple, Optional
import numpy as np

from .._spatial._validation import ensure_contiguous
from .._trueform.volume import (
    VolumeWrapperFloat,
    VolumeWrapperDouble,
    VolumeWrapperInt16,
    VolumeWrapperUInt16,
    VolumeWrapperUInt8,
)


class _Row(NamedTuple):
    """One sample dtype the layer accepts, and the native entries it carries.

    `boolean` is the registered volume-boolean entry for this row, or None
    where no such entry exists — an unsigned field is not an SDF, so the row
    states its absence rather than a property a caller has to re-derive.
    `resample` names the row's regrid entry the same way; the integer rows
    state None until an integer-sourced regrid is registered.
    """

    wrapper: type
    suffix: str
    coordinate_dtype: np.dtype
    boolean: Optional[str]
    resample: Optional[str]


# A volume is a sampled function, so a row names both of its types: what a
# sample IS (the key) and where samples STAND (the coordinate dtype). They
# coincide for a real-valued field; an integer field is measured in counts and
# placed on a float32 grid, which is what lets int16 CT keep 0.7 mm spacing.
_ROWS = {
    np.dtype(np.float32): _Row(
        VolumeWrapperFloat, "float", np.dtype(np.float32),
        "volume_boolean_float", "resampled_volume_float"
    ),
    np.dtype(np.float64): _Row(
        VolumeWrapperDouble, "double", np.dtype(np.float64),
        "volume_boolean_double", "resampled_volume_double"
    ),
    np.dtype(np.int16): _Row(
        VolumeWrapperInt16, "int16", np.dtype(np.float32),
        "volume_boolean_int16", None
    ),
    np.dtype(np.uint16): _Row(
        VolumeWrapperUInt16, "uint16", np.dtype(np.float32), None, None
    ),
    np.dtype(np.uint8): _Row(
        VolumeWrapperUInt8, "uint8", np.dtype(np.float32), None, None
    ),
}

# The codes the native entries take for the type they emit in.
_DTYPE_CODES = {
    np.dtype(np.float32): 0,
    np.dtype(np.float64): 1,
}


def _validate_triple(value, name, dtype=float):
    """Validate a length-3 sequence and return it as a tuple of `dtype`."""
    try:
        result = tuple(dtype(v) for v in value)
    except (TypeError, ValueError):
        raise TypeError(f"{name} must be a sequence of 3 numbers, got {value!r}")
    if len(result) != 3:
        raise ValueError(f"{name} must have length 3, got {len(result)}")
    return result


def _to_accepted_sample_dtype(samples):
    """The samples in a dtype a row carries.

    A byte-swapped array is byteswapped into its own row rather than widened:
    NIfTI stores big-endian int16, and a CT scan should not become float32 for
    having crossed a network. Anything with no row at all becomes float32.
    """
    if samples.dtype in _ROWS:
        return samples
    native = samples.dtype.newbyteorder("=")
    if native in _ROWS:
        return samples.astype(native)
    return samples.astype(np.float32)


def _validate_pose(mat, coordinate_dtype):
    """The 4x4 world pose a volume takes, in the grid's own coordinate dtype."""
    if mat.shape != (4, 4):
        raise ValueError(
            f"Volume transformation must be 4x4, got shape {mat.shape}"
        )
    if mat.dtype != coordinate_dtype:
        raise TypeError(
            f"Volume transformation dtype ({mat.dtype}) must match the "
            f"volume's coordinate dtype ({coordinate_dtype})"
        )
    return ensure_contiguous(mat)


def _resolve_dtype(dtype, source_dtype):
    """The dtype a call emits in: the request, or the source's when there is none.

    The source is the volume's COORDINATE dtype, never its sample dtype — an
    unstated request asks for the grid the samples stand on.
    """
    if dtype is None:
        return source_dtype
    dtype = np.dtype(dtype)
    if dtype not in _DTYPE_CODES:
        raise TypeError(f"dtype must be float32 or float64, got {dtype}")
    return dtype


class Volume:
    """
    Dense, axis-aligned 3D scalar field (a regular voxel grid).

    Stores ``dims[0] * dims[1] * dims[2]`` samples on a regular grid, indexed
    ``samples[x, y, z]``. The mapping from a voxel index to a local-space point
    is ``origin + (x, y, z) * spacing``. The field is commonly interpreted as a
    signed distance field (SDF): negative inside, positive outside, zero on the
    surface — extract it with :func:`isosurface`.

    A volume is a sampled function, so it has TWO types: :attr:`dtype`, what a
    sample is, and :attr:`coordinate_dtype`, where the samples stand. They
    coincide for a real-valued field. An integer field — int16 CT counts,
    uint16, a uint8 mask — is stored as given and placed on a float32 grid, so
    native millimetre spacing survives without widening a single sample.

    The wrapper borrows the samples zero-copy: it retains the NumPy array and
    the native algorithms range over its memory, so in-place writes to the
    source array (or through the ``samples`` property, which views the same
    memory) mutate the field directly. The boundary normalization copies only
    when it must — a non-Fortran-ordered array, or one whose dtype is not an
    accepted sample type, is converted once at construction, after which that
    converted array is the retained source.

    Parameters
    ----------
    samples : np.ndarray
        3D array of shape (nx, ny, nz), indexed ``samples[x, y, z]``, with
        dtype float32, float64, int16, uint16 or uint8. A byte-swapped array
        of one of those is byteswapped into it; any other dtype is converted
        to float32. Any memory order is accepted; the field is normalized at
        the boundary.
    spacing : sequence of 3 floats, optional
        Physical size of one voxel step along x, y, z. Default (1, 1, 1).
    origin : sequence of 3 floats, optional
        Local-space position of sample (0, 0, 0). Default (0, 0, 0).
    transformation : np.ndarray, optional
        4x4 world pose in :attr:`coordinate_dtype`, mirroring
        :class:`~trueform.Mesh`: the grid stays axis-aligned in its local
        space and posed entries emit and resample through the pose.

    Examples
    --------
    >>> import numpy as np
    >>> import trueform as tf
    >>> x, y, z = np.meshgrid(np.arange(16.0), np.arange(16.0),
    ...                       np.arange(16.0), indexing="ij")
    >>> sdf = np.sqrt((x - 8)**2 + (y - 8)**2 + (z - 8)**2) - 5.0
    >>> vol = tf.Volume(sdf.astype(np.float32))
    >>> vol.dims
    (16, 16, 16)
    >>> faces, points = tf.isosurface(vol)
    """

    def __init__(
        self,
        samples: np.ndarray,
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
        transformation: np.ndarray = None,
    ):
        if not isinstance(samples, np.ndarray):
            raise TypeError(
                f"Expected numpy array for samples, got {type(samples)}"
            )
        if samples.ndim != 3:
            raise ValueError(
                f"Expected 3D array for samples, got shape {samples.shape}"
            )
        if samples.size == 0:
            raise ValueError("samples must contain at least one element")

        samples = _to_accepted_sample_dtype(samples)

        spacing = _validate_triple(spacing, "spacing")
        if any(not np.isfinite(s) or s <= 0.0 for s in spacing):
            raise ValueError(f"spacing must be finite and positive, got {spacing}")
        origin = _validate_triple(origin, "origin")

        # The native field is flat and x-fastest; a Fortran-ordered view of
        # samples[x, y, z] raveled in F order is exactly that layout.
        flat = np.asfortranarray(samples).reshape(-1, order="F")

        self._dtype = samples.dtype
        self._wrapper = _ROWS[self._dtype].wrapper(
            flat, samples.shape, spacing, origin
        )
        if transformation is not None:
            self.transformation = transformation

    @classmethod
    def _from_parts(cls, flat, dims, spacing, origin) -> "Volume":
        """Wrap a natively produced field (flat samples + metadata), zero-copy."""
        vol = object.__new__(cls)
        flat = np.asarray(flat)
        vol._dtype = flat.dtype
        vol._wrapper = _ROWS[vol._dtype].wrapper(
            flat,
            tuple(int(d) for d in dims),
            tuple(float(s) for s in spacing),
            tuple(float(o) for o in origin),
        )
        return vol

    @property
    def samples(self) -> np.ndarray:
        """
        The sample grid as a zero-copy view, indexed ``samples[x, y, z]``.

        The view shares memory with the retained source array; in-place writes
        mutate the field directly.
        """
        flat = self._wrapper.samples_array()
        return flat.reshape(self.dims, order="F")

    @property
    def dims(self) -> tuple:
        """Number of samples along x, y, z."""
        return tuple(self._wrapper.dims())

    @property
    def spacing(self) -> tuple:
        """Physical size of one voxel step along x, y, z."""
        return tuple(self._wrapper.spacing())

    @spacing.setter
    def spacing(self, value) -> None:
        value = _validate_triple(value, "spacing")
        if any(not np.isfinite(s) or s <= 0.0 for s in value):
            raise ValueError(f"spacing must be finite and positive, got {value}")
        self._wrapper.set_spacing(value)

    @property
    def origin(self) -> tuple:
        """Local-space position of sample (0, 0, 0)."""
        return tuple(self._wrapper.origin())

    @origin.setter
    def origin(self, value) -> None:
        self._wrapper.set_origin(_validate_triple(value, "origin"))

    @property
    def voxel_count(self) -> int:
        """Total number of samples (dims[0] * dims[1] * dims[2])."""
        return self._wrapper.voxel_count()

    @property
    def dtype(self) -> np.dtype:
        """
        Sample dtype — what a sample IS.

        float32, float64, int16, uint16 or uint8. Integer fields are stored as
        given; see :attr:`coordinate_dtype` for the grid they stand on.
        """
        return self._dtype

    @property
    def transformation(self):
        """
        The 4x4 world pose, or None when the volume is unposed.

        Mirrors :class:`~trueform.Mesh`: the matrix is in
        :attr:`coordinate_dtype`, the grid stays axis-aligned in its own
        local space, and posed entries (isosurface, resampling, booleans,
        slice contours) read the pose natively — emitted geometry lands in
        world space, positional inputs stay local. Set to None to clear;
        setting exactly the identity is the same statement, so an identity
        pose reads back as None. A pose that merely rounds to the identity
        is a pose, as the native combine's exact matrix equality reads it.
        """
        return self._wrapper.transformation()

    @transformation.setter
    def transformation(self, mat: np.ndarray) -> None:
        if mat is None:
            self._wrapper.clear_transformation()
            return
        mat = _validate_pose(mat, self.coordinate_dtype)
        # Standing at the identity is standing nowhere: unposed is how the
        # volume says it, in both directions. Exactly the identity, as the
        # native pose comparison reads equality.
        if np.array_equal(mat, np.eye(4)):
            self._wrapper.clear_transformation()
            return
        self._wrapper.set_transformation(mat)

    @property
    def coordinate_dtype(self) -> np.dtype:
        """
        Coordinate dtype — where the samples STAND.

        The type ``spacing``, ``origin`` and every emitted position answer in,
        and the dtype a call emits in when it is asked for none. Equal to
        :attr:`dtype` for a real-valued field; float32 for an integer one.
        """
        return _ROWS[self._dtype].coordinate_dtype

    def __repr__(self) -> str:
        nx, ny, nz = self.dims
        return (
            f"Volume({nx}x{ny}x{nz}, spacing={self.spacing}, "
            f"origin={self.origin}, dtype={self.dtype})"
        )
