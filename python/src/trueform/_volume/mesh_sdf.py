"""
Mesh signed distance field sampling

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

from typing import Tuple, Union
import numpy as np
from .. import _trueform
from .._spatial import Mesh
from .._dispatch import extract_meta, build_suffix
from .volume import Volume, _DTYPE_CODES, _resolve_dtype, _validate_triple


def mesh_sdf(
    data: Union[Mesh, Tuple[np.ndarray, np.ndarray]],
    dims,
    spacing,
    origin,
    dtype=None,
    mode: str = "exact",
    band: int = 2,
) -> Volume:
    """
    Sample the signed distance field of a closed mesh onto a regular grid.

    Evaluates the signed distance at every grid sample: negative inside the
    surface, positive outside, zero on it. Extracting the isosurface at value
    0 recovers the input surface. The sign is exact crossing parity on the
    integer lattice — negative inside by winding, so an inverted shell
    inverts its field and nested shells stay solid inside.

    The grid samples the mesh in the frame it states: world space when the
    mesh carries a transformation, its local frame otherwise. Build
    dims/spacing/origin from the mesh's bounds in that same space (plus any
    outside margin you want).

    Parameters
    ----------
    data : Mesh or tuple
        Input mesh data:
        - Mesh object (3D triangular or dynamic mesh)
        - Tuple (faces, points) with faces of shape (N, 3) (int32/int64) or an
          OffsetBlockedArray, and points of shape (M, 3) (float32/float64)
    dims : sequence of 3 ints
        Number of samples along x, y, z.
    spacing : sequence of 3 floats
        Physical size of one voxel step along x, y, z.
    origin : sequence of 3 floats
        Local-space position of sample (0, 0, 0).
    dtype : numpy.dtype, optional
        The sample type the field is computed and stored in, float32 or
        float64. Default: the mesh's own coordinate dtype.
    mode : str, optional
        "exact" (default) measures every sample; "banded" measures only the
        samples within ``band`` voxels of the surface and propagates the far
        field with a seeded distance sweep — an order of magnitude faster,
        exact in the band and in its sign everywhere. Beyond the band, on
        grids whose lines resolve the surface, the 99th percentile of the
        error is under about one voxel and shrinks with resolution; the
        deep interior near the medial axis may locally undershoot by a few
        voxels. A feature no grid line meets is not found when the band is
        measured, so its neighbourhood takes the swept far field's value.
    band : int, optional
        Banded only: voxels of exactly measured magnitude on each side of
        the surface. Default 2.

    Returns
    -------
    Volume
        A volume holding the mesh SDF, in `dtype`.

    Examples
    --------
    >>> import trueform as tf
    >>> faces, points = tf.make_sphere_mesh(5.0)
    >>> vol = tf.mesh_sdf((faces, points), (32, 32, 32),
    ...                   (0.4, 0.4, 0.4), (-6.0, -6.0, -6.0))
    >>> out_faces, out_points = tf.isosurface(vol)
    """
    # Normalize input to Mesh object
    if isinstance(data, tuple):
        if len(data) != 2:
            raise ValueError(
                f"Tuple input must have exactly 2 elements (faces, points), "
                f"got {len(data)}"
            )
        faces, points = data
        mesh = Mesh(faces, points)
    elif isinstance(data, Mesh):
        mesh = data
    else:
        raise TypeError(
            f"Expected Mesh or (faces, points) tuple, got {type(data).__name__}"
        )

    if mesh.dims != 3:
        raise ValueError(f"mesh_sdf only supports 3D meshes, got {mesh.dims}D")

    dims = _validate_triple(dims, "dims", dtype=int)
    if any(d < 1 for d in dims):
        raise ValueError(f"dims must be at least 1 along each axis, got {dims}")
    spacing = _validate_triple(spacing, "spacing")
    origin = _validate_triple(origin, "origin")

    if mode not in ("exact", "banded"):
        raise ValueError(f"mode must be 'exact' or 'banded', got {mode!r}")
    out_dtype = _resolve_dtype(dtype, np.dtype(mesh.dtype))
    suffix = build_suffix(extract_meta(mesh))
    func = getattr(_trueform.volume, f"make_mesh_sdf_{suffix}")
    flat, out_dims, out_spacing, out_origin = func(
        mesh._wrapper, dims, spacing, origin, _DTYPE_CODES[out_dtype],
        1 if mode == "banded" else 0, int(band)
    )
    return Volume._from_parts(flat, out_dims, out_spacing, out_origin)
