"""
k_rings function implementation

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

from .. import _trueform
from .._core import OffsetBlockedArray
from .._dispatch import topology_suffix


def k_rings(
    connectivity: OffsetBlockedArray,
    k: int,
    inclusive: bool = False
) -> OffsetBlockedArray:
    """
    Compute k-ring neighborhoods for all vertices.

    For each vertex, computes all vertices reachable within k hops along
    mesh edges using breadth-first traversal. The 1-ring is the immediate
    neighbors, 2-ring includes neighbors of neighbors, etc.

    Parameters
    ----------
    connectivity : OffsetBlockedArray
        Vertex connectivity (1-ring) from vertex_link_edges or vertex_link_faces.

    k : int
        Number of hops (rings) to traverse.

    inclusive : bool, optional
        If True, include the seed vertex in its own neighborhood.
        Default is False.

    Returns
    -------
    OffsetBlockedArray
        K-ring neighborhoods where block i contains the indices of all vertices
        reachable from vertex i within k hops.

    Examples
    --------
    >>> import trueform as tf
    >>> import numpy as np
    >>>
    >>> # Triangle mesh
    >>> faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
    >>> n_ids = 4
    >>>
    >>> # Build connectivity
    >>> fm = tf.cell_membership(faces, n_ids)
    >>> connectivity = tf.vertex_link_faces(faces, fm)
    >>>
    >>> # Compute 2-ring neighborhoods
    >>> k2 = tf.k_rings(connectivity, k=2)
    >>> # k2[0] contains all vertices within 2 hops of vertex 0
    """

    # ===== VALIDATE connectivity =====
    if not isinstance(connectivity, OffsetBlockedArray):
        raise TypeError(
            f"connectivity must be OffsetBlockedArray, "
            f"got {type(connectivity).__name__}"
        )

    # ===== VALIDATE k =====
    if not isinstance(k, int) or k < 1:
        raise ValueError(f"k must be a positive integer, got {k}")

    # ===== BUILD SUFFIX AND DISPATCH =====
    suffix = topology_suffix(connectivity.offsets.dtype)
    func_name = f"make_k_rings_{suffix}"
    cpp_func = getattr(_trueform.topology, func_name)

    # Call C++ function
    wrapper = cpp_func(connectivity._wrapper, k, inclusive)

    return OffsetBlockedArray(wrapper.offsets_array(), wrapper.data_array())
