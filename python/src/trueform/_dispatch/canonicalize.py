"""
Index type canonicalization for form-form operations.

C++ implements: int x int, int x int64, int64 x int64
If Python passes int64 x int32, we swap to int32 x int64.

This preserves symmetry while ensuring we call the right C++ function.

Copyright (c) 2025 Ziga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""
import numpy as np
from typing import Tuple, Any


def canonicalize_index_order(form0: Any, form1: Any) -> Tuple[Any, Any, bool]:
    """
    Ensure int32 comes before int64 for same-type forms.

    C++ only implements certain index type orderings. This function
    swaps forms if needed to match the C++ implementation.

    Parameters
    ----------
    form0 : Mesh or EdgeMesh
        First spatial form
    form1 : Mesh or EdgeMesh
        Second spatial form

    Returns
    -------
    form0 : Any
        First form (possibly swapped)
    form1 : Any
        Second form (possibly swapped)
    was_swapped : bool
        True if forms were swapped. Caller must handle result swapping
        (e.g., swap columns in result array, flip labels).
    """
    # Import locally to avoid cycles
    from .._spatial import Mesh, EdgeMesh

    form0_type = type(form0)
    form1_type = type(form1)

    # Only canonicalize same-type forms
    if form0_type != form1_type:
        return form0, form1, False

    # Get index dtypes
    if form0_type is Mesh:
        faces0 = form0.faces
        faces1 = form1.faces
        idx0 = faces0.dtype if hasattr(faces0, 'dtype') else faces0.data.dtype
        idx1 = faces1.dtype if hasattr(faces1, 'dtype') else faces1.data.dtype
    elif form0_type is EdgeMesh:
        idx0 = form0.edges.dtype
        idx1 = form1.edges.dtype
    else:
        # PointCloud has no index type
        return form0, form1, False

    # Swap if int64 x int32 -> int32 x int64
    if idx0 == np.int64 and idx1 == np.int32:
        return form1, form0, True

    return form0, form1, False
