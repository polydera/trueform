"""
Cutting operations

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

from .isobands import isobands
from .boolean import boolean_union, boolean_intersection, boolean_difference
from .embedded_self_intersection_curves import embedded_self_intersection_curves

__all__ = [
    'isobands',
    'boolean_union',
    'boolean_intersection',
    'boolean_difference',
    'embedded_self_intersection_curves',
]
