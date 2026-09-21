"""
Intersection operations

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

from .intersection_curves import intersection_curves
from .self_intersection_curves import self_intersection_curves
from .has_self_intersections import has_self_intersections

__all__ = ['intersection_curves', 'self_intersection_curves',
           'has_self_intersections']
