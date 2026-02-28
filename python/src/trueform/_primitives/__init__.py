"""
Geometric primitives for trueform

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

from .primitive import Primitive, PrimitiveType
from .point import Point
from .segment import Segment
from .triangle import Triangle
from .polygon import Polygon
from .aabb import AABB
from .ray import Ray
from .line import Line
from .plane import Plane

__all__ = ['Primitive', 'PrimitiveType', 'Point', 'Segment', 'Triangle', 'Polygon', 'AABB', 'Ray', 'Line', 'Plane']
