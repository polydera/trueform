"""
Core data structures for trueform

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

from .closest_metric_point_pair import closest_metric_point_pair, closest_metric_point
from .offset_blocked_array import OffsetBlockedArray
from .as_offset_blocked import as_offset_blocked


__all__ = [
    'closest_metric_point_pair',
    'closest_metric_point',
    'OffsetBlockedArray',
    'as_offset_blocked',
]
