"""
IO utilities for reading and writing mesh files

Copyright (c) 2025 Ziga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

from .nifti import NiftiHeader, read_nifti, read_nifti_header, write_nifti
from .stl import read_stl, write_stl
from .obj import read_obj, write_obj

__all__ = ['NiftiHeader', 'read_nifti', 'read_nifti_header', 'write_nifti',
           'read_stl', 'write_stl', 'read_obj', 'write_obj']
