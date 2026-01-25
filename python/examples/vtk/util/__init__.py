"""
VTK utilities for trueform Python examples

Provides common functionality for VTK-based examples including:
- Geometry and mesh utilities
- Interactor base classes and helpers
- Scene setup and text rendering
- Performance timing utilities
"""

from .geometry import (
    MeshData,
    random_rotation_matrix,
    numpy_to_polydata,
    compute_centering_and_scaling_transform,
    curves_to_polydata,
    create_tube_filter,
    load_mesh,
    load_mesh_shared,
)

from .interaction import (
    BaseInteractor,
    get_camera_ray,
    NORMAL_COLOR,
    HIGHLIGHT_COLOR,
)

from .timing import (
    RollingAverage,
    format_time_us,
    format_time_ms,
)

from .scene import (
    create_text_actor,
    create_renderer_with_text_strip,
    setup_basic_scene,
)

from .cli import (
    create_parser,
)

__all__ = [
    # Geometry
    'MeshData',
    'random_rotation_matrix',
    'numpy_to_polydata',
    'compute_centering_and_scaling_transform',
    'curves_to_polydata',
    'create_tube_filter',
    'load_mesh',
    'load_mesh_shared',
    # Interaction
    'BaseInteractor',
    'get_camera_ray',
    'NORMAL_COLOR',
    'HIGHLIGHT_COLOR',
    # Timing
    'RollingAverage',
    'format_time_us',
    'format_time_ms',
    # Scene
    'create_text_actor',
    'create_renderer_with_text_strip',
    'setup_basic_scene',
    # CLI
    'create_parser',
]
