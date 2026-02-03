"""
Trueform Blender add-on.

High-performance geometry processing tools including boolean operations
and intersection curves.
"""

bl_info = {
    "name": "Trueform",
    "author": "Z. Sajovic, M. Zukovec",
    "version": (2, 0, 0),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar > Trueform",
    "description": "High-performance geometry processing tools",
    "category": "Mesh",
}

from . import core
from . import tools


def register():
    core.register()
    tools.register()


def unregister():
    tools.unregister()
    core.unregister()


if __name__ == "__main__":
    register()
