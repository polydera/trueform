"""Trueform tools package."""

from . import boolean
from . import curves


def register():
    boolean.register()
    curves.register()


def unregister():
    curves.unregister()
    boolean.unregister()
