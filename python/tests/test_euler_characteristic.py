"""
Tests for euler_characteristic (V - E + F)

Copyright (c) 2025 Žiga Sajovic, XLAB
"""

import sys
import numpy as np
import pytest
import trueform as tf


INDEX_DTYPES = [np.int32, np.int64]
REAL_DTYPES = [np.float32, np.float64]


# ==============================================================================
# Helper Functions
# ==============================================================================

def create_torus_mesh(index_dtype, real_dtype, major=2.0, minor=0.5, n_u=48, n_v=24):
    """Create a triangulated torus (major radius around z, outward winding)."""
    u = np.linspace(0.0, 2.0 * np.pi, n_u, endpoint=False)
    v = np.linspace(0.0, 2.0 * np.pi, n_v, endpoint=False)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    ring = major + minor * np.cos(vv)
    points = np.stack(
        [ring * np.cos(uu), ring * np.sin(uu), minor * np.sin(vv)], axis=-1
    ).reshape(-1, 3).astype(real_dtype)
    faces = []
    for i in range(n_u):
        for j in range(n_v):
            a = i * n_v + j
            b = ((i + 1) % n_u) * n_v + j
            c = i * n_v + (j + 1) % n_v
            d = ((i + 1) % n_u) * n_v + (j + 1) % n_v
            faces.append([a, b, d])
            faces.append([a, d, c])
    return np.array(faces, dtype=index_dtype), points


# ==============================================================================
# Oracles: sphere 2, torus 0, triangle 1, open grid 1
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_euler_characteristic_sphere(index_dtype, real_dtype):
    faces, points = tf.make_sphere_mesh(
        1.0, stacks=16, segments=16, dtype=real_dtype, index_dtype=index_dtype
    )
    assert tf.euler_characteristic(tf.Mesh(faces, points)) == 2


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_euler_characteristic_torus(index_dtype, real_dtype):
    faces, points = create_torus_mesh(index_dtype, real_dtype)
    assert tf.euler_characteristic(tf.Mesh(faces, points)) == 0


def test_euler_characteristic_single_triangle():
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    assert tf.euler_characteristic(tf.Mesh(faces, points)) == 1


def test_euler_characteristic_open_grid():
    # An open disk-like patch: every boundary edge counted once.
    faces, points = tf.make_plane_mesh(1.0, 1.0, 4, 4)
    assert tf.euler_characteristic(tf.Mesh(faces, points)) == 1


# ==============================================================================
# Entry shapes: tuple and dynamic
# ==============================================================================

def test_euler_characteristic_accepts_tuple():
    faces, points = tf.make_sphere_mesh(1.0, stacks=12, segments=12)
    assert tf.euler_characteristic((faces, points)) == 2


def test_euler_characteristic_dynamic():
    faces, points = tf.make_sphere_mesh(1.0, stacks=12, segments=12)
    dyn = tf.as_offset_blocked(faces)
    assert tf.euler_characteristic((dyn, points)) == 2


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
