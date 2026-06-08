"""
Tests for error-budget simplification (tf.simplified)

Copyright (c) 2025 Žiga Sajovic, XLAB
"""

import numpy as np
import pytest
import trueform as tf


# Test parameters
REAL_DTYPES = [np.float32, np.float64]
INDEX_DTYPES = [np.int32, np.int64]


# ==============================================================================
# Basic behavior
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_sphere_error_budget_reduces_faces(dtype, index_dtype):
    """Simplifying a sphere within an error budget reduces the face count."""
    faces, points = tf.make_sphere_mesh(
        1.0, 20, 20, dtype=dtype, index_dtype=index_dtype
    )
    orig_faces = faces.shape[0]

    s_faces, s_points = tf.simplified((faces, points), error_rel=0.01)

    assert s_faces.shape[0] < orig_faces, "face count should decrease"
    assert s_faces.shape[0] > 0, "should have faces"
    assert s_points.shape[0] > 0, "should have points"
    assert s_faces.shape[1] == 3, "faces must be triangles"
    assert s_points.shape[1] == 3, "points must be 3D"

    # Output dtypes follow the input mesh dtypes.
    assert s_faces.dtype == index_dtype
    assert s_points.dtype == dtype


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_sphere_volume_preserved(dtype, index_dtype):
    """Error-budget simplification keeps curved detail, so volume is preserved."""
    faces, points = tf.make_sphere_mesh(
        1.0, 20, 20, dtype=dtype, index_dtype=index_dtype
    )
    orig_vol = tf.volume((faces, points))

    s_faces, s_points = tf.simplified((faces, points), error_rel=0.01)
    new_vol = tf.volume((s_faces, s_points))

    ratio = new_vol / orig_vol
    assert 0.5 < ratio < 1.5, f"volume ratio {ratio:.3f} should be near 1"


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_box_flat_regions_collapse(dtype, index_dtype):
    """Flat box faces collapse for ~0 quadric error and never increase count."""
    faces, points = tf.make_box_mesh(
        2, 3, 4, 4, 4, 4, dtype=dtype, index_dtype=index_dtype
    )
    orig_faces = faces.shape[0]

    s_faces, s_points = tf.simplified((faces, points))

    assert s_faces.shape[0] > 0, "should have at least 1 face"
    assert s_faces.shape[0] <= orig_faces, "face count should not increase"
    assert np.all(s_faces >= 0), "no negative indices"
    assert np.all(s_faces < s_points.shape[0]), "no out-of-range indices"


# ==============================================================================
# Options
# ==============================================================================

@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_with_options(dtype):
    """All keyword options are accepted and produce a valid mesh."""
    faces, points = tf.make_sphere_mesh(1.0, 15, 15, dtype=dtype)

    s_faces, s_points = tf.simplified(
        (faces, points),
        error_rel=0.005,
        optimize_iterations=2,
        min_quality=0.2,
        preserve_boundary=False,
        stabilizer=1e-3,
        parallel=True,
        feature_angle=30.0,
        feature_weight=100.0,
    )

    assert s_faces.shape[0] > 0, "should produce a valid mesh"
    assert s_faces.shape[0] <= faces.shape[0], "should not increase faces"


@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_default_config(dtype):
    """Simplifying with all defaults works."""
    faces, points = tf.make_sphere_mesh(1.0, 15, 15, dtype=dtype)
    s_faces, s_points = tf.simplified((faces, points))
    assert s_faces.shape[0] > 0
    assert s_points.shape[0] > 0


# ==============================================================================
# Input forms
# ==============================================================================

@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_accepts_mesh_and_tuple(dtype):
    """Both a Mesh and a (faces, points) tuple are accepted.

    No equivalence check between the two: half-edge construction and the
    collapse both run in parallel, so ordering (and thus the exact output)
    differs run-to-run. We only assert each form yields a valid triangle mesh.
    """
    faces, points = tf.make_sphere_mesh(1.0, 15, 15, dtype=dtype)

    for s_faces, s_points in (
        tf.simplified((faces, points), error_rel=0.01),
        tf.simplified(tf.Mesh(faces, points), error_rel=0.01),
    ):
        assert s_faces.shape[0] > 0, "should have faces"
        assert s_faces.shape[1] == 3, "faces must be triangles"
        assert s_points.shape[1] == 3, "points must be 3D"


# ==============================================================================
# Validation
# ==============================================================================

def test_rejects_non_mesh():
    """A non-Mesh, non-tuple input raises TypeError."""
    with pytest.raises(TypeError):
        tf.simplified(42)


def test_rejects_non_triangle_mesh():
    """A quad (ngon != 3) mesh cannot be simplified.

    A fixed-size Mesh already rejects non-triangle faces at construction, so
    the ValueError surfaces no later than the Mesh build either way.
    """
    quad_points = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32
    )
    quad_faces = np.array([[0, 1, 2, 3]], dtype=np.int32)
    with pytest.raises(ValueError):
        tf.simplified(tf.Mesh(quad_faces, quad_points))


def test_rejects_non_3d_mesh():
    """A 2D mesh raises ValueError."""
    points_2d = np.array(
        [[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    mesh = tf.Mesh(faces, points_2d)
    with pytest.raises(ValueError):
        tf.simplified(mesh)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
