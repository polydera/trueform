"""
Tests for OffsetBlockedArray

Copyright (c) 2025 Žiga Sajovic, XLAB
"""

import sys

import pytest
import numpy as np
import trueform as tf

# Parameter sets
INDEX_DTYPES = [np.int32, np.int64]


# ==============================================================================
# from_uniform Tests
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_from_uniform_quads(index_dtype):
    """Test creating OffsetBlockedArray from quad array."""
    quads = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=index_dtype)
    oba = tf.OffsetBlockedArray.from_uniform(quads)

    assert len(oba) == 2
    assert np.array_equal(oba.offsets, [0, 4, 8])
    assert np.array_equal(oba.data, [0, 1, 2, 3, 4, 5, 6, 7])
    assert oba.dtype == index_dtype


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_from_uniform_triangles(index_dtype):
    """Test creating OffsetBlockedArray from triangle array."""
    triangles = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=index_dtype)
    oba = tf.OffsetBlockedArray.from_uniform(triangles)

    assert len(oba) == 3
    assert np.array_equal(oba.offsets, [0, 3, 6, 9])
    assert np.array_equal(oba[0], [0, 1, 2])
    assert np.array_equal(oba[1], [3, 4, 5])
    assert np.array_equal(oba[2], [6, 7, 8])


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_from_uniform_pentagons(index_dtype):
    """Test creating OffsetBlockedArray from pentagon array."""
    pentagons = np.array([[0, 1, 2, 3, 4]], dtype=index_dtype)
    oba = tf.OffsetBlockedArray.from_uniform(pentagons)

    assert len(oba) == 1
    assert np.array_equal(oba.offsets, [0, 5])
    assert np.array_equal(oba[0], [0, 1, 2, 3, 4])


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_from_uniform_empty(index_dtype):
    """Test creating OffsetBlockedArray from empty array."""
    empty = np.array([], dtype=index_dtype).reshape(0, 4)
    oba = tf.OffsetBlockedArray.from_uniform(empty)

    assert len(oba) == 0
    assert np.array_equal(oba.offsets, [0])
    assert len(oba.data) == 0


# ==============================================================================
# from_uniform Error Tests
# ==============================================================================

def test_from_uniform_wrong_dtype():
    """Test that wrong dtype raises TypeError."""
    with pytest.raises(TypeError, match="dtype must be int32 or int64"):
        tf.OffsetBlockedArray.from_uniform(
            np.array([[0, 1, 2, 3]], dtype=np.float32)
        )


def test_from_uniform_wrong_ndim_1d():
    """Test that 1D array raises ValueError."""
    with pytest.raises(ValueError, match="Expected 2D array"):
        tf.OffsetBlockedArray.from_uniform(
            np.array([0, 1, 2, 3], dtype=np.int32)
        )


def test_from_uniform_wrong_ndim_3d():
    """Test that 3D array raises ValueError."""
    with pytest.raises(ValueError, match="Expected 2D array"):
        tf.OffsetBlockedArray.from_uniform(
            np.array([[[0, 1], [2, 3]]], dtype=np.int32)
        )


def test_from_uniform_wrong_type():
    """Test that non-ndarray raises TypeError."""
    with pytest.raises(TypeError, match="Expected np.ndarray"):
        tf.OffsetBlockedArray.from_uniform([[0, 1, 2, 3]])


# ==============================================================================
# as_offset_blocked Tests
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_as_offset_blocked_basic(index_dtype):
    """Test as_offset_blocked convenience function."""
    quads = np.array([[0, 1, 2, 3]], dtype=index_dtype)
    oba = tf.as_offset_blocked(quads)

    assert isinstance(oba, tf.OffsetBlockedArray)
    assert len(oba) == 1
    assert oba.dtype == index_dtype


def test_as_offset_blocked_equals_from_uniform():
    """Test that as_offset_blocked produces same result as from_uniform."""
    quads = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int32)

    oba1 = tf.as_offset_blocked(quads)
    oba2 = tf.OffsetBlockedArray.from_uniform(quads)

    assert np.array_equal(oba1.offsets, oba2.offsets)
    assert np.array_equal(oba1.data, oba2.data)


# ==============================================================================
# Mesh with from_uniform Tests
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", [np.float32, np.float64])
def test_mesh_quad_creation(index_dtype, real_dtype):
    """Test creating quad mesh using as_offset_blocked."""
    quads = np.array([[0, 1, 2, 3]], dtype=index_dtype)
    points = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=real_dtype
    )
    mesh = tf.Mesh(tf.as_offset_blocked(quads), points)

    assert mesh.is_dynamic is True
    assert mesh.ngon is None
    assert isinstance(mesh.faces, tf.OffsetBlockedArray)
    assert mesh.number_of_faces == 1
    assert mesh.number_of_points == 4


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", [np.float32, np.float64])
def test_mesh_quad_neighbor_search(index_dtype, real_dtype):
    """Test neighbor search on quad mesh."""
    quads = np.array([[0, 1, 2, 3]], dtype=index_dtype)
    points = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=real_dtype
    )
    mesh = tf.Mesh(tf.as_offset_blocked(quads), points)

    result = tf.neighbor_search(mesh, tf.Point([0.5, 0.5, 0.0]))
    assert result is not None


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", [np.float32, np.float64])
def test_mesh_quad_boundary_edges(index_dtype, real_dtype):
    """Test boundary edges on quad mesh."""
    quads = np.array([[0, 1, 2, 3]], dtype=index_dtype)
    points = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=real_dtype
    )
    mesh = tf.Mesh(tf.as_offset_blocked(quads), points)

    edges = tf.boundary_edges(mesh)
    assert edges.shape[0] == 4  # Quad has 4 boundary edges


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", [np.float32, np.float64])
def test_mesh_pentagon_creation(index_dtype, real_dtype):
    """Test creating pentagon mesh."""
    pentagons = np.array([[0, 1, 2, 3, 4]], dtype=index_dtype)
    # Pentagon points in 3D
    angles = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    points = np.column_stack([
        np.cos(angles),
        np.sin(angles),
        np.zeros(5)
    ]).astype(real_dtype)

    mesh = tf.Mesh(tf.as_offset_blocked(pentagons), points)

    assert mesh.is_dynamic is True
    assert mesh.number_of_faces == 1
    assert mesh.number_of_points == 5


# ==============================================================================
# Main runner
# ==============================================================================

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
