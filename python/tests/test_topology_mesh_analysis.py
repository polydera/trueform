"""
Test is_closed, is_open, is_manifold, is_non_manifold

Copyright (c) 2025 Ziga Sajovic, XLAB
"""

import sys

import pytest
import numpy as np
import trueform as tf

# Parameter sets
INDEX_DTYPES = [np.int32, np.int64]
REAL_DTYPES = [np.float32, np.float64]


# ==============================================================================
# Test data generators
# ==============================================================================

def create_single_triangle(index_dtype, real_dtype):
    """Create a single triangle (open mesh)."""
    faces = np.array([
        [0, 1, 2]
    ], dtype=index_dtype)
    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0]
    ], dtype=real_dtype)
    return faces, points


def create_two_triangles(index_dtype, real_dtype):
    """Create two triangles sharing one edge (open, manifold)."""
    faces = np.array([
        [0, 1, 2],
        [1, 3, 2]
    ], dtype=index_dtype)
    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0],
        [1.5, 1.0, 0.0]
    ], dtype=real_dtype)
    return faces, points


def create_tetrahedron(index_dtype, real_dtype):
    """Create a closed tetrahedron (closed, manifold)."""
    faces = np.array([
        [0, 1, 2],
        [0, 2, 3],
        [0, 3, 1],
        [1, 3, 2]
    ], dtype=index_dtype)
    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0],
        [0.5, 0.5, 1.0]
    ], dtype=real_dtype)
    return faces, points


def create_non_manifold_mesh(index_dtype, real_dtype):
    """Create a mesh with a non-manifold edge (3 triangles sharing edge 0-1)."""
    faces = np.array([
        [0, 1, 2],
        [0, 1, 3],
        [0, 1, 4]
    ], dtype=index_dtype)
    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0],
        [0.5, -1.0, 0.0],
        [0.5, 0.0, 1.0]
    ], dtype=real_dtype)
    return faces, points


def create_dynamic_single_triangle(index_dtype, real_dtype):
    """Create a single triangle as dynamic mesh."""
    offsets = np.array([0, 3], dtype=index_dtype)
    data = np.array([0, 1, 2], dtype=index_dtype)
    faces = tf.OffsetBlockedArray(offsets, data)
    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0]
    ], dtype=real_dtype)
    return faces, points


def create_dynamic_tetrahedron(index_dtype, real_dtype):
    """Create a closed tetrahedron as dynamic mesh."""
    offsets = np.array([0, 3, 6, 9, 12], dtype=index_dtype)
    data = np.array([0, 1, 2, 0, 2, 3, 0, 3, 1, 1, 3, 2], dtype=index_dtype)
    faces = tf.OffsetBlockedArray(offsets, data)
    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0],
        [0.5, 0.5, 1.0]
    ], dtype=real_dtype)
    return faces, points


def create_dynamic_non_manifold_mesh(index_dtype, real_dtype):
    """Create a dynamic mesh with a non-manifold edge."""
    offsets = np.array([0, 3, 6, 9], dtype=index_dtype)
    data = np.array([0, 1, 2, 0, 1, 3, 0, 1, 4], dtype=index_dtype)
    faces = tf.OffsetBlockedArray(offsets, data)
    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0],
        [0.5, -1.0, 0.0],
        [0.5, 0.0, 1.0]
    ], dtype=real_dtype)
    return faces, points


# ==============================================================================
# is_closed Tests
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_is_closed_single_triangle(index_dtype, real_dtype):
    """Single triangle is open (not closed)."""
    faces, points = create_single_triangle(index_dtype, real_dtype)
    mesh = tf.Mesh(faces, points)
    assert tf.is_closed(mesh) is False


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_is_closed_two_triangles(index_dtype, real_dtype):
    """Two triangles sharing an edge is open."""
    faces, points = create_two_triangles(index_dtype, real_dtype)
    mesh = tf.Mesh(faces, points)
    assert tf.is_closed(mesh) is False


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_is_closed_tetrahedron(index_dtype, real_dtype):
    """Tetrahedron is closed."""
    faces, points = create_tetrahedron(index_dtype, real_dtype)
    mesh = tf.Mesh(faces, points)
    assert tf.is_closed(mesh) is True


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_is_closed_dynamic_single_triangle(index_dtype, real_dtype):
    """Dynamic single triangle is open."""
    faces, points = create_dynamic_single_triangle(index_dtype, real_dtype)
    mesh = tf.Mesh(faces, points)
    assert tf.is_closed(mesh) is False


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_is_closed_dynamic_tetrahedron(index_dtype, real_dtype):
    """Dynamic tetrahedron is closed."""
    faces, points = create_dynamic_tetrahedron(index_dtype, real_dtype)
    mesh = tf.Mesh(faces, points)
    assert tf.is_closed(mesh) is True


# ==============================================================================
# is_open Tests
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_is_open_single_triangle(index_dtype, real_dtype):
    """Single triangle is open."""
    faces, points = create_single_triangle(index_dtype, real_dtype)
    mesh = tf.Mesh(faces, points)
    assert tf.is_open(mesh) is True


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_is_open_tetrahedron(index_dtype, real_dtype):
    """Tetrahedron is not open (it's closed)."""
    faces, points = create_tetrahedron(index_dtype, real_dtype)
    mesh = tf.Mesh(faces, points)
    assert tf.is_open(mesh) is False


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_is_open_dynamic_single_triangle(index_dtype, real_dtype):
    """Dynamic single triangle is open."""
    faces, points = create_dynamic_single_triangle(index_dtype, real_dtype)
    mesh = tf.Mesh(faces, points)
    assert tf.is_open(mesh) is True


# ==============================================================================
# is_manifold Tests
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_is_manifold_single_triangle(index_dtype, real_dtype):
    """Single triangle is manifold."""
    faces, points = create_single_triangle(index_dtype, real_dtype)
    mesh = tf.Mesh(faces, points)
    assert tf.is_manifold(mesh) is True


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_is_manifold_two_triangles(index_dtype, real_dtype):
    """Two triangles sharing an edge is manifold."""
    faces, points = create_two_triangles(index_dtype, real_dtype)
    mesh = tf.Mesh(faces, points)
    assert tf.is_manifold(mesh) is True


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_is_manifold_tetrahedron(index_dtype, real_dtype):
    """Tetrahedron is manifold."""
    faces, points = create_tetrahedron(index_dtype, real_dtype)
    mesh = tf.Mesh(faces, points)
    assert tf.is_manifold(mesh) is True


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_is_manifold_non_manifold_mesh(index_dtype, real_dtype):
    """Three triangles sharing an edge is non-manifold."""
    faces, points = create_non_manifold_mesh(index_dtype, real_dtype)
    mesh = tf.Mesh(faces, points)
    assert tf.is_manifold(mesh) is False


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_is_manifold_dynamic_two_triangles(index_dtype, real_dtype):
    """Dynamic two triangles is manifold."""
    offsets = np.array([0, 3, 6], dtype=index_dtype)
    data = np.array([0, 1, 2, 1, 3, 2], dtype=index_dtype)
    faces = tf.OffsetBlockedArray(offsets, data)
    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0],
        [1.5, 1.0, 0.0]
    ], dtype=real_dtype)
    mesh = tf.Mesh(faces, points)
    assert tf.is_manifold(mesh) is True


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_is_manifold_dynamic_non_manifold(index_dtype, real_dtype):
    """Dynamic non-manifold mesh is not manifold."""
    faces, points = create_dynamic_non_manifold_mesh(index_dtype, real_dtype)
    mesh = tf.Mesh(faces, points)
    assert tf.is_manifold(mesh) is False


# ==============================================================================
# is_non_manifold Tests
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_is_non_manifold_manifold_mesh(index_dtype, real_dtype):
    """Manifold mesh is not non-manifold."""
    faces, points = create_two_triangles(index_dtype, real_dtype)
    mesh = tf.Mesh(faces, points)
    assert tf.is_non_manifold(mesh) is False


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_is_non_manifold_non_manifold_mesh(index_dtype, real_dtype):
    """Non-manifold mesh is non-manifold."""
    faces, points = create_non_manifold_mesh(index_dtype, real_dtype)
    mesh = tf.Mesh(faces, points)
    assert tf.is_non_manifold(mesh) is True


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_is_non_manifold_tetrahedron(index_dtype, real_dtype):
    """Tetrahedron is not non-manifold."""
    faces, points = create_tetrahedron(index_dtype, real_dtype)
    mesh = tf.Mesh(faces, points)
    assert tf.is_non_manifold(mesh) is False


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_is_non_manifold_dynamic_non_manifold(index_dtype, real_dtype):
    """Dynamic non-manifold mesh is non-manifold."""
    faces, points = create_dynamic_non_manifold_mesh(index_dtype, real_dtype)
    mesh = tf.Mesh(faces, points)
    assert tf.is_non_manifold(mesh) is True


# ==============================================================================
# Consistency Tests
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_is_closed_is_open_consistency(index_dtype, real_dtype):
    """is_closed and is_open are inverses."""
    faces, points = create_tetrahedron(index_dtype, real_dtype)
    mesh = tf.Mesh(faces, points)
    assert tf.is_closed(mesh) != tf.is_open(mesh)

    faces, points = create_single_triangle(index_dtype, real_dtype)
    mesh = tf.Mesh(faces, points)
    assert tf.is_closed(mesh) != tf.is_open(mesh)


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_is_manifold_is_non_manifold_consistency(index_dtype, real_dtype):
    """is_manifold and is_non_manifold are inverses."""
    faces, points = create_two_triangles(index_dtype, real_dtype)
    mesh = tf.Mesh(faces, points)
    assert tf.is_manifold(mesh) != tf.is_non_manifold(mesh)

    faces, points = create_non_manifold_mesh(index_dtype, real_dtype)
    mesh = tf.Mesh(faces, points)
    assert tf.is_manifold(mesh) != tf.is_non_manifold(mesh)


# ==============================================================================
# Error Validation Tests
# ==============================================================================

def test_is_closed_invalid_input():
    """Test is_closed with invalid input."""
    with pytest.raises(TypeError, match="mesh must be Mesh"):
        tf.is_closed("not a mesh")


def test_is_open_invalid_input():
    """Test is_open with invalid input."""
    with pytest.raises(TypeError, match="mesh must be Mesh"):
        tf.is_open("not a mesh")


def test_is_manifold_invalid_input():
    """Test is_manifold with invalid input."""
    with pytest.raises(TypeError, match="mesh must be Mesh"):
        tf.is_manifold("not a mesh")


def test_is_non_manifold_invalid_input():
    """Test is_non_manifold with invalid input."""
    with pytest.raises(TypeError, match="mesh must be Mesh"):
        tf.is_non_manifold("not a mesh")


def test_is_closed_edge_mesh():
    """Test is_closed with EdgeMesh (not supported)."""
    edges = np.array([[0, 1], [1, 2]], dtype=np.int32)
    points = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32)
    edge_mesh = tf.EdgeMesh(edges, points)

    with pytest.raises(TypeError, match="mesh must be Mesh"):
        tf.is_closed(edge_mesh)


# ==============================================================================
# Main runner
# ==============================================================================

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
