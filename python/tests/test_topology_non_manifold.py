"""
Test non_manifold_edges

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
# Canonicalization helpers
# ==============================================================================

def canonicalize_edge(edge):
    """Canonicalize a single edge by sorting vertices."""
    return tuple(sorted(edge))


def canonicalize_edges(edges):
    """Canonicalize edges array for comparison (order-independent)."""
    if len(edges) == 0:
        return set()
    # Sort each edge, then return as set of tuples
    return set(canonicalize_edge(edge) for edge in edges)


# ==============================================================================
# Test data generators
# ==============================================================================

def create_two_triangles_mesh(index_dtype, real_dtype):
    """
    Create two triangles sharing one edge.

    Triangles share edge (1, 2), so 4 boundary edges remain.
    """
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


def create_closed_tetrahedron(index_dtype, real_dtype):
    """
    Create a closed tetrahedron (no boundary edges).

    4 triangles forming a closed surface.
    """
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
    """
    Create a mesh with a non-manifold edge.

    Three triangles sharing the same edge (0, 1).
    """
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


def create_dynamic_two_triangles(index_dtype, real_dtype):
    """Create two triangles as dynamic mesh."""
    offsets = np.array([0, 3, 6], dtype=index_dtype)
    data = np.array([0, 1, 2, 1, 3, 2], dtype=index_dtype)
    faces = tf.OffsetBlockedArray(offsets, data)
    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0],
        [1.5, 1.0, 0.0]
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
# non_manifold_edges Tests
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_non_manifold_edges_manifold_mesh(index_dtype, real_dtype):
    """Test non_manifold_edges for a manifold mesh."""
    faces, points = create_two_triangles_mesh(index_dtype, real_dtype)
    mesh = tf.Mesh(faces, points)

    nm_edges = tf.non_manifold_edges(mesh)

    # No non-manifold edges
    assert nm_edges.shape == (0, 2)
    assert nm_edges.dtype == index_dtype


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_non_manifold_edges_with_non_manifold(index_dtype, real_dtype):
    """Test non_manifold_edges for a mesh with non-manifold edges."""
    faces, points = create_non_manifold_mesh(index_dtype, real_dtype)
    mesh = tf.Mesh(faces, points)

    nm_edges = tf.non_manifold_edges(mesh)

    # One non-manifold edge (0, 1) shared by 3 faces
    assert nm_edges.shape == (1, 2)
    assert nm_edges.dtype == index_dtype

    # Check it's edge (0, 1)
    expected = {(0, 1)}
    result = canonicalize_edges(nm_edges)
    assert result == expected


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_non_manifold_edges_closed_mesh(index_dtype, real_dtype):
    """Test non_manifold_edges for a closed manifold mesh."""
    faces, points = create_closed_tetrahedron(index_dtype, real_dtype)
    mesh = tf.Mesh(faces, points)

    nm_edges = tf.non_manifold_edges(mesh)

    # No non-manifold edges
    assert nm_edges.shape == (0, 2)


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_non_manifold_edges_dynamic(index_dtype, real_dtype):
    """Test non_manifold_edges for a dynamic mesh (manifold)."""
    faces, points = create_dynamic_two_triangles(index_dtype, real_dtype)
    mesh = tf.Mesh(faces, points)

    nm_edges = tf.non_manifold_edges(mesh)

    # No non-manifold edges (manifold mesh)
    assert nm_edges.shape == (0, 2)
    assert nm_edges.dtype == index_dtype


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_non_manifold_edges_dynamic_with_non_manifold(index_dtype, real_dtype):
    """Test non_manifold_edges for a dynamic mesh with non-manifold edges."""
    faces, points = create_dynamic_non_manifold_mesh(index_dtype, real_dtype)
    mesh = tf.Mesh(faces, points)

    nm_edges = tf.non_manifold_edges(mesh)

    # One non-manifold edge (0, 1) shared by 3 faces
    assert nm_edges.shape == (1, 2)
    assert nm_edges.dtype == index_dtype

    expected = {(0, 1)}
    result = canonicalize_edges(nm_edges)
    assert result == expected


# ==============================================================================
# Error Validation Tests
# ==============================================================================

def test_non_manifold_edges_invalid_input():
    """Test non_manifold_edges with invalid input."""
    with pytest.raises(TypeError, match="mesh must be Mesh"):
        tf.non_manifold_edges("not a mesh")


def test_non_manifold_edges_edge_mesh():
    """Test non_manifold_edges with EdgeMesh (not supported)."""
    edges = np.array([[0, 1], [1, 2]], dtype=np.int32)
    points = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32)
    edge_mesh = tf.EdgeMesh(edges, points)

    with pytest.raises(TypeError, match="mesh must be Mesh"):
        tf.non_manifold_edges(edge_mesh)


# ==============================================================================
# Return Type Tests
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_non_manifold_edges_return_dtype(index_dtype):
    """Test that non_manifold_edges preserves index dtype."""
    faces, points = create_non_manifold_mesh(index_dtype, np.float32)
    mesh = tf.Mesh(faces, points)

    nm_edges = tf.non_manifold_edges(mesh)
    assert nm_edges.dtype == index_dtype


# ==============================================================================
# Main runner
# ==============================================================================

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
