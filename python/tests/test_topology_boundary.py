"""
Test boundary_edges, boundary_paths, boundary_curves

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


def canonicalize_path(path):
    """
    Canonicalize a closed path (loop) for comparison.

    Handles closed paths where first == last by stripping the duplicate.
    Finds canonical rotation and direction by:
    1. Finding minimum vertex
    2. Rotating to start at minimum
    3. Choosing direction that gives lexicographically smaller sequence
    """
    if len(path) == 0:
        return tuple()

    path = list(path)

    # Strip last element if path is closed (first == last)
    if len(path) > 1 and path[0] == path[-1]:
        path = path[:-1]

    n = len(path)
    if n == 0:
        return tuple()

    # Find all positions of the minimum element
    min_val = min(path)
    min_positions = [i for i, v in enumerate(path) if v == min_val]

    # Try all rotations starting from min positions, both directions
    candidates = []
    for pos in min_positions:
        # Forward direction
        forward = tuple(path[(pos + i) % n] for i in range(n))
        candidates.append(forward)

        # Reverse direction (starting from pos, going backwards)
        reverse = tuple(path[(pos - i) % n] for i in range(n))
        candidates.append(reverse)

    # Return lexicographically smallest
    return min(candidates)


def canonicalize_paths(paths):
    """Canonicalize all paths in an OffsetBlockedArray for comparison."""
    canonical = []
    for path in paths:
        canonical.append(canonicalize_path(path))
    # Sort paths for order-independent comparison
    return sorted(canonical)


# ==============================================================================
# Test data generators
# ==============================================================================

def create_open_triangle_mesh(index_dtype, real_dtype):
    """
    Create a single triangle (all edges are boundary).

    Triangle: vertices 0, 1, 2
    All 3 edges are boundary edges.
    """
    faces = np.array([
        [0, 1, 2]
    ], dtype=index_dtype)
    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0]
    ], dtype=real_dtype)
    return faces, points


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


def create_mesh_with_hole(index_dtype, real_dtype):
    """
    Create a mesh with a hole (two boundary loops).

    A square with a triangular hole in the middle.
    Outer boundary: 4 edges
    Inner boundary (hole): 3 edges
    """
    # Outer square: 0, 1, 2, 3
    # Inner triangle hole: 4, 5, 6
    faces = np.array([
        # Triangles connecting outer to inner
        [0, 1, 4],
        [1, 5, 4],
        [1, 2, 5],
        [2, 6, 5],
        [2, 3, 6],
        [3, 4, 6],
        [3, 0, 4]
    ], dtype=index_dtype)
    points = np.array([
        # Outer square
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        # Inner triangle
        [0.3, 0.3, 0.0],
        [0.7, 0.3, 0.0],
        [0.5, 0.7, 0.0]
    ], dtype=real_dtype)
    return faces, points


def create_dynamic_mesh_open(index_dtype, real_dtype):
    """Create an open dynamic mesh (single triangle as dynamic)."""
    offsets = np.array([0, 3], dtype=index_dtype)
    data = np.array([0, 1, 2], dtype=index_dtype)
    faces = tf.OffsetBlockedArray(offsets, data)
    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0]
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


# ==============================================================================
# boundary_edges Tests
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_boundary_edges_single_triangle(index_dtype, real_dtype):
    """Test boundary_edges for a single triangle (all edges are boundary)."""
    faces, points = create_open_triangle_mesh(index_dtype, real_dtype)
    mesh = tf.Mesh(faces, points)

    edges = tf.boundary_edges(mesh)

    # Check return type and shape
    assert isinstance(edges, np.ndarray)
    assert edges.shape == (3, 2)  # 3 boundary edges
    assert edges.dtype == index_dtype

    # Check all 3 edges are present
    expected = {(0, 1), (1, 2), (0, 2)}
    result = canonicalize_edges(edges)
    assert result == expected


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_boundary_edges_two_triangles(index_dtype, real_dtype):
    """Test boundary_edges for two triangles sharing an edge."""
    faces, points = create_two_triangles_mesh(index_dtype, real_dtype)
    mesh = tf.Mesh(faces, points)

    edges = tf.boundary_edges(mesh)

    # 4 boundary edges (shared edge is not boundary)
    assert edges.shape == (4, 2)
    assert edges.dtype == index_dtype

    # Expected boundary edges
    expected = {(0, 1), (0, 2), (1, 3), (2, 3)}
    result = canonicalize_edges(edges)
    assert result == expected


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_boundary_edges_closed_mesh(index_dtype, real_dtype):
    """Test boundary_edges for a closed mesh (no boundaries)."""
    faces, points = create_closed_tetrahedron(index_dtype, real_dtype)
    mesh = tf.Mesh(faces, points)

    edges = tf.boundary_edges(mesh)

    # No boundary edges
    assert edges.shape == (0, 2)
    assert edges.dtype == index_dtype


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_boundary_edges_dynamic(index_dtype, real_dtype):
    """Test boundary_edges for a dynamic mesh."""
    faces, points = create_dynamic_mesh_open(index_dtype, real_dtype)
    mesh = tf.Mesh(faces, points)

    edges = tf.boundary_edges(mesh)

    # 3 boundary edges for a single triangle
    assert edges.shape == (3, 2)
    assert edges.dtype == index_dtype

    expected = {(0, 1), (1, 2), (0, 2)}
    result = canonicalize_edges(edges)
    assert result == expected


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_boundary_edges_dynamic_two_triangles(index_dtype, real_dtype):
    """Test boundary_edges for dynamic mesh with two triangles."""
    faces, points = create_dynamic_two_triangles(index_dtype, real_dtype)
    mesh = tf.Mesh(faces, points)

    edges = tf.boundary_edges(mesh)

    # 4 boundary edges (shared edge is not boundary)
    assert edges.shape == (4, 2)
    assert edges.dtype == index_dtype

    expected = {(0, 1), (0, 2), (1, 3), (2, 3)}
    result = canonicalize_edges(edges)
    assert result == expected


# ==============================================================================
# boundary_paths Tests
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_boundary_paths_single_triangle(index_dtype, real_dtype):
    """Test boundary_paths for a single triangle."""
    faces, points = create_open_triangle_mesh(index_dtype, real_dtype)
    mesh = tf.Mesh(faces, points)

    paths = tf.boundary_paths(mesh)

    # Check return type
    assert isinstance(paths, tf.OffsetBlockedArray)

    # Single boundary loop with 4 vertices (closed: first == last)
    assert len(paths) == 1
    assert len(paths[0]) == 4
    assert paths[0][0] == paths[0][-1], "Closed path should have first == last"

    # Check that path contains all 3 unique vertices
    path_set = set(paths[0])
    assert path_set == {0, 1, 2}


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_boundary_paths_two_triangles(index_dtype, real_dtype):
    """Test boundary_paths for two triangles sharing an edge."""
    faces, points = create_two_triangles_mesh(index_dtype, real_dtype)
    mesh = tf.Mesh(faces, points)

    paths = tf.boundary_paths(mesh)

    # Single boundary loop with 5 vertices (closed: first == last)
    assert len(paths) == 1
    assert len(paths[0]) == 5
    assert paths[0][0] == paths[0][-1], "Closed path should have first == last"

    # Check that path contains all 4 unique vertices
    path_set = set(paths[0])
    assert path_set == {0, 1, 2, 3}


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_boundary_paths_closed_mesh(index_dtype, real_dtype):
    """Test boundary_paths for a closed mesh (no boundaries)."""
    faces, points = create_closed_tetrahedron(index_dtype, real_dtype)
    mesh = tf.Mesh(faces, points)

    paths = tf.boundary_paths(mesh)

    # No boundary paths
    assert len(paths) == 0
    assert len(paths.data) == 0


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_boundary_paths_dynamic(index_dtype, real_dtype):
    """Test boundary_paths for a dynamic mesh."""
    faces, points = create_dynamic_mesh_open(index_dtype, real_dtype)
    mesh = tf.Mesh(faces, points)

    paths = tf.boundary_paths(mesh)

    # Single boundary loop with 4 vertices (closed: first == last)
    assert len(paths) == 1
    assert len(paths[0]) == 4
    assert paths[0][0] == paths[0][-1], "Closed path should have first == last"

    path_set = set(paths[0])
    assert path_set == {0, 1, 2}


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_boundary_paths_dynamic_two_triangles(index_dtype, real_dtype):
    """Test boundary_paths for dynamic mesh with two triangles."""
    faces, points = create_dynamic_two_triangles(index_dtype, real_dtype)
    mesh = tf.Mesh(faces, points)

    paths = tf.boundary_paths(mesh)

    # Single boundary loop with 5 vertices (closed: first == last)
    assert len(paths) == 1
    assert len(paths[0]) == 5
    assert paths[0][0] == paths[0][-1], "Closed path should have first == last"

    path_set = set(paths[0])
    assert path_set == {0, 1, 2, 3}


# ==============================================================================
# boundary_curves Tests
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_boundary_curves_single_triangle(index_dtype, real_dtype):
    """Test boundary_curves for a single triangle."""
    faces, points = create_open_triangle_mesh(index_dtype, real_dtype)
    mesh = tf.Mesh(faces, points)

    curves, curve_points = tf.boundary_curves(mesh)

    # Check return types
    assert isinstance(curves, tf.OffsetBlockedArray)
    assert isinstance(curve_points, np.ndarray)

    # Single curve with 4 vertices (closed: first == last)
    assert len(curves) == 1
    assert len(curves[0]) == 4
    assert curves[0][0] == curves[0][-1], "Closed curve should have first == last"

    # Curve points should have 3 unique points
    assert curve_points.shape == (3, 3)
    assert curve_points.dtype == real_dtype

    # Indices should be 0, 1, 2 (remapped)
    curve_indices = set(curves[0])
    assert curve_indices == {0, 1, 2}


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_boundary_curves_closed_mesh(index_dtype, real_dtype):
    """Test boundary_curves for a closed mesh (no boundaries)."""
    faces, points = create_closed_tetrahedron(index_dtype, real_dtype)
    mesh = tf.Mesh(faces, points)

    curves, curve_points = tf.boundary_curves(mesh)

    # No curves
    assert len(curves) == 0
    assert curve_points.shape == (0, 3)


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_boundary_curves_point_values(index_dtype, real_dtype):
    """Test that boundary_curves returns correct point coordinates."""
    faces, points = create_open_triangle_mesh(index_dtype, real_dtype)
    mesh = tf.Mesh(faces, points)

    curves, curve_points = tf.boundary_curves(mesh)

    # Get original boundary points
    paths = tf.boundary_paths(mesh)
    original_boundary_points = points[paths[0]]

    # Curve points should match (possibly in different order)
    original_set = set(map(tuple, original_boundary_points))
    curve_set = set(map(tuple, curve_points))
    assert original_set == curve_set


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_boundary_curves_dynamic(index_dtype, real_dtype):
    """Test boundary_curves for a dynamic mesh."""
    faces, points = create_dynamic_mesh_open(index_dtype, real_dtype)
    mesh = tf.Mesh(faces, points)

    curves, curve_points = tf.boundary_curves(mesh)

    # Check return types
    assert isinstance(curves, tf.OffsetBlockedArray)
    assert isinstance(curve_points, np.ndarray)

    # Single curve with 4 vertices (closed: first == last)
    assert len(curves) == 1
    assert len(curves[0]) == 4
    assert curves[0][0] == curves[0][-1], "Closed curve should have first == last"

    # Curve points should have 3 unique points
    assert curve_points.shape == (3, 3)
    assert curve_points.dtype == real_dtype


# ==============================================================================
# Error Validation Tests
# ==============================================================================

def test_boundary_edges_invalid_input():
    """Test boundary_edges with invalid input."""
    with pytest.raises(TypeError, match="mesh must be Mesh"):
        tf.boundary_edges("not a mesh")


def test_boundary_edges_edge_mesh():
    """Test boundary_edges with EdgeMesh (not supported)."""
    edges = np.array([[0, 1], [1, 2]], dtype=np.int32)
    points = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float32)
    edge_mesh = tf.EdgeMesh(edges, points)

    with pytest.raises(TypeError, match="mesh must be Mesh"):
        tf.boundary_edges(edge_mesh)


def test_boundary_paths_invalid_input():
    """Test boundary_paths with invalid input."""
    with pytest.raises(TypeError, match="mesh must be Mesh"):
        tf.boundary_paths("not a mesh")


def test_boundary_curves_invalid_input():
    """Test boundary_curves with invalid input."""
    with pytest.raises(TypeError, match="mesh must be Mesh"):
        tf.boundary_curves("not a mesh")


# ==============================================================================
# Return Type Tests
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_boundary_edges_return_dtype(index_dtype):
    """Test that boundary_edges preserves index dtype."""
    faces = np.array([[0, 1, 2]], dtype=index_dtype)
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    mesh = tf.Mesh(faces, points)

    edges = tf.boundary_edges(mesh)
    assert edges.dtype == index_dtype


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_boundary_paths_return_dtype(index_dtype):
    """Test that boundary_paths preserves index dtype."""
    faces = np.array([[0, 1, 2]], dtype=index_dtype)
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    mesh = tf.Mesh(faces, points)

    paths = tf.boundary_paths(mesh)
    assert paths.data.dtype == index_dtype


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_boundary_curves_return_dtypes(index_dtype, real_dtype):
    """Test that boundary_curves returns correct dtypes."""
    faces = np.array([[0, 1, 2]], dtype=index_dtype)
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=real_dtype)
    mesh = tf.Mesh(faces, points)

    curves, curve_points = tf.boundary_curves(mesh)
    assert curves.data.dtype == index_dtype
    assert curve_points.dtype == real_dtype


# ==============================================================================
# Main runner
# ==============================================================================

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
