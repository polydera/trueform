"""
Test cleaned functionality

Copyright (c) 2025 Žiga Sajovic, XLAB
"""

import sys

import pytest
import numpy as np
import trueform as tf

# Parameter sets
INDEX_DTYPES = [np.int32, np.int64]
REAL_DTYPES = [np.float32, np.float64]
DIMS = [2, 3]
NGONS = [3]  # Only triangles for fixed-size (dynamic for variable-sized)
SOUP_VERTICES = [2, 3]  # segments, triangles (no quads for soups)

# ==============================================================================
# Canonicalization helpers
# ==============================================================================

def canonicalize_points(points):
    """
    Sort points lexicographically for comparison.

    Parameters
    ----------
    points : np.ndarray
        Shape (N, Dims)

    Returns
    -------
    canonical_points : np.ndarray
        Points sorted lexicographically
    """
    # Convert to tuples for lexicographic sorting
    point_tuples = [tuple(p) for p in points]
    sort_order = sorted(range(len(point_tuples)), key=lambda i: point_tuples[i])
    return points[sort_order]


def canonicalize_edges(edges, points):
    """
    Create canonical ordering of edge mesh for comparison.

    1. For each edge, order the two point indices (smaller first)
    2. Sort edges lexicographically by their point coordinates

    Parameters
    ----------
    edges : np.ndarray
        Shape (N, 2) - edge connectivity
    points : np.ndarray
        Shape (M, Dims) - point coordinates

    Returns
    -------
    canonical_edges : np.ndarray
        Canonicalized edges
    points : np.ndarray
        Points (unchanged)
    """
    canonical_edges = []

    for edge in edges:
        # Order indices so smaller is first
        ordered_edge = np.sort(edge)
        canonical_edges.append(ordered_edge)

    canonical_edges = np.array(canonical_edges)

    # Sort edges by their point coordinates
    sort_keys = []
    for edge in canonical_edges:
        edge_points = points[edge]
        # Flatten to single tuple for sorting
        sort_key = tuple(edge_points.flatten())
        sort_keys.append(sort_key)

    # Get sort order
    sort_order = sorted(range(len(sort_keys)), key=lambda i: sort_keys[i])

    # Apply sort order
    canonical_edges = canonical_edges[sort_order]

    return canonical_edges, points


def canonicalize_faces(faces, points):
    """
    Create canonical ordering of mesh for comparison.

    1. For each face, rotate indices to start with the lexicographically smallest point
    2. Sort faces lexicographically by their point coordinates

    Parameters
    ----------
    faces : np.ndarray
        Shape (N, V) where V = 3 or 4 - face connectivity
    points : np.ndarray
        Shape (M, Dims) - point coordinates

    Returns
    -------
    canonical_faces : np.ndarray
        Canonicalized faces
    points : np.ndarray
        Points (unchanged)
    """
    V = faces.shape[1]
    canonical_faces = []

    for face in faces:
        # Get the points of this face
        face_points = points[face]

        # Find the index (0, 1, ..., V-1) of the lexicographically smallest point
        point_tuples = [tuple(p) for p in face_points]
        min_idx = min(range(V), key=lambda i: point_tuples[i])

        # Rotate face so it starts with the smallest point
        rotated_face = np.roll(face, -min_idx)
        canonical_faces.append(rotated_face)

    canonical_faces = np.array(canonical_faces)

    # Sort faces by their point coordinates
    sort_keys = []
    for face in canonical_faces:
        face_points = points[face]
        # Flatten to single tuple for sorting
        sort_key = tuple(face_points.flatten())
        sort_keys.append(sort_key)

    # Get sort order
    sort_order = sorted(range(len(sort_keys)), key=lambda i: sort_keys[i])

    # Apply sort order
    canonical_faces = canonical_faces[sort_order]

    return canonical_faces, points


def validate_index_map(f, kept_ids, original_count):
    """
    Validate the index map returned by cleaned().

    Parameters
    ----------
    f : np.ndarray
        Maps old id to new id (f[i] == len(kept_ids) means removed)
    kept_ids : np.ndarray
        Array of kept old ids
    original_count : int
        Original number of elements
    """
    # f should map all original indices
    assert f.shape == (original_count,), \
        f"f should have shape ({original_count},), got {f.shape}"

    # kept_ids should be <= original count
    assert len(kept_ids) <= original_count, \
        f"Can't keep more than original count: {len(kept_ids)} > {original_count}"

    # All kept indices should map to valid new indices
    assert np.all(f[kept_ids] < len(kept_ids)), \
        "Kept IDs should map to valid new indices"

    # Check that kept_ids are within valid range
    assert np.all(kept_ids >= 0) and np.all(kept_ids < original_count), \
        "kept_ids should be within [0, original_count)"

# ==============================================================================
# Test data generators
# ==============================================================================

def create_points_with_duplicates(dims, dtype):
    """Create point set with known duplicates."""
    if dims == 2:
        # 6 points, indices 0 and 2 are duplicates, 3 and 5 are duplicates
        points = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 0.0],  # duplicate of 0
            [2.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.0],  # duplicate of 3
        ], dtype=dtype)
    else:  # dims == 3
        points = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],  # duplicate of 0
            [2.0, 1.0, 0.0],
            [1.0, 1.0, 0.5],
            [2.0, 1.0, 0.0],  # duplicate of 3
        ], dtype=dtype)

    return points


def create_points_with_tolerance_duplicates(dims, dtype):
    """Create points with near-duplicates within tolerance."""
    if dims == 2:
        # Points with pairs within 0.01 distance
        points = np.array([
            [0.0, 0.0],
            [0.001, 0.001],  # near 0
            [1.0, 0.0],
            [1.001, 0.0],  # near 2
            [2.0, 2.0],
        ], dtype=dtype)
    else:  # dims == 3
        points = np.array([
            [0.0, 0.0, 0.0],
            [0.001, 0.0, 0.0],  # near 0
            [1.0, 0.0, 0.0],
            [1.0, 0.001, 0.0],  # near 2
            [2.0, 2.0, 2.0],
        ], dtype=dtype)

    return points


def create_mesh_with_duplicates(dims, ngon, index_dtype, real_dtype):
    """Create mesh with duplicate vertices and potentially degenerate faces."""
    if dims == 2:
        # 2D mesh (planar)
        points = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, 1.0],
            [0.0, 0.0],  # duplicate of 0
            [1.5, 0.5],
        ], dtype=real_dtype)

        faces = np.array([
            [0, 1, 2],
            [3, 1, 4],  # uses duplicate point 3
        ], dtype=index_dtype)
    else:  # dims == 3
        points = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [0.0, 0.0, 0.0],  # duplicate of 0
            [1.5, 0.5, 0.5],
        ], dtype=real_dtype)

        faces = np.array([
            [0, 1, 2],
            [3, 1, 4],  # uses duplicate point 3
        ], dtype=index_dtype)

    return faces, points


def create_dynamic_mesh_with_duplicates(dims, index_dtype, real_dtype):
    """Create dynamic mesh with duplicate vertices and mixed polygon sizes."""
    if dims == 2:
        points = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, 1.0],
            [0.0, 0.0],  # duplicate of 0
            [2.0, 0.0],
            [1.5, 1.0],
        ], dtype=real_dtype)
    else:  # dims == 3
        points = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [0.0, 0.0, 0.0],  # duplicate of 0
            [2.0, 0.0, 0.0],
            [1.5, 1.0, 0.0],
        ], dtype=real_dtype)

    # Create dynamic faces: triangle + quad
    offsets = np.array([0, 3, 7], dtype=index_dtype)
    data = np.array([0, 1, 2, 3, 1, 4, 5], dtype=index_dtype)  # tri uses dup point 3
    faces = tf.OffsetBlockedArray(offsets, data)

    return faces, points


def create_edge_mesh_with_duplicates(dims, index_dtype, real_dtype):
    """Create edge mesh with duplicate vertices."""
    if dims == 2:
        points = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 0.0],  # duplicate of 0
            [1.0, 1.0],
        ], dtype=real_dtype)

        edges = np.array([
            [0, 1],
            [2, 3],  # uses duplicate point 2
        ], dtype=index_dtype)
    else:  # dims == 3
        points = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],  # duplicate of 0
            [1.0, 1.0, 0.0],
        ], dtype=real_dtype)

        edges = np.array([
            [0, 1],
            [2, 3],  # uses duplicate point 2
        ], dtype=index_dtype)

    return edges, points


def create_polygon_soup_with_duplicates(dims, V, dtype):
    """Create polygon soup with duplicate vertices."""
    if V == 2:  # Segments
        if dims == 2:
            soup = np.array([
                [[0.0, 0.0], [1.0, 0.0]],
                [[1.0, 0.0], [1.0, 1.0]],  # shares vertex [1, 0]
            ], dtype=dtype)
        else:  # dims == 3
            soup = np.array([
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],  # shares vertex [1, 0, 0]
            ], dtype=dtype)
    else:  # V == 3, Triangles
        if dims == 2:
            soup = np.array([
                [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]],
                [[1.0, 0.0], [2.0, 0.0], [1.5, 1.0]],  # shares vertex [1, 0]
            ], dtype=dtype)
        else:  # dims == 3
            soup = np.array([
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]],
                [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.5, 1.0, 0.0]],  # shares vertex
            ], dtype=dtype)

    return soup

# ==============================================================================
# Points Tests
# ==============================================================================

@pytest.mark.parametrize("dims", DIMS)
@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_cleaned_points_exact_duplicates(dims, dtype):
    """Test removing exact duplicate points."""
    points = create_points_with_duplicates(dims, dtype)

    # Clean without index map
    cleaned_points = tf.cleaned(points)

    # Should have 4 unique points (removed 2 duplicates)
    assert cleaned_points.shape[0] == 4, \
        f"Expected 4 unique points, got {cleaned_points.shape[0]}"
    assert cleaned_points.shape[1] == dims
    assert cleaned_points.dtype == dtype

    # Verify no duplicates remain
    for i in range(len(cleaned_points)):
        for j in range(i + 1, len(cleaned_points)):
            assert not np.allclose(cleaned_points[i], cleaned_points[j], atol=1e-10), \
                f"Found duplicate points at indices {i} and {j}"


@pytest.mark.parametrize("dims", DIMS)
@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_cleaned_points_with_tolerance(dims, dtype):
    """Test merging points within tolerance."""
    points = create_points_with_tolerance_duplicates(dims, dtype)

    # Clean with tolerance
    tolerance = 0.01 if dtype == np.float32 else 0.01
    cleaned_points = tf.cleaned(points, tolerance=tolerance)

    # Should merge near-duplicates: [0,1] merge, [2,3] merge, [4] stays
    # Result: 3 points
    assert cleaned_points.shape[0] == 3, \
        f"Expected 3 points after tolerance merging, got {cleaned_points.shape[0]}"
    assert cleaned_points.shape[1] == dims
    assert cleaned_points.dtype == dtype


@pytest.mark.parametrize("dims", DIMS)
@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_cleaned_points_with_index_map(dims, dtype):
    """Test index map correctness for points."""
    points = create_points_with_duplicates(dims, dtype)
    original_count = len(points)

    # Clean with index map
    cleaned_points, (f, kept_ids) = tf.cleaned(points, return_index_map=True)

    # Validate index map structure
    validate_index_map(f, kept_ids, original_count)

    # Should have 4 unique points
    assert len(kept_ids) == 4, f"Expected 4 kept points, got {len(kept_ids)}"
    assert cleaned_points.shape[0] == 4

    # Verify that reindexing works: cleaned_points should equal original[kept_ids]
    assert np.allclose(cleaned_points, points[kept_ids], atol=1e-10)


@pytest.mark.parametrize("dims", DIMS)
@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_cleaned_points_no_duplicates(dims, dtype):
    """Test cleaning points with no duplicates."""
    if dims == 2:
        points = np.array([[0, 0], [1, 0], [0, 1]], dtype=dtype)
    else:
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=dtype)

    cleaned_points = tf.cleaned(points)

    # Should return same number of points
    assert cleaned_points.shape == points.shape

    # Points should match (possibly reordered)
    cleaned_sorted = canonicalize_points(cleaned_points)
    original_sorted = canonicalize_points(points)
    assert np.allclose(cleaned_sorted, original_sorted, atol=1e-10)


@pytest.mark.parametrize("dims", DIMS)
@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_cleaned_points_empty(dims, dtype):
    """Test cleaning empty point array."""
    points = np.empty((0, dims), dtype=dtype)

    cleaned_points = tf.cleaned(points)

    assert cleaned_points.shape == (0, dims)
    assert cleaned_points.dtype == dtype


# ==============================================================================
# PointCloud Tests
# ==============================================================================

@pytest.mark.parametrize("dims", DIMS)
@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_cleaned_pointcloud_exact_duplicates(dims, dtype):
    """Test cleaning PointCloud objects."""
    points = create_points_with_duplicates(dims, dtype)
    pc = tf.PointCloud(points)

    # Clean without index map
    cleaned_points = tf.cleaned(pc)

    # Should have 4 unique points
    assert cleaned_points.shape[0] == 4
    assert cleaned_points.shape[1] == dims
    assert cleaned_points.dtype == dtype


@pytest.mark.parametrize("dims", DIMS)
@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_cleaned_pointcloud_with_index_map(dims, dtype):
    """Test PointCloud cleaning with index map."""
    points = create_points_with_duplicates(dims, dtype)
    original_count = len(points)
    pc = tf.PointCloud(points)

    # Clean with index map
    cleaned_points, (f, kept_ids) = tf.cleaned(pc, return_index_map=True)

    # Validate index map
    validate_index_map(f, kept_ids, original_count)

    assert len(kept_ids) == 4
    assert cleaned_points.shape[0] == 4


# ==============================================================================
# EdgeMesh Tests
# ==============================================================================

@pytest.mark.parametrize("dims", DIMS)
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_cleaned_edgemesh_exact_duplicates(dims, index_dtype, real_dtype):
    """Test cleaning EdgeMesh objects."""
    edges, points = create_edge_mesh_with_duplicates(dims, index_dtype, real_dtype)
    em = tf.EdgeMesh(edges, points)

    # Clean without index map
    (cleaned_edges, cleaned_points) = tf.cleaned(em)

    # Should have 3 unique points (removed 1 duplicate)
    assert cleaned_points.shape[0] == 3
    assert cleaned_points.shape[1] == dims
    assert cleaned_points.dtype == real_dtype

    # Should have 2 edges
    assert cleaned_edges.shape[0] == 2
    assert cleaned_edges.shape[1] == 2
    assert cleaned_edges.dtype == index_dtype

    # All edge indices should be valid
    assert np.all(cleaned_edges >= 0)
    assert np.all(cleaned_edges < len(cleaned_points))


@pytest.mark.parametrize("dims", DIMS)
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_cleaned_edgemesh_with_index_map(dims, index_dtype, real_dtype):
    """Test EdgeMesh cleaning with index maps."""
    edges, points = create_edge_mesh_with_duplicates(dims, index_dtype, real_dtype)
    original_point_count = len(points)
    original_edge_count = len(edges)
    em = tf.EdgeMesh(edges, points)

    # Clean with index maps
    (cleaned_edges, cleaned_points), (f_edges, kept_edge_ids), (f_points, kept_point_ids) = \
        tf.cleaned(em, return_index_map=True)

    # Validate edge map
    validate_index_map(f_edges, kept_edge_ids, original_edge_count)

    # Validate point map
    validate_index_map(f_points, kept_point_ids, original_point_count)

    # Should have 3 unique points
    assert len(kept_point_ids) == 3


# ==============================================================================
# Mesh Tests
# ==============================================================================

@pytest.mark.parametrize("dims", DIMS)
@pytest.mark.parametrize("ngon", NGONS)
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_cleaned_mesh_exact_duplicates(dims, ngon, index_dtype, real_dtype):
    """Test cleaning Mesh objects."""
    faces, points = create_mesh_with_duplicates(dims, ngon, index_dtype, real_dtype)
    mesh = tf.Mesh(faces, points)

    # Clean without index map
    (cleaned_faces, cleaned_points) = tf.cleaned(mesh)

    # Should have 4 unique points (removed 1 duplicate)
    assert cleaned_points.shape[0] == 4, \
        f"Expected 4 unique points, got {cleaned_points.shape[0]}"
    assert cleaned_points.shape[1] == dims
    assert cleaned_points.dtype == real_dtype

    # Should have same number of faces
    assert cleaned_faces.shape[0] == 2
    assert cleaned_faces.shape[1] == ngon
    assert cleaned_faces.dtype == index_dtype

    # All face indices should be valid
    assert np.all(cleaned_faces >= 0)
    assert np.all(cleaned_faces < len(cleaned_points))


@pytest.mark.parametrize("dims", DIMS)
@pytest.mark.parametrize("ngon", NGONS)
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_cleaned_mesh_with_index_map(dims, ngon, index_dtype, real_dtype):
    """Test Mesh cleaning with index maps."""
    faces, points = create_mesh_with_duplicates(dims, ngon, index_dtype, real_dtype)
    original_point_count = len(points)
    original_face_count = len(faces)
    mesh = tf.Mesh(faces, points)

    # Clean with index maps
    (cleaned_faces, cleaned_points), (f_faces, kept_face_ids), (f_points, kept_point_ids) = \
        tf.cleaned(mesh, return_index_map=True)

    # Validate face map
    validate_index_map(f_faces, kept_face_ids, original_face_count)

    # Validate point map
    validate_index_map(f_points, kept_point_ids, original_point_count)

    # Should have 4 unique points
    assert len(kept_point_ids) == 4


# ==============================================================================
# Tuple Input Tests (Indexed Geometry)
# ==============================================================================

@pytest.mark.parametrize("dims", DIMS)
@pytest.mark.parametrize("V", [2, 3])  # edges, triangles (dynamic for variable-sized)
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_cleaned_tuple_input_exact_duplicates(dims, V, index_dtype, real_dtype):
    """Test cleaning indexed geometry passed as (indices, points) tuple."""
    # Create indexed geometry with duplicate points
    if V == 2:
        edges, points = create_edge_mesh_with_duplicates(dims, index_dtype, real_dtype)
        indices = edges
    else:
        faces, points = create_mesh_with_duplicates(dims, V, index_dtype, real_dtype)
        indices = faces

    # Clean using tuple input
    (cleaned_indices, cleaned_points) = tf.cleaned((indices, points))

    # Verify output shapes
    assert cleaned_points.shape[1] == dims
    assert cleaned_points.dtype == real_dtype
    assert cleaned_indices.shape[1] == V
    assert cleaned_indices.dtype == index_dtype

    # Verify point deduplication
    if V == 2:
        # EdgeMesh has 3 unique points (removed 1 duplicate)
        assert cleaned_points.shape[0] == 3
        assert cleaned_indices.shape[0] == 2
    else:
        # Mesh has 4 unique points (removed 1 duplicate)
        assert cleaned_points.shape[0] == 4
        assert cleaned_indices.shape[0] == 2

    # All indices should be valid
    assert np.all(cleaned_indices >= 0)
    assert np.all(cleaned_indices < len(cleaned_points))


@pytest.mark.parametrize("dims", DIMS)
@pytest.mark.parametrize("V", [2, 3])  # edges, triangles
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_cleaned_tuple_input_with_index_map(dims, V, index_dtype, real_dtype):
    """Test tuple input cleaning with index maps."""
    # Create indexed geometry with duplicate points
    if V == 2:
        edges, points = create_edge_mesh_with_duplicates(dims, index_dtype, real_dtype)
        indices = edges
        original_count = 2
        expected_unique_points = 3
    else:
        faces, points = create_mesh_with_duplicates(dims, V, index_dtype, real_dtype)
        indices = faces
        original_count = 2
        expected_unique_points = 4

    original_point_count = len(points)

    # Clean with index maps using tuple input
    (cleaned_indices, cleaned_points), (f_indices, kept_index_ids), (f_points, kept_point_ids) = \
        tf.cleaned((indices, points), return_index_map=True)

    # Validate index map
    validate_index_map(f_indices, kept_index_ids, original_count)

    # Validate point map
    validate_index_map(f_points, kept_point_ids, original_point_count)

    # Should have expected number of unique points
    assert len(kept_point_ids) == expected_unique_points


@pytest.mark.parametrize("dims", DIMS)
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_cleaned_tuple_input_with_tolerance(dims, index_dtype, real_dtype):
    """Test tuple input with tolerance-based merging."""
    # Create points with near-duplicates
    if dims == 2:
        points = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.001, 0.001],  # Near-duplicate of first point
            [1.0, 1.0]
        ], dtype=real_dtype)
    else:
        points = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.001, 0.001, 0.001],  # Near-duplicate of first point
            [1.0, 1.0, 0.0]
        ], dtype=real_dtype)

    # Create triangle indices
    indices = np.array([[0, 1, 3], [2, 1, 3]], dtype=index_dtype)

    # Clean with tolerance
    (cleaned_indices, cleaned_points) = tf.cleaned((indices, points), tolerance=0.01)

    # Should merge near-duplicates - expect 3 unique points
    assert cleaned_points.shape[0] == 3
    assert cleaned_points.shape[1] == dims
    assert cleaned_points.dtype == real_dtype

    # Should have 1 triangles
    assert cleaned_indices.shape[0] == 1
    assert cleaned_indices.shape[1] == 3
    assert cleaned_indices.dtype == index_dtype

    # All indices should be valid
    assert np.all(cleaned_indices >= 0)
    assert np.all(cleaned_indices < len(cleaned_points))


# ==============================================================================
# Dynamic Tuple Input Tests (Variable-sized Polygons)
# ==============================================================================

@pytest.mark.parametrize("dims", DIMS)
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_cleaned_dynamic_tuple_input(dims, index_dtype, real_dtype):
    """Test cleaning dynamic (variable-sized) indexed geometry."""
    faces, points = create_dynamic_mesh_with_duplicates(dims, index_dtype, real_dtype)

    # Clean using tuple input with OffsetBlockedArray faces
    (cleaned_faces, cleaned_points) = tf.cleaned((faces, points))

    # Should return OffsetBlockedArray for faces
    assert isinstance(cleaned_faces, tf.OffsetBlockedArray), \
        f"Dynamic cleaned faces should be OffsetBlockedArray, got {type(cleaned_faces)}"

    # Verify output types
    assert cleaned_faces.dtype == index_dtype
    assert cleaned_points.dtype == real_dtype
    assert cleaned_points.shape[1] == dims

    # Should have 5 unique points (removed 1 duplicate)
    assert cleaned_points.shape[0] == 5

    # Should have 2 faces
    assert len(cleaned_faces) == 2

    # All indices should be valid
    assert np.all(cleaned_faces.data >= 0)
    assert np.all(cleaned_faces.data < len(cleaned_points))


@pytest.mark.parametrize("dims", DIMS)
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_cleaned_dynamic_tuple_input_with_index_map(dims, index_dtype, real_dtype):
    """Test dynamic tuple input cleaning with index maps."""
    faces, points = create_dynamic_mesh_with_duplicates(dims, index_dtype, real_dtype)
    original_face_count = len(faces)
    original_point_count = len(points)

    # Clean with index maps
    (cleaned_faces, cleaned_points), (f_faces, kept_face_ids), (f_points, kept_point_ids) = \
        tf.cleaned((faces, points), return_index_map=True)

    # Validate face map
    validate_index_map(f_faces, kept_face_ids, original_face_count)

    # Validate point map
    validate_index_map(f_points, kept_point_ids, original_point_count)

    # Should have 5 unique points
    assert len(kept_point_ids) == 5


# ==============================================================================
# Polygon Soup Tests
# ==============================================================================

@pytest.mark.parametrize("dims", DIMS)
@pytest.mark.parametrize("V", SOUP_VERTICES)
@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_cleaned_polygon_soup(dims, V, dtype):
    """Test cleaning polygon soups."""
    soup = create_polygon_soup_with_duplicates(dims, V, dtype)

    # Clean soup
    (connectivity, points) = tf.cleaned(soup)

    # Should return int32 connectivity
    assert connectivity.dtype == np.int32, \
        f"Soup connectivity should be int32, got {connectivity.dtype}"

    # Should have correct shapes
    assert connectivity.shape[1] == V
    assert points.shape[1] == dims
    assert points.dtype == dtype

    # All indices should be valid
    assert np.all(connectivity >= 0)
    assert np.all(connectivity < len(points))

    # Should have fewer points than original (due to deduplication)
    original_point_count = soup.shape[0] * V
    assert points.shape[0] < original_point_count


@pytest.mark.parametrize("dims", DIMS)
@pytest.mark.parametrize("V", SOUP_VERTICES)
@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_cleaned_polygon_soup_with_tolerance(dims, V, dtype):
    """Test soup cleaning with tolerance."""
    soup = create_polygon_soup_with_duplicates(dims, V, dtype)

    # Clean with tolerance
    (connectivity, points) = tf.cleaned(soup, tolerance=0.01)

    # Should still work and deduplicate
    assert connectivity.dtype == np.int32
    assert points.dtype == dtype


# ==============================================================================
# Error Validation Tests
# ==============================================================================

def test_cleaned_invalid_dtype():
    """Test error for invalid dtype."""
    points = np.array([[0, 0], [1, 0]], dtype=np.int32)

    with pytest.raises(TypeError, match="dtype must be float32 or float64"):
        tf.cleaned(points)


def test_cleaned_negative_tolerance():
    """Test error for negative tolerance."""
    points = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="tolerance must be non-negative"):
        tf.cleaned(points, tolerance=-0.1)


def test_cleaned_soup_with_index_map():
    """Test that soups don't support index maps."""
    soup = np.array([[[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]]], dtype=np.float32)

    with pytest.raises(ValueError, match="return_index_map is not supported"):
        tf.cleaned(soup, return_index_map=True)


def test_cleaned_invalid_ndim():
    """Test error for wrong array dimensions."""
    points = np.array([0.0, 1.0, 2.0], dtype=np.float32)  # 1D

    with pytest.raises(ValueError, match="Expected 2D array .* or 3D array"):
        tf.cleaned(points)


def test_cleaned_invalid_point_dims():
    """Test error for invalid point dimensions."""
    points = np.array([[0, 0, 0, 0]], dtype=np.float32)  # 4D points

    with pytest.raises(ValueError, match="Points must have 2 or 3 dimensions"):
        tf.cleaned(points)


def test_cleaned_invalid_soup_V():
    """Test error for invalid soup element vertices."""
    soup = np.array([[[0, 0], [1, 0], [0.5, 1], [1, 1]]], dtype=np.float32)  # V=4 (not supported)

    with pytest.raises(ValueError, match="Soup elements must have 2.*or 3.*vertices"):
        tf.cleaned(soup)


def test_cleaned_invalid_tuple_V():
    """Test error for invalid V in tuple input (V=4 not supported for fixed-size)."""
    indices = np.array([[0, 1, 2, 3]], dtype=np.int32)  # V=4
    points = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)

    with pytest.raises(ValueError, match="Fixed-size indices must have 2.*or 3.*columns"):
        tf.cleaned((indices, points))


def test_cleaned_invalid_type():
    """Test error for unsupported input type."""
    invalid_input = "not a valid input"

    with pytest.raises(TypeError, match="Expected np.ndarray, tuple, or form object"):
        tf.cleaned(invalid_input)


# ==============================================================================
# Main runner
# ==============================================================================

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
