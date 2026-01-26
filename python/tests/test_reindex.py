"""
Test reindex functionality

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
NGONS = [3]  # Only triangles for fixed-size, dynamic for variable-sized

# ==============================================================================
# Canonicalization helpers (reuse from test_cleaned.py)
# ==============================================================================

def canonicalize_points(points):
    """Sort points lexicographically for comparison."""
    point_tuples = [tuple(p) for p in points]
    sort_order = sorted(range(len(point_tuples)), key=lambda i: point_tuples[i])
    return points[sort_order]


def canonicalize_edges(edges, points):
    """Create canonical ordering of edge mesh for comparison."""
    canonical_edges = []
    for edge in edges:
        ordered_edge = np.sort(edge)
        canonical_edges.append(ordered_edge)
    canonical_edges = np.array(canonical_edges)

    # Sort edges by their point coordinates
    sort_keys = []
    for edge in canonical_edges:
        edge_points = points[edge]
        sort_key = tuple(edge_points.flatten())
        sort_keys.append(sort_key)

    sort_order = sorted(range(len(sort_keys)), key=lambda i: sort_keys[i])
    canonical_edges = canonical_edges[sort_order]
    return canonical_edges, points


def canonicalize_faces(faces, points):
    """Create canonical ordering of mesh for comparison."""
    V = faces.shape[1]
    canonical_faces = []

    for face in faces:
        face_points = points[face]
        point_tuples = [tuple(p) for p in face_points]
        min_idx = min(range(V), key=lambda i: point_tuples[i])
        rotated_face = np.roll(face, -min_idx)
        canonical_faces.append(rotated_face)

    canonical_faces = np.array(canonical_faces)

    # Sort faces by their point coordinates
    sort_keys = []
    for face in canonical_faces:
        face_points = points[face]
        sort_key = tuple(face_points.flatten())
        sort_keys.append(sort_key)

    sort_order = sorted(range(len(sort_keys)), key=lambda i: sort_keys[i])
    canonical_faces = canonical_faces[sort_order]
    return canonical_faces, points


def validate_index_map(f, kept_ids, original_count):
    """Validate the index map structure."""
    assert f.shape == (original_count,), \
        f"f should have shape ({original_count},), got {f.shape}"
    assert len(kept_ids) <= original_count, \
        f"Can't keep more than original count: {len(kept_ids)} > {original_count}"
    assert np.all(f[kept_ids] < len(kept_ids)), \
        "Kept IDs should map to valid new indices"
    assert np.all(kept_ids >= 0) and np.all(kept_ids < original_count), \
        "kept_ids should be within [0, original_count)"

# ==============================================================================
# Test data generators
# ==============================================================================

def create_test_points(dims, dtype):
    """Create simple point set for reindexing tests."""
    if dims == 2:
        points = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 2.0],
        ], dtype=dtype)
    else:  # dims == 3
        points = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, 2.0, 2.0],
        ], dtype=dtype)
    return points


def create_test_mesh(dims, ngon, index_dtype, real_dtype):
    """Create simple mesh for reindexing tests."""
    if dims == 2:
        points = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, 1.0],
            [1.5, 0.5],
            [2.0, 1.0],
        ], dtype=real_dtype)

        faces = np.array([
            [0, 1, 2],
            [1, 3, 2],
            [3, 4, 2],
        ], dtype=index_dtype)
    else:  # dims == 3
        points = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [1.5, 0.5, 0.5],
            [2.0, 1.0, 0.0],
        ], dtype=real_dtype)

        faces = np.array([
            [0, 1, 2],
            [1, 3, 2],
            [3, 4, 2],
        ], dtype=index_dtype)

    return faces, points


def create_dynamic_mesh(dims, index_dtype, real_dtype):
    """Create a dynamic (variable-sized polygon) mesh for reindexing tests."""
    if dims == 2:
        points = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, 1.0],
            [1.5, 0.5],
            [2.0, 1.0],
        ], dtype=real_dtype)
    else:  # dims == 3
        points = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [1.5, 0.5, 0.5],
            [2.0, 1.0, 0.0],
        ], dtype=real_dtype)

    # Three faces: triangle, triangle, triangle (same as fixed but wrapped as dynamic)
    offsets = np.array([0, 3, 6, 9], dtype=index_dtype)
    data = np.array([0, 1, 2, 1, 3, 2, 3, 4, 2], dtype=index_dtype)
    faces = tf.OffsetBlockedArray(offsets, data)

    return faces, points


def create_test_edge_mesh(dims, index_dtype, real_dtype):
    """Create simple edge mesh for reindexing tests."""
    if dims == 2:
        points = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ], dtype=real_dtype)

        edges = np.array([
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
        ], dtype=index_dtype)
    else:  # dims == 3
        points = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=real_dtype)

        edges = np.array([
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
        ], dtype=index_dtype)

    return edges, points

# ==============================================================================
# Points Tests - reindex_by_ids
# ==============================================================================

@pytest.mark.parametrize("dims", DIMS)
@pytest.mark.parametrize("dtype", REAL_DTYPES)
@pytest.mark.parametrize("id_dtype", INDEX_DTYPES)
def test_reindex_by_ids_points_basic(dims, dtype, id_dtype):
    """Test reindexing points by IDs."""
    points = create_test_points(dims, dtype)
    ids = np.array([0, 2, 4], dtype=id_dtype)

    result = tf.reindex_by_ids(tf.PointCloud(points), ids)

    assert result.shape == (3, dims)
    assert result.dtype == dtype
    assert np.allclose(result[0], points[0])
    assert np.allclose(result[1], points[2])
    assert np.allclose(result[2], points[4])


@pytest.mark.parametrize("dims", DIMS)
@pytest.mark.parametrize("dtype", REAL_DTYPES)
@pytest.mark.parametrize("id_dtype", INDEX_DTYPES)
def test_reindex_by_ids_points_with_index_map(dims, dtype, id_dtype):
    """Test reindexing points with index map return."""
    points = create_test_points(dims, dtype)
    original_count = len(points)
    ids = np.array([1, 3], dtype=id_dtype)

    result, (f, kept_ids) = tf.reindex_by_ids(
        tf.PointCloud(points), ids, return_index_map=True
    )

    validate_index_map(f, kept_ids, original_count)
    assert result.shape == (2, dims)
    assert len(kept_ids) == 2
    assert np.array_equal(kept_ids, ids)
    assert np.allclose(result, points[kept_ids])


@pytest.mark.parametrize("dims", DIMS)
@pytest.mark.parametrize("dtype", REAL_DTYPES)
@pytest.mark.parametrize("id_dtype", INDEX_DTYPES)
def test_reindex_by_ids_points_empty_ids(dims, dtype, id_dtype):
    """Test reindexing with empty ID list."""
    points = create_test_points(dims, dtype)
    ids = np.array([], dtype=id_dtype)

    result = tf.reindex_by_ids(tf.PointCloud(points), ids)

    assert result.shape == (0, dims)
    assert result.dtype == dtype


@pytest.mark.parametrize("dims", DIMS)
@pytest.mark.parametrize("dtype", REAL_DTYPES)
@pytest.mark.parametrize("id_dtype", INDEX_DTYPES)
def test_reindex_by_ids_points_all_ids(dims, dtype, id_dtype):
    """Test reindexing with all IDs."""
    points = create_test_points(dims, dtype)
    ids = np.arange(len(points), dtype=id_dtype)

    result = tf.reindex_by_ids(tf.PointCloud(points), ids)

    assert result.shape == points.shape
    # Points should match (possibly reordered)
    result_sorted = canonicalize_points(result)
    points_sorted = canonicalize_points(points)
    assert np.allclose(result_sorted, points_sorted)

# ==============================================================================
# Points Tests - reindex_by_mask
# ==============================================================================

@pytest.mark.parametrize("dims", DIMS)
@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_reindex_by_mask_points_basic(dims, dtype):
    """Test filtering points by mask."""
    points = create_test_points(dims, dtype)
    mask = np.array([True, False, True, False, True], dtype=bool)

    result = tf.reindex_by_mask(tf.PointCloud(points), mask)

    assert result.shape == (3, dims)
    assert result.dtype == dtype
    assert np.allclose(result[0], points[0])
    assert np.allclose(result[1], points[2])
    assert np.allclose(result[2], points[4])


@pytest.mark.parametrize("dims", DIMS)
@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_reindex_by_mask_points_with_index_map(dims, dtype):
    """Test filtering points with index map return."""
    points = create_test_points(dims, dtype)
    original_count = len(points)
    mask = np.array([False, True, False, True, False], dtype=bool)

    result, (f, kept_ids) = tf.reindex_by_mask(
        tf.PointCloud(points), mask, return_index_map=True
    )

    validate_index_map(f, kept_ids, original_count)
    assert result.shape == (2, dims)
    assert len(kept_ids) == 2
    assert kept_ids[0] == 1
    assert kept_ids[1] == 3
    assert np.allclose(result, points[kept_ids])


@pytest.mark.parametrize("dims", DIMS)
@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_reindex_by_mask_points_all_false(dims, dtype):
    """Test with all False mask."""
    points = create_test_points(dims, dtype)
    mask = np.array([False] * len(points), dtype=bool)

    result = tf.reindex_by_mask(tf.PointCloud(points), mask)

    assert result.shape == (0, dims)
    assert result.dtype == dtype


@pytest.mark.parametrize("dims", DIMS)
@pytest.mark.parametrize("dtype", REAL_DTYPES)
def test_reindex_by_mask_points_all_true(dims, dtype):
    """Test with all True mask."""
    points = create_test_points(dims, dtype)
    mask = np.array([True] * len(points), dtype=bool)

    result = tf.reindex_by_mask(tf.PointCloud(points), mask)

    assert result.shape == points.shape
    # Points should match (possibly reordered)
    result_sorted = canonicalize_points(result)
    points_sorted = canonicalize_points(points)
    assert np.allclose(result_sorted, points_sorted)

# ==============================================================================
# Mesh Tests - reindex_by_ids
# ==============================================================================

@pytest.mark.parametrize("dims", DIMS)
@pytest.mark.parametrize("ngon", NGONS)
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_reindex_by_ids_mesh_basic(dims, ngon, index_dtype, real_dtype):
    """Test reindexing mesh by face IDs."""
    faces, points = create_test_mesh(dims, ngon, index_dtype, real_dtype)
    mesh = tf.Mesh(faces, points)

    # Select first and last face
    face_ids = np.array([0, 2], dtype=index_dtype)

    (new_faces, new_points) = tf.reindex_by_ids(mesh, face_ids)

    assert new_faces.shape[0] == len(face_ids)
    assert new_faces.shape[1] == ngon
    assert new_faces.dtype == index_dtype
    assert new_points.shape[1] == dims
    assert new_points.dtype == real_dtype
    # All indices should be valid
    assert np.all(new_faces >= 0)
    assert np.all(new_faces < len(new_points))


@pytest.mark.parametrize("dims", DIMS)
@pytest.mark.parametrize("ngon", NGONS)
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_reindex_by_ids_mesh_with_index_map(dims, ngon, index_dtype, real_dtype):
    """Test reindexing mesh with index maps."""
    faces, points = create_test_mesh(dims, ngon, index_dtype, real_dtype)
    original_face_count = len(faces)
    original_point_count = len(points)
    mesh = tf.Mesh(faces, points)

    face_ids = np.array([0], dtype=index_dtype)

    (new_faces, new_points), (f_map, kept_faces), (p_map, kept_points) = \
        tf.reindex_by_ids(mesh, face_ids, return_index_map=True)

    validate_index_map(f_map, kept_faces, original_face_count)
    validate_index_map(p_map, kept_points, original_point_count)

    assert len(kept_faces) == 1
    assert np.array_equal(kept_faces, face_ids)

# ==============================================================================
# Mesh Tests - reindex_by_mask
# ==============================================================================

@pytest.mark.parametrize("dims", DIMS)
@pytest.mark.parametrize("ngon", NGONS)
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_reindex_by_mask_mesh_basic(dims, ngon, index_dtype, real_dtype):
    """Test filtering mesh by face mask."""
    faces, points = create_test_mesh(dims, ngon, index_dtype, real_dtype)
    mesh = tf.Mesh(faces, points)

    face_mask = np.array([True, False, True], dtype=bool)

    (new_faces, new_points) = tf.reindex_by_mask(mesh, face_mask)

    expected_faces = face_mask.sum()
    assert new_faces.shape[0] == expected_faces
    assert new_faces.shape[1] == ngon
    assert new_faces.dtype == index_dtype
    assert new_points.shape[1] == dims
    assert new_points.dtype == real_dtype
    # All indices should be valid
    assert np.all(new_faces >= 0)
    assert np.all(new_faces < len(new_points))


@pytest.mark.parametrize("dims", DIMS)
@pytest.mark.parametrize("ngon", NGONS)
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_reindex_by_mask_mesh_with_index_map(dims, ngon, index_dtype, real_dtype):
    """Test filtering mesh with index maps."""
    faces, points = create_test_mesh(dims, ngon, index_dtype, real_dtype)
    original_face_count = len(faces)
    original_point_count = len(points)
    mesh = tf.Mesh(faces, points)

    face_mask = np.array([True, False, True], dtype=bool)

    (new_faces, new_points), (f_map, kept_faces), (p_map, kept_points) = \
        tf.reindex_by_mask(mesh, face_mask, return_index_map=True)

    validate_index_map(f_map, kept_faces, original_face_count)
    validate_index_map(p_map, kept_points, original_point_count)

    assert len(kept_faces) == face_mask.sum()

# ==============================================================================
# EdgeMesh Tests - reindex_by_ids
# ==============================================================================

@pytest.mark.parametrize("dims", DIMS)
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_reindex_by_ids_edgemesh_basic(dims, index_dtype, real_dtype):
    """Test reindexing EdgeMesh by edge IDs."""
    edges, points = create_test_edge_mesh(dims, index_dtype, real_dtype)
    em = tf.EdgeMesh(edges, points)
    edge_ids = np.array([0, 2], dtype=index_dtype)

    (new_edges, new_points) = tf.reindex_by_ids(em, edge_ids)

    assert new_edges.shape[0] == 2
    assert new_edges.shape[1] == 2
    assert new_edges.dtype == index_dtype
    assert new_points.shape[1] == dims
    assert new_points.dtype == real_dtype
    # All indices should be valid
    assert np.all(new_edges >= 0)
    assert np.all(new_edges < len(new_points))


@pytest.mark.parametrize("dims", DIMS)
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_reindex_by_ids_edgemesh_with_index_map(dims, index_dtype, real_dtype):
    """Test EdgeMesh reindexing with index maps."""
    edges, points = create_test_edge_mesh(dims, index_dtype, real_dtype)
    original_edge_count = len(edges)
    original_point_count = len(points)
    em = tf.EdgeMesh(edges, points)
    edge_ids = np.array([1, 3], dtype=index_dtype)

    (new_edges, new_points), (e_map, kept_edges), (p_map, kept_points) = \
        tf.reindex_by_ids(em, edge_ids, return_index_map=True)

    validate_index_map(e_map, kept_edges, original_edge_count)
    validate_index_map(p_map, kept_points, original_point_count)

    assert len(kept_edges) == 2
    assert np.array_equal(kept_edges, edge_ids)

# ==============================================================================
# EdgeMesh Tests - reindex_by_mask
# ==============================================================================

@pytest.mark.parametrize("dims", DIMS)
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_reindex_by_mask_edgemesh_basic(dims, index_dtype, real_dtype):
    """Test filtering EdgeMesh by edge mask."""
    edges, points = create_test_edge_mesh(dims, index_dtype, real_dtype)
    em = tf.EdgeMesh(edges, points)
    edge_mask = np.array([True, False, True, False], dtype=bool)

    (new_edges, new_points) = tf.reindex_by_mask(em, edge_mask)

    assert new_edges.shape[0] == 2
    assert new_edges.shape[1] == 2
    assert new_edges.dtype == index_dtype
    assert new_points.shape[1] == dims
    assert new_points.dtype == real_dtype
    # All indices should be valid
    assert np.all(new_edges >= 0)
    assert np.all(new_edges < len(new_points))


@pytest.mark.parametrize("dims", DIMS)
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_reindex_by_mask_edgemesh_with_index_map(dims, index_dtype, real_dtype):
    """Test EdgeMesh filtering with index maps."""
    edges, points = create_test_edge_mesh(dims, index_dtype, real_dtype)
    original_edge_count = len(edges)
    original_point_count = len(points)
    em = tf.EdgeMesh(edges, points)
    edge_mask = np.array([False, True, False, True], dtype=bool)

    (new_edges, new_points), (e_map, kept_edges), (p_map, kept_points) = \
        tf.reindex_by_mask(em, edge_mask, return_index_map=True)

    validate_index_map(e_map, kept_edges, original_edge_count)
    validate_index_map(p_map, kept_points, original_point_count)

    assert len(kept_edges) == 2

# ==============================================================================
# Tuple Input Tests - reindex_by_ids
# ==============================================================================

@pytest.mark.parametrize("dims", DIMS)
@pytest.mark.parametrize("V", [2, 3])
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_reindex_by_ids_tuple_input_basic(dims, V, index_dtype, real_dtype):
    """Test reindexing with tuple input."""
    if V == 2:
        indices, points = create_test_edge_mesh(dims, index_dtype, real_dtype)
        ids = np.array([0, 2], dtype=index_dtype)
    else:
        indices, points = create_test_mesh(dims, V, index_dtype, real_dtype)
        ids = np.array([0, 2], dtype=index_dtype)

    (new_indices, new_points) = tf.reindex_by_ids((indices, points), ids)

    assert new_indices.shape[0] == len(ids)
    assert new_indices.shape[1] == V
    assert new_indices.dtype == index_dtype
    assert new_points.shape[1] == dims
    assert new_points.dtype == real_dtype
    # All indices should be valid
    assert np.all(new_indices >= 0)
    assert np.all(new_indices < len(new_points))


@pytest.mark.parametrize("dims", DIMS)
@pytest.mark.parametrize("V", [2, 3])
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_reindex_by_ids_tuple_input_with_index_map(dims, V, index_dtype, real_dtype):
    """Test tuple input with index maps."""
    if V == 2:
        indices, points = create_test_edge_mesh(dims, index_dtype, real_dtype)
        ids = np.array([1], dtype=index_dtype)
    else:
        indices, points = create_test_mesh(dims, V, index_dtype, real_dtype)
        ids = np.array([0], dtype=index_dtype)

    original_index_count = len(indices)
    original_point_count = len(points)

    (new_indices, new_points), (i_map, kept_index_ids), (p_map, kept_point_ids) = \
        tf.reindex_by_ids((indices, points), ids, return_index_map=True)

    validate_index_map(i_map, kept_index_ids, original_index_count)
    validate_index_map(p_map, kept_point_ids, original_point_count)

    assert len(kept_index_ids) == len(ids)
    assert np.array_equal(kept_index_ids, ids)

# ==============================================================================
# Tuple Input Tests - reindex_by_mask
# ==============================================================================

@pytest.mark.parametrize("dims", DIMS)
@pytest.mark.parametrize("V", [2, 3])
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_reindex_by_mask_tuple_input_basic(dims, V, index_dtype, real_dtype):
    """Test filtering with tuple input."""
    if V == 2:
        indices, points = create_test_edge_mesh(dims, index_dtype, real_dtype)
        mask = np.array([True, False, True, False], dtype=bool)
    else:
        indices, points = create_test_mesh(dims, V, index_dtype, real_dtype)
        mask = np.array([True, False, True], dtype=bool)

    (new_indices, new_points) = tf.reindex_by_mask((indices, points), mask)

    assert new_indices.shape[0] == mask.sum()
    assert new_indices.shape[1] == V
    assert new_indices.dtype == index_dtype
    assert new_points.shape[1] == dims
    assert new_points.dtype == real_dtype
    # All indices should be valid
    assert np.all(new_indices >= 0)
    assert np.all(new_indices < len(new_points))


@pytest.mark.parametrize("dims", DIMS)
@pytest.mark.parametrize("V", [2, 3])
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_reindex_by_mask_tuple_input_with_index_map(dims, V, index_dtype, real_dtype):
    """Test tuple input filtering with index maps."""
    if V == 2:
        indices, points = create_test_edge_mesh(dims, index_dtype, real_dtype)
        mask = np.array([False, True, False, True], dtype=bool)
    else:
        indices, points = create_test_mesh(dims, V, index_dtype, real_dtype)
        mask = np.array([True, False, False], dtype=bool)

    original_index_count = len(indices)
    original_point_count = len(points)

    (new_indices, new_points), (i_map, kept_index_ids), (p_map, kept_point_ids) = \
        tf.reindex_by_mask((indices, points), mask, return_index_map=True)

    validate_index_map(i_map, kept_index_ids, original_index_count)
    validate_index_map(p_map, kept_point_ids, original_point_count)

    assert len(kept_index_ids) == mask.sum()

# ==============================================================================
# Dynamic Mesh Tests - reindex_by_ids
# ==============================================================================

@pytest.mark.parametrize("dims", DIMS)
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_reindex_by_ids_dynamic_mesh_basic(dims, index_dtype, real_dtype):
    """Test reindexing dynamic mesh by face IDs."""
    faces, points = create_dynamic_mesh(dims, index_dtype, real_dtype)
    mesh = tf.Mesh(faces, points)

    face_ids = np.array([0, 2], dtype=index_dtype)

    (new_faces, new_points) = tf.reindex_by_ids(mesh, face_ids)

    assert isinstance(new_faces, tf.OffsetBlockedArray)
    assert len(new_faces) == len(face_ids)
    assert new_faces.dtype == index_dtype
    assert new_points.shape[1] == dims
    assert new_points.dtype == real_dtype


@pytest.mark.parametrize("dims", DIMS)
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_reindex_by_ids_dynamic_mesh_with_index_map(dims, index_dtype, real_dtype):
    """Test reindexing dynamic mesh with index maps."""
    faces, points = create_dynamic_mesh(dims, index_dtype, real_dtype)
    original_face_count = len(faces)
    original_point_count = len(points)
    mesh = tf.Mesh(faces, points)

    face_ids = np.array([0], dtype=index_dtype)

    (new_faces, new_points), (f_map, kept_faces), (p_map, kept_points) = \
        tf.reindex_by_ids(mesh, face_ids, return_index_map=True)

    validate_index_map(f_map, kept_faces, original_face_count)
    validate_index_map(p_map, kept_points, original_point_count)

    assert isinstance(new_faces, tf.OffsetBlockedArray)
    assert len(kept_faces) == 1
    assert np.array_equal(kept_faces, face_ids)


# ==============================================================================
# Dynamic Mesh Tests - reindex_by_mask
# ==============================================================================

@pytest.mark.parametrize("dims", DIMS)
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_reindex_by_mask_dynamic_mesh_basic(dims, index_dtype, real_dtype):
    """Test filtering dynamic mesh by face mask."""
    faces, points = create_dynamic_mesh(dims, index_dtype, real_dtype)
    mesh = tf.Mesh(faces, points)

    face_mask = np.array([True, False, True], dtype=bool)

    (new_faces, new_points) = tf.reindex_by_mask(mesh, face_mask)

    expected_faces = face_mask.sum()
    assert isinstance(new_faces, tf.OffsetBlockedArray)
    assert len(new_faces) == expected_faces
    assert new_faces.dtype == index_dtype
    assert new_points.shape[1] == dims
    assert new_points.dtype == real_dtype


@pytest.mark.parametrize("dims", DIMS)
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_reindex_by_mask_dynamic_mesh_with_index_map(dims, index_dtype, real_dtype):
    """Test filtering dynamic mesh with index maps."""
    faces, points = create_dynamic_mesh(dims, index_dtype, real_dtype)
    original_face_count = len(faces)
    original_point_count = len(points)
    mesh = tf.Mesh(faces, points)

    face_mask = np.array([True, False, True], dtype=bool)

    (new_faces, new_points), (f_map, kept_faces), (p_map, kept_points) = \
        tf.reindex_by_mask(mesh, face_mask, return_index_map=True)

    validate_index_map(f_map, kept_faces, original_face_count)
    validate_index_map(p_map, kept_points, original_point_count)

    assert isinstance(new_faces, tf.OffsetBlockedArray)
    assert len(kept_faces) == face_mask.sum()


# ==============================================================================
# Dynamic Tuple Input Tests - reindex_by_ids
# ==============================================================================

@pytest.mark.parametrize("dims", DIMS)
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_reindex_by_ids_dynamic_tuple_basic(dims, index_dtype, real_dtype):
    """Test reindexing with dynamic tuple input (OffsetBlockedArray)."""
    faces, points = create_dynamic_mesh(dims, index_dtype, real_dtype)
    ids = np.array([0, 2], dtype=index_dtype)

    (new_faces, new_points) = tf.reindex_by_ids((faces, points), ids)

    assert isinstance(new_faces, tf.OffsetBlockedArray)
    assert len(new_faces) == len(ids)
    assert new_faces.dtype == index_dtype
    assert new_points.shape[1] == dims
    assert new_points.dtype == real_dtype


@pytest.mark.parametrize("dims", DIMS)
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_reindex_by_ids_dynamic_tuple_with_index_map(dims, index_dtype, real_dtype):
    """Test dynamic tuple input with index maps."""
    faces, points = create_dynamic_mesh(dims, index_dtype, real_dtype)
    ids = np.array([1], dtype=index_dtype)

    original_face_count = len(faces)
    original_point_count = len(points)

    (new_faces, new_points), (f_map, kept_face_ids), (p_map, kept_point_ids) = \
        tf.reindex_by_ids((faces, points), ids, return_index_map=True)

    validate_index_map(f_map, kept_face_ids, original_face_count)
    validate_index_map(p_map, kept_point_ids, original_point_count)

    assert isinstance(new_faces, tf.OffsetBlockedArray)
    assert len(kept_face_ids) == len(ids)
    assert np.array_equal(kept_face_ids, ids)


# ==============================================================================
# Dynamic Tuple Input Tests - reindex_by_mask
# ==============================================================================

@pytest.mark.parametrize("dims", DIMS)
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_reindex_by_mask_dynamic_tuple_basic(dims, index_dtype, real_dtype):
    """Test filtering with dynamic tuple input (OffsetBlockedArray)."""
    faces, points = create_dynamic_mesh(dims, index_dtype, real_dtype)
    mask = np.array([True, False, True], dtype=bool)

    (new_faces, new_points) = tf.reindex_by_mask((faces, points), mask)

    assert isinstance(new_faces, tf.OffsetBlockedArray)
    assert len(new_faces) == mask.sum()
    assert new_faces.dtype == index_dtype
    assert new_points.shape[1] == dims
    assert new_points.dtype == real_dtype


@pytest.mark.parametrize("dims", DIMS)
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_reindex_by_mask_dynamic_tuple_with_index_map(dims, index_dtype, real_dtype):
    """Test dynamic tuple input filtering with index maps."""
    faces, points = create_dynamic_mesh(dims, index_dtype, real_dtype)
    mask = np.array([True, False, False], dtype=bool)

    original_face_count = len(faces)
    original_point_count = len(points)

    (new_faces, new_points), (f_map, kept_face_ids), (p_map, kept_point_ids) = \
        tf.reindex_by_mask((faces, points), mask, return_index_map=True)

    validate_index_map(f_map, kept_face_ids, original_face_count)
    validate_index_map(p_map, kept_point_ids, original_point_count)

    assert isinstance(new_faces, tf.OffsetBlockedArray)
    assert len(kept_face_ids) == mask.sum()


# ==============================================================================
# reindex_by_mask_on_points Tests - Triangle Mesh
# ==============================================================================

@pytest.mark.parametrize("dims", DIMS)
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_reindex_by_mask_on_points_mesh_basic(dims, index_dtype, real_dtype):
    """Test filtering mesh faces based on point mask."""
    faces, points = create_test_mesh(dims, 3, index_dtype, real_dtype)
    mesh = tf.Mesh(faces, points)

    # Mask that keeps only points 2, 3, 4 (face [3, 4, 2] should survive)
    point_mask = np.array([False, False, True, True, True], dtype=bool)

    (new_faces, new_points) = tf.reindex_by_mask_on_points(mesh, point_mask)

    # Only face [3, 4, 2] has all vertices in the mask
    assert new_faces.shape[0] == 1
    assert new_faces.shape[1] == 3
    assert new_faces.dtype == index_dtype
    assert new_points.dtype == real_dtype
    # All indices should be valid
    assert np.all(new_faces >= 0)
    assert np.all(new_faces < len(new_points))


@pytest.mark.parametrize("dims", DIMS)
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_reindex_by_mask_on_points_mesh_with_index_map(dims, index_dtype, real_dtype):
    """Test filtering mesh with index maps based on point mask."""
    faces, points = create_test_mesh(dims, 3, index_dtype, real_dtype)
    original_face_count = len(faces)
    original_point_count = len(points)
    mesh = tf.Mesh(faces, points)

    # Mask that keeps points 2, 3, 4
    point_mask = np.array([False, False, True, True, True], dtype=bool)

    (new_faces, new_points), (f_map, kept_faces), (p_map, kept_points) = \
        tf.reindex_by_mask_on_points(mesh, point_mask, return_index_map=True)

    validate_index_map(f_map, kept_faces, original_face_count)
    validate_index_map(p_map, kept_points, original_point_count)

    assert new_faces.shape[0] == 1


@pytest.mark.parametrize("dims", DIMS)
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_reindex_by_mask_on_points_all_pass(dims, index_dtype, real_dtype):
    """Test with all points passing mask."""
    faces, points = create_test_mesh(dims, 3, index_dtype, real_dtype)
    mesh = tf.Mesh(faces, points)

    point_mask = np.array([True] * len(points), dtype=bool)

    (new_faces, new_points) = tf.reindex_by_mask_on_points(mesh, point_mask)

    # All faces should survive
    assert new_faces.shape[0] == len(faces)


@pytest.mark.parametrize("dims", DIMS)
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_reindex_by_mask_on_points_none_pass(dims, index_dtype, real_dtype):
    """Test with no faces surviving (mask excludes vertices needed by all faces)."""
    faces, points = create_test_mesh(dims, 3, index_dtype, real_dtype)
    mesh = tf.Mesh(faces, points)

    # Mask that only keeps point 4, which is not enough for any face
    point_mask = np.array([False, False, False, False, True], dtype=bool)

    (new_faces, new_points) = tf.reindex_by_mask_on_points(mesh, point_mask)

    assert new_faces.shape[0] == 0


# ==============================================================================
# reindex_by_mask_on_points Tests - Dynamic Mesh
# ==============================================================================

@pytest.mark.parametrize("dims", DIMS)
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_reindex_by_mask_on_points_dynamic_mesh_basic(dims, index_dtype, real_dtype):
    """Test filtering dynamic mesh faces based on point mask."""
    faces, points = create_dynamic_mesh(dims, index_dtype, real_dtype)
    mesh = tf.Mesh(faces, points)

    # Mask that keeps only points 2, 3, 4 (face [3, 4, 2] should survive)
    point_mask = np.array([False, False, True, True, True], dtype=bool)

    (new_faces, new_points) = tf.reindex_by_mask_on_points(mesh, point_mask)

    assert isinstance(new_faces, tf.OffsetBlockedArray)
    # Only face [3, 4, 2] has all vertices in the mask
    assert len(new_faces) == 1
    assert new_faces.dtype == index_dtype
    assert new_points.dtype == real_dtype


@pytest.mark.parametrize("dims", DIMS)
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_reindex_by_mask_on_points_dynamic_mesh_with_index_map(dims, index_dtype, real_dtype):
    """Test filtering dynamic mesh with index maps based on point mask."""
    faces, points = create_dynamic_mesh(dims, index_dtype, real_dtype)
    original_face_count = len(faces)
    original_point_count = len(points)
    mesh = tf.Mesh(faces, points)

    # Mask that keeps points 2, 3, 4
    point_mask = np.array([False, False, True, True, True], dtype=bool)

    (new_faces, new_points), (f_map, kept_faces), (p_map, kept_points) = \
        tf.reindex_by_mask_on_points(mesh, point_mask, return_index_map=True)

    validate_index_map(f_map, kept_faces, original_face_count)
    validate_index_map(p_map, kept_points, original_point_count)

    assert isinstance(new_faces, tf.OffsetBlockedArray)
    assert len(new_faces) == 1


# ==============================================================================
# reindex_by_ids_on_points Tests - Triangle Mesh
# ==============================================================================

@pytest.mark.parametrize("dims", DIMS)
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_reindex_by_ids_on_points_mesh_basic(dims, index_dtype, real_dtype):
    """Test filtering mesh faces based on point IDs."""
    faces, points = create_test_mesh(dims, 3, index_dtype, real_dtype)
    mesh = tf.Mesh(faces, points)

    # Keep only points 2, 3, 4 (face [3, 4, 2] should survive)
    point_ids = np.array([2, 3, 4], dtype=index_dtype)

    (new_faces, new_points) = tf.reindex_by_ids_on_points(mesh, point_ids)

    # Only face [3, 4, 2] has all vertices in the IDs
    assert new_faces.shape[0] == 1
    assert new_faces.shape[1] == 3
    assert new_faces.dtype == index_dtype
    assert new_points.dtype == real_dtype
    # All indices should be valid
    assert np.all(new_faces >= 0)
    assert np.all(new_faces < len(new_points))


@pytest.mark.parametrize("dims", DIMS)
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_reindex_by_ids_on_points_mesh_with_index_map(dims, index_dtype, real_dtype):
    """Test filtering mesh with index maps based on point IDs."""
    faces, points = create_test_mesh(dims, 3, index_dtype, real_dtype)
    original_face_count = len(faces)
    original_point_count = len(points)
    mesh = tf.Mesh(faces, points)

    # Keep only points 2, 3, 4
    point_ids = np.array([2, 3, 4], dtype=index_dtype)

    (new_faces, new_points), (f_map, kept_faces), (p_map, kept_points) = \
        tf.reindex_by_ids_on_points(mesh, point_ids, return_index_map=True)

    validate_index_map(f_map, kept_faces, original_face_count)
    validate_index_map(p_map, kept_points, original_point_count)

    assert new_faces.shape[0] == 1


# ==============================================================================
# reindex_by_ids_on_points Tests - Dynamic Mesh
# ==============================================================================

@pytest.mark.parametrize("dims", DIMS)
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_reindex_by_ids_on_points_dynamic_mesh_basic(dims, index_dtype, real_dtype):
    """Test filtering dynamic mesh faces based on point IDs."""
    faces, points = create_dynamic_mesh(dims, index_dtype, real_dtype)
    mesh = tf.Mesh(faces, points)

    # Keep only points 2, 3, 4 (face [3, 4, 2] should survive)
    point_ids = np.array([2, 3, 4], dtype=index_dtype)

    (new_faces, new_points) = tf.reindex_by_ids_on_points(mesh, point_ids)

    assert isinstance(new_faces, tf.OffsetBlockedArray)
    # Only face [3, 4, 2] has all vertices in the IDs
    assert len(new_faces) == 1
    assert new_faces.dtype == index_dtype
    assert new_points.dtype == real_dtype


@pytest.mark.parametrize("dims", DIMS)
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_reindex_by_ids_on_points_dynamic_mesh_with_index_map(dims, index_dtype, real_dtype):
    """Test filtering dynamic mesh with index maps based on point IDs."""
    faces, points = create_dynamic_mesh(dims, index_dtype, real_dtype)
    original_face_count = len(faces)
    original_point_count = len(points)
    mesh = tf.Mesh(faces, points)

    # Keep only points 2, 3, 4
    point_ids = np.array([2, 3, 4], dtype=index_dtype)

    (new_faces, new_points), (f_map, kept_faces), (p_map, kept_points) = \
        tf.reindex_by_ids_on_points(mesh, point_ids, return_index_map=True)

    validate_index_map(f_map, kept_faces, original_face_count)
    validate_index_map(p_map, kept_points, original_point_count)

    assert isinstance(new_faces, tf.OffsetBlockedArray)
    assert len(new_faces) == 1


# ==============================================================================
# reindex_by_mask_on_points Tests - EdgeMesh
# ==============================================================================

@pytest.mark.parametrize("dims", DIMS)
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_reindex_by_mask_on_points_edgemesh_basic(dims, index_dtype, real_dtype):
    """Test filtering EdgeMesh edges based on point mask."""
    edges, points = create_test_edge_mesh(dims, index_dtype, real_dtype)
    em = tf.EdgeMesh(edges, points)

    # Mask that keeps only points 0, 1 (edge [0, 1] should survive)
    point_mask = np.array([True, True, False, False], dtype=bool)

    (new_edges, new_points) = tf.reindex_by_mask_on_points(em, point_mask)

    # Only edge [0, 1] has all vertices in the mask
    assert new_edges.shape[0] == 1
    assert new_edges.shape[1] == 2
    assert new_edges.dtype == index_dtype
    assert new_points.dtype == real_dtype


@pytest.mark.parametrize("dims", DIMS)
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_reindex_by_ids_on_points_edgemesh_basic(dims, index_dtype, real_dtype):
    """Test filtering EdgeMesh edges based on point IDs."""
    edges, points = create_test_edge_mesh(dims, index_dtype, real_dtype)
    em = tf.EdgeMesh(edges, points)

    # Keep only points 0, 1 (edge [0, 1] should survive)
    point_ids = np.array([0, 1], dtype=index_dtype)

    (new_edges, new_points) = tf.reindex_by_ids_on_points(em, point_ids)

    # Only edge [0, 1] has all vertices in the IDs
    assert new_edges.shape[0] == 1
    assert new_edges.shape[1] == 2
    assert new_edges.dtype == index_dtype
    assert new_points.dtype == real_dtype


# ==============================================================================
# reindex_by_mask_on_points Tests - Tuple Input
# ==============================================================================

@pytest.mark.parametrize("dims", DIMS)
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_reindex_by_mask_on_points_tuple_basic(dims, index_dtype, real_dtype):
    """Test filtering tuple input (faces, points) based on point mask."""
    faces, points = create_test_mesh(dims, 3, index_dtype, real_dtype)

    # Mask that keeps only points 2, 3, 4
    point_mask = np.array([False, False, True, True, True], dtype=bool)

    (new_faces, new_points) = tf.reindex_by_mask_on_points((faces, points), point_mask)

    # Only face [3, 4, 2] has all vertices in the mask
    assert new_faces.shape[0] == 1


@pytest.mark.parametrize("dims", DIMS)
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_reindex_by_mask_on_points_dynamic_tuple_basic(dims, index_dtype, real_dtype):
    """Test filtering dynamic tuple input based on point mask."""
    faces, points = create_dynamic_mesh(dims, index_dtype, real_dtype)

    # Mask that keeps only points 2, 3, 4
    point_mask = np.array([False, False, True, True, True], dtype=bool)

    (new_faces, new_points) = tf.reindex_by_mask_on_points((faces, points), point_mask)

    assert isinstance(new_faces, tf.OffsetBlockedArray)
    assert len(new_faces) == 1


# ==============================================================================
# Error Validation Tests
# ==============================================================================

def test_reindex_by_ids_invalid_ids_dtype():
    """Test that non-integer IDs raise TypeError."""
    points = np.array([[0, 0, 0]], dtype=np.float32)
    ids = np.array([0], dtype=np.float32)

    with pytest.raises(TypeError, match="ids dtype must be int32 or int64"):
        tf.reindex_by_ids(tf.PointCloud(points), ids)


def test_reindex_by_mask_invalid_mask_dtype():
    """Test that non-bool mask raises TypeError."""
    points = np.array([[0, 0, 0]], dtype=np.float32)
    mask = np.array([1], dtype=np.int32)

    with pytest.raises(TypeError, match="mask dtype must be bool"):
        tf.reindex_by_mask(tf.PointCloud(points), mask)


def test_reindex_by_mask_wrong_size():
    """Test that wrong-sized mask raises ValueError."""
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    mask = np.array([True, False], dtype=bool)  # Wrong size

    with pytest.raises(ValueError, match="mask size.*must match"):
        tf.reindex_by_mask((faces, points), mask)


def test_reindex_by_ids_2d_ids():
    """Test that 2D IDs array raises ValueError."""
    points = np.array([[0, 0, 0]], dtype=np.float32)
    ids = np.array([[0]], dtype=np.int32)

    with pytest.raises(ValueError, match="ids must be 1D array"):
        tf.reindex_by_ids(tf.PointCloud(points), ids)


def test_reindex_by_mask_2d_mask():
    """Test that 2D mask array raises ValueError."""
    points = np.array([[0, 0, 0]], dtype=np.float32)
    mask = np.array([[True]], dtype=bool)

    with pytest.raises(ValueError, match="mask must be 1D array"):
        tf.reindex_by_mask(tf.PointCloud(points), mask)


def test_reindex_by_ids_invalid_type():
    """Test error for unsupported input type."""
    invalid_input = "not a valid input"
    ids = np.array([0], dtype=np.int32)

    with pytest.raises(TypeError, match="Expected tuple or form object"):
        tf.reindex_by_ids(invalid_input, ids)


def test_reindex_by_mask_invalid_type():
    """Test error for unsupported input type."""
    invalid_input = "not a valid input"
    mask = np.array([True], dtype=bool)

    with pytest.raises(TypeError, match="Expected tuple or form object"):
        tf.reindex_by_mask(invalid_input, mask)


def test_reindex_by_ids_mismatched_index_dtype():
    """Test that mismatched index dtypes raise TypeError."""
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    ids = np.array([0], dtype=np.int64)  # Doesn't match faces dtype

    with pytest.raises(TypeError, match="indices dtype.*must match ids dtype"):
        tf.reindex_by_ids((faces, points), ids)


def test_reindex_by_ids_v4_not_supported():
    """Test that V=4 (quads) raises ValueError - use OffsetBlockedArray instead."""
    quads = np.array([[0, 1, 2, 3]], dtype=np.int32)
    points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32)
    ids = np.array([0], dtype=np.int32)

    with pytest.raises(ValueError, match="Fixed-size indices must have 2.*or 3.*columns"):
        tf.reindex_by_ids((quads, points), ids)


def test_reindex_by_mask_v4_not_supported():
    """Test that V=4 (quads) raises ValueError - use OffsetBlockedArray instead."""
    quads = np.array([[0, 1, 2, 3]], dtype=np.int32)
    points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32)
    mask = np.array([True], dtype=bool)

    with pytest.raises(ValueError, match="Fixed-size indices must have 2.*or 3.*columns"):
        tf.reindex_by_mask((quads, points), mask)


# ==============================================================================
# Main runner
# ==============================================================================

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
