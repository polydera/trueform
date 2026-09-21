"""
Test non_manifold_vertices and split_non_manifold_vertices

Copyright (c) 2025 Ziga Sajovic, XLAB
"""

import pytest
import numpy as np
import trueform as tf

# Parameter sets
INDEX_DTYPES = [np.int32, np.int64]
REAL_DTYPES = [np.float32, np.float64]


# ==============================================================================
# Test data generators
# ==============================================================================

def create_bowtie_mesh(index_dtype, real_dtype):
    """Two triangles meeting at a single vertex (vertex 2)."""
    faces = np.array([
        [0, 1, 2],
        [2, 3, 4]
    ], dtype=index_dtype)
    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0],
        [1.5, 2.0, 0.0],
        [0.5, 2.0, 0.0]
    ], dtype=real_dtype)
    return faces, points


def create_three_on_one_edge_mesh(index_dtype, real_dtype):
    """Three triangles sharing the edge (0, 1)."""
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


def create_manifold_mesh(index_dtype, real_dtype):
    """Two triangles sharing one edge: every vertex is one fan."""
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


def create_dynamic_bowtie_mesh(index_dtype, real_dtype):
    """A triangle and a quad meeting at a single vertex (vertex 2)."""
    offsets = np.array([0, 3, 7], dtype=index_dtype)
    data = np.array([0, 1, 2, 2, 3, 4, 5], dtype=index_dtype)
    faces = tf.OffsetBlockedArray(offsets, data)
    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0],
        [1.5, 2.0, 0.0],
        [1.5, 3.0, 0.0],
        [0.5, 3.0, 0.0]
    ], dtype=real_dtype)
    return faces, points


# ==============================================================================
# non_manifold_vertices
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_non_manifold_vertices_bowtie(index_dtype, real_dtype):
    """The pinch vertex of a bowtie is named, and nothing else."""
    faces, points = create_bowtie_mesh(index_dtype, real_dtype)
    mesh = tf.Mesh(faces, points)

    vertices = tf.non_manifold_vertices(mesh)

    assert vertices.dtype == index_dtype
    np.testing.assert_array_equal(vertices, np.array([2], dtype=index_dtype))


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_non_manifold_vertices_three_on_one_edge(index_dtype):
    """A vertex on a 3-face edge is non-manifold."""
    faces, points = create_three_on_one_edge_mesh(index_dtype, np.float32)
    mesh = tf.Mesh(faces, points)

    vertices = tf.non_manifold_vertices(mesh)

    np.testing.assert_array_equal(
        vertices, np.array([0, 1], dtype=index_dtype))


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_non_manifold_vertices_manifold_mesh(index_dtype):
    """A manifold mesh names no vertex."""
    faces, points = create_manifold_mesh(index_dtype, np.float32)
    mesh = tf.Mesh(faces, points)

    vertices = tf.non_manifold_vertices(mesh)

    assert vertices.shape == (0,)


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_non_manifold_vertices_dynamic(index_dtype):
    """The dynamic entry names the pinch vertex too."""
    faces, points = create_dynamic_bowtie_mesh(index_dtype, np.float32)
    mesh = tf.Mesh(faces, points)

    vertices = tf.non_manifold_vertices(mesh)

    np.testing.assert_array_equal(vertices, np.array([2], dtype=index_dtype))


# ==============================================================================
# split_non_manifold_vertices
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_split_bowtie(index_dtype, real_dtype):
    """The fan holding the smallest face keeps the vertex, the other mints."""
    faces, points = create_bowtie_mesh(index_dtype, real_dtype)
    mesh = tf.Mesh(faces, points)

    split, point_map = tf.split_non_manifold_vertices(mesh)

    assert split.number_of_points == 6
    assert split.number_of_faces == 2
    assert split.faces.dtype == index_dtype
    assert split.points.dtype == real_dtype
    np.testing.assert_array_equal(
        split.faces, np.array([[0, 1, 2], [5, 3, 4]], dtype=index_dtype))

    # The minted point copies the pinch vertex's coordinates.
    assert point_map.dtype == index_dtype
    np.testing.assert_array_equal(
        point_map, np.array([0, 1, 2, 3, 4, 2], dtype=index_dtype))
    np.testing.assert_array_equal(split.points[:5], points)
    np.testing.assert_array_equal(split.points[5], points[2])

    assert tf.is_manifold(split)


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_split_leaves_three_on_one_edge_untouched(index_dtype):
    """A vertex on a 3+-face edge is left as it was and still named."""
    faces, points = create_three_on_one_edge_mesh(index_dtype, np.float32)
    mesh = tf.Mesh(faces, points)

    split, point_map = tf.split_non_manifold_vertices(mesh)

    assert split.number_of_points == 5
    np.testing.assert_array_equal(split.faces, faces)
    np.testing.assert_array_equal(split.points, points)
    np.testing.assert_array_equal(
        point_map, np.arange(5, dtype=index_dtype))

    # non_manifold_vertices still names the untouched vertices.
    np.testing.assert_array_equal(
        tf.non_manifold_vertices(split), np.array([0, 1], dtype=index_dtype))


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_split_manifold_mesh_is_identity(index_dtype):
    """A manifold mesh comes back as it was."""
    faces, points = create_manifold_mesh(index_dtype, np.float32)
    mesh = tf.Mesh(faces, points)

    split, point_map = tf.split_non_manifold_vertices(mesh)

    np.testing.assert_array_equal(split.faces, faces)
    np.testing.assert_array_equal(split.points, points)
    np.testing.assert_array_equal(point_map, np.arange(4, dtype=index_dtype))


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_split_dynamic_keeps_arity(index_dtype):
    """A dynamic split rewires the quad's corner and keeps its arity."""
    faces, points = create_dynamic_bowtie_mesh(index_dtype, np.float32)
    mesh = tf.Mesh(faces, points)

    split, point_map = tf.split_non_manifold_vertices(mesh)

    assert split.is_dynamic
    assert split.number_of_points == 7
    np.testing.assert_array_equal(
        point_map, np.array([0, 1, 2, 3, 4, 5, 2], dtype=index_dtype))
    np.testing.assert_array_equal(split.faces[0],
                                  np.array([0, 1, 2], dtype=index_dtype))
    np.testing.assert_array_equal(split.faces[1],
                                  np.array([6, 3, 4, 5], dtype=index_dtype))
    np.testing.assert_array_equal(split.points[6], points[2])

    assert tf.is_manifold(split)
