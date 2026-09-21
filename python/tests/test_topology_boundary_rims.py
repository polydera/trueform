"""
Test boundary_rims

Copyright (c) 2025 Ziga Sajovic, XLAB
"""

import pytest
import numpy as np
import trueform as tf

# Parameter sets
INDEX_DTYPES = [np.int32, np.int64]
REAL_DTYPES = [np.float32, np.float64]


# ==============================================================================
# Helpers
# ==============================================================================

def boundary_edge_owners(faces_blocks):
    """Map each boundary edge (as an ordered pair) to the face that winds it."""
    owners = {}
    edge_count = {}
    for face_id, face in enumerate(faces_blocks):
        n = len(face)
        for k in range(n):
            a, b = int(face[k]), int(face[(k + 1) % n])
            edge_count[(min(a, b), max(a, b))] = \
                edge_count.get((min(a, b), max(a, b)), 0) + 1
            owners[(a, b)] = face_id
    return {
        edge: face_id for edge, face_id in owners.items()
        if edge_count[(min(edge), max(edge))] == 1
    }


def assert_rim_edges_match_their_faces(faces_blocks, vertices, rim_faces,
                                       closed):
    """Every rim edge is a boundary edge carried by the face the rim names."""
    owners = boundary_edge_owners(faces_blocks)
    for i in range(len(vertices)):
        rim = vertices[i]
        n_edges = len(rim) if closed[i] else len(rim) - 1
        assert len(rim_faces[i]) == n_edges
        for k in range(n_edges):
            a = int(rim[k])
            b = int(rim[(k + 1) % len(rim)])
            edge = (a, b) if (a, b) in owners else (b, a)
            assert edge in owners
            assert owners[edge] == rim_faces[i][k]


# ==============================================================================
# Test data generators
# ==============================================================================

def create_strip_mesh(index_dtype, real_dtype):
    """A quad of two triangles: one closed rim of four vertices."""
    faces = np.array([
        [0, 1, 2],
        [0, 2, 3]
    ], dtype=index_dtype)
    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0]
    ], dtype=real_dtype)
    return faces, points


def create_pinched_mesh(index_dtype, real_dtype):
    """Two triangles meeting at vertex 2: the boundary pinches there."""
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
    """Three triangles sharing the edge (0, 1): three open rims."""
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


def create_closed_tetrahedron(index_dtype, real_dtype):
    """A closed tetrahedron: no boundary, no rims."""
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


def create_dynamic_strip_mesh(index_dtype, real_dtype):
    """The strip as a dynamic mesh."""
    offsets = np.array([0, 3, 6], dtype=index_dtype)
    data = np.array([0, 1, 2, 0, 2, 3], dtype=index_dtype)
    faces = tf.OffsetBlockedArray(offsets, data)
    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0]
    ], dtype=real_dtype)
    return faces, points


# ==============================================================================
# Tests
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_boundary_rims_strip(index_dtype, real_dtype):
    """One closed rim, walked the way the faces wind their boundary."""
    faces, points = create_strip_mesh(index_dtype, real_dtype)
    mesh = tf.Mesh(faces, points)

    vertices, rim_faces, closed = tf.boundary_rims(mesh)

    assert len(vertices) == 1
    assert len(rim_faces) == 1
    assert closed.dtype == np.int8
    assert closed.shape == (1,)
    assert closed[0]

    rim = vertices[0]
    assert len(rim) == 4
    assert len(rim_faces[0]) == 4

    # The rim runs as wound: the cyclic sequence is 0 -> 1 -> 2 -> 3.
    start = int(np.where(rim == 0)[0][0])
    rolled = np.roll(rim, -start)
    np.testing.assert_array_equal(rolled, np.array([0, 1, 2, 3]))

    assert_rim_edges_match_their_faces(list(faces), vertices, rim_faces,
                                       closed)


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_boundary_rims_pinched(index_dtype):
    """A pinched boundary comes back as its pieces, both closed."""
    faces, points = create_pinched_mesh(index_dtype, np.float32)
    mesh = tf.Mesh(faces, points)

    vertices, rim_faces, closed = tf.boundary_rims(mesh)

    assert len(vertices) == 2
    assert closed.shape == (2,)
    assert closed[0] and closed[1]

    for i in range(2):
        assert len(vertices[i]) == 3
        assert len(rim_faces[i]) == 3
        # Each rim's edges are carried by one lobe alone.
        assert len(set(int(f) for f in rim_faces[i])) == 1
        # The pinch vertex is on both rims.
        assert 2 in set(int(v) for v in vertices[i])

    assert_rim_edges_match_their_faces(list(faces), vertices, rim_faces,
                                       closed)


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_boundary_rims_open(index_dtype):
    """A vertex the boundary reaches three times ends rims: three open ones."""
    faces, points = create_three_on_one_edge_mesh(index_dtype, np.float32)
    mesh = tf.Mesh(faces, points)

    vertices, rim_faces, closed = tf.boundary_rims(mesh)

    assert len(vertices) == 3
    middles = set()
    for i in range(3):
        assert not closed[i]
        assert len(vertices[i]) == 3
        assert len(rim_faces[i]) == 2
        middles.add(int(vertices[i][1]))
    assert middles == {2, 3, 4}

    assert_rim_edges_match_their_faces(list(faces), vertices, rim_faces,
                                       closed)


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_boundary_rims_closed_mesh(index_dtype):
    """A closed mesh has no boundary and no rims."""
    faces, points = create_closed_tetrahedron(index_dtype, np.float32)
    mesh = tf.Mesh(faces, points)

    vertices, rim_faces, closed = tf.boundary_rims(mesh)

    assert len(vertices) == 0
    assert len(rim_faces) == 0
    assert closed.shape == (0,)


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_boundary_rims_dynamic(index_dtype):
    """The dynamic entry states the same rim."""
    faces, points = create_dynamic_strip_mesh(index_dtype, np.float32)
    mesh = tf.Mesh(faces, points)

    vertices, rim_faces, closed = tf.boundary_rims(mesh)

    assert len(vertices) == 1
    assert closed[0]
    assert len(vertices[0]) == 4
    assert_rim_edges_match_their_faces(
        [faces[i] for i in range(len(faces))], vertices, rim_faces, closed)
