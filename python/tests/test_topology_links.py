"""
Test face_link and vertex_link functionality

Copyright (c) 2025 Ziga Sajovic, XLAB
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import trueform as tf

# Parameter sets
INDEX_DTYPES = [np.int32, np.int64]
REAL_DTYPES = [np.float32, np.float64]


# ==============================================================================
# Test data generators
# ==============================================================================

def create_triangle_mesh(index_dtype, real_dtype):
    """Create a simple triangle mesh (2 triangles sharing an edge)."""
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


def create_quad_mesh(index_dtype, real_dtype):
    """Create a simple quad mesh (2 quads sharing an edge)."""
    faces = np.array([
        [0, 1, 2, 3],
        [1, 4, 5, 2]
    ], dtype=index_dtype)
    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 1.0, 0.0]
    ], dtype=real_dtype)
    return faces, points


def create_edge_mesh(index_dtype, real_dtype):
    """Create a simple edge mesh (3 edges forming a path)."""
    edges = np.array([
        [0, 1],
        [1, 2],
        [2, 3]
    ], dtype=index_dtype)
    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [3.0, 0.0, 0.0]
    ], dtype=real_dtype)
    return edges, points


def create_larger_triangle_mesh(index_dtype, real_dtype):
    """Create a larger triangle mesh for more comprehensive testing."""
    # 4 triangles forming a strip
    faces = np.array([
        [0, 1, 2],
        [1, 3, 2],
        [2, 3, 4],
        [3, 5, 4]
    ], dtype=index_dtype)
    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0],
        [1.5, 1.0, 0.0],
        [1.0, 2.0, 0.0],
        [2.0, 2.0, 0.0]
    ], dtype=real_dtype)
    return faces, points


# ==============================================================================
# manifold_edge_link Tests - Mesh only (V=3,4)
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_manifold_edge_link_mesh_triangles(index_dtype, real_dtype):
    """Test manifold_edge_link matches Mesh.manifold_edge_link for triangles."""
    faces, points = create_triangle_mesh(index_dtype, real_dtype)
    mesh = tf.Mesh(faces, points)

    # Get from mesh property
    mesh_mel = mesh.manifold_edge_link

    # Get cell_membership first
    fm = tf.cell_membership(faces, mesh.number_of_points)

    # Get manifold_edge_link from standalone function
    standalone_mel = tf.manifold_edge_link(faces, fm)

    # Compare
    assert np.array_equal(mesh_mel, standalone_mel), \
        "Manifold edge link should match between mesh property and standalone function"


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_manifold_edge_link_mesh_quads(index_dtype, real_dtype):
    """Test manifold_edge_link matches Mesh.manifold_edge_link for quads."""
    faces, points = create_quad_mesh(index_dtype, real_dtype)
    mesh = tf.Mesh(faces, points)

    # Get from mesh property
    mesh_mel = mesh.manifold_edge_link

    # Get cell_membership first
    fm = tf.cell_membership(faces, mesh.number_of_points)

    # Get manifold_edge_link from standalone function
    standalone_mel = tf.manifold_edge_link(faces, fm)

    # Compare
    assert np.array_equal(mesh_mel, standalone_mel)


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_manifold_edge_link_mesh_assignment(index_dtype, real_dtype):
    """Test assigning manifold_edge_link back to mesh."""
    faces, points = create_triangle_mesh(index_dtype, real_dtype)
    mesh = tf.Mesh(faces, points)

    # Get cell_membership and manifold_edge_link
    fm = tf.cell_membership(faces, mesh.number_of_points)
    standalone_mel = tf.manifold_edge_link(faces, fm)

    # Assign back to mesh
    mesh.manifold_edge_link = standalone_mel

    # Get it back and verify
    retrieved_mel = mesh.manifold_edge_link

    assert np.array_equal(retrieved_mel, standalone_mel)


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_manifold_edge_link_structure(index_dtype, real_dtype):
    """Test manifold_edge_link structure for triangles."""
    faces, points = create_triangle_mesh(index_dtype, real_dtype)

    # Get cell_membership and manifold_edge_link
    fm = tf.cell_membership(faces, len(points))
    mel = tf.manifold_edge_link(faces, fm)

    # Verify shape
    assert mel.shape == faces.shape
    assert mel.dtype == faces.dtype

    # Two triangles sharing edge (1, 2):
    # Face 0: edges (0,1), (1,2), (2,0)
    # Face 1: edges (1,3), (3,2), (2,1)
    # Edge (1,2) in face 0 should link to face 1
    # Edge (2,1) in face 1 should link to face 0

    # Check that there's at least one valid link (not -1)
    has_valid_link = np.any(mel >= 0)
    assert has_valid_link, "Should have at least one adjacent face pair"


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_manifold_edge_link_larger_mesh(index_dtype, real_dtype):
    """Test manifold_edge_link with larger mesh."""
    faces, points = create_larger_triangle_mesh(index_dtype, real_dtype)
    mesh = tf.Mesh(faces, points)

    # Get from mesh property
    mesh_mel = mesh.manifold_edge_link

    # Get from standalone
    fm = tf.cell_membership(faces, mesh.number_of_points)
    standalone_mel = tf.manifold_edge_link(faces, fm)

    # Compare
    assert np.array_equal(mesh_mel, standalone_mel)

    # Verify shape
    assert standalone_mel.shape == (4, 3)  # 4 triangles, 3 edges each


# ==============================================================================
# face_link Tests - Mesh only (V=3,4)
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_face_link_mesh_triangles(index_dtype, real_dtype):
    """Test face_link matches Mesh.face_link for triangles."""
    faces, points = create_triangle_mesh(index_dtype, real_dtype)
    mesh = tf.Mesh(faces, points)

    # Get from mesh property
    mesh_fl = mesh.face_link

    # Get cell_membership first
    fm = tf.cell_membership(faces, mesh.number_of_points)

    # Get face_link from standalone function
    standalone_fl = tf.face_link(faces, fm)

    # Compare offsets and data
    assert np.array_equal(mesh_fl.offsets, standalone_fl.offsets), \
        "Offsets should match between mesh property and standalone function"
    assert np.array_equal(mesh_fl.data, standalone_fl.data), \
        "Data should match between mesh property and standalone function"


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_face_link_mesh_quads(index_dtype, real_dtype):
    """Test face_link matches Mesh.face_link for quads."""
    faces, points = create_quad_mesh(index_dtype, real_dtype)
    mesh = tf.Mesh(faces, points)

    # Get from mesh property
    mesh_fl = mesh.face_link

    # Get cell_membership first
    fm = tf.cell_membership(faces, mesh.number_of_points)

    # Get face_link from standalone function
    standalone_fl = tf.face_link(faces, fm)

    # Compare
    assert np.array_equal(mesh_fl.offsets, standalone_fl.offsets)
    assert np.array_equal(mesh_fl.data, standalone_fl.data)


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_face_link_mesh_assignment(index_dtype, real_dtype):
    """Test assigning face_link back to mesh."""
    faces, points = create_triangle_mesh(index_dtype, real_dtype)
    mesh = tf.Mesh(faces, points)

    # Get face_link from standalone function
    fm = tf.cell_membership(faces, mesh.number_of_points)
    standalone_fl = tf.face_link(faces, fm)

    # Assign back to mesh
    mesh.face_link = standalone_fl

    # Get it back and verify
    retrieved_fl = mesh.face_link

    assert np.array_equal(retrieved_fl.offsets, standalone_fl.offsets)
    assert np.array_equal(retrieved_fl.data, standalone_fl.data)


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_face_link_larger_mesh(index_dtype, real_dtype):
    """Test face_link with larger mesh."""
    faces, points = create_larger_triangle_mesh(index_dtype, real_dtype)
    mesh = tf.Mesh(faces, points)

    # Get from mesh property
    mesh_fl = mesh.face_link

    # Get from standalone
    fm = tf.cell_membership(faces, mesh.number_of_points)
    standalone_fl = tf.face_link(faces, fm)

    # Compare
    assert np.array_equal(mesh_fl.offsets, standalone_fl.offsets)
    assert np.array_equal(mesh_fl.data, standalone_fl.data)

    # Verify structure makes sense - face_link is indexed by faces
    assert len(standalone_fl) == mesh.number_of_faces


# ==============================================================================
# vertex_link Tests - Mesh (vertex_link_faces)
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_vertex_link_mesh_triangles(index_dtype, real_dtype):
    """Test vertex_link_faces matches Mesh.vertex_link for triangles."""
    faces, points = create_triangle_mesh(index_dtype, real_dtype)
    mesh = tf.Mesh(faces, points)

    # Get from mesh property
    mesh_vl = mesh.vertex_link

    # Get cell_membership first
    fm = tf.cell_membership(faces, mesh.number_of_points)

    # Get vertex_link from standalone function
    standalone_vl = tf.vertex_link_faces(faces, fm)

    # Compare offsets and data
    assert np.array_equal(mesh_vl.offsets, standalone_vl.offsets), \
        "Offsets should match between mesh property and standalone function"
    assert np.array_equal(mesh_vl.data, standalone_vl.data), \
        "Data should match between mesh property and standalone function"


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_vertex_link_mesh_quads(index_dtype, real_dtype):
    """Test vertex_link_faces matches Mesh.vertex_link for quads."""
    faces, points = create_quad_mesh(index_dtype, real_dtype)
    mesh = tf.Mesh(faces, points)

    # Get from mesh property
    mesh_vl = mesh.vertex_link

    # Get cell_membership first
    fm = tf.cell_membership(faces, mesh.number_of_points)

    # Get vertex_link from standalone function
    standalone_vl = tf.vertex_link_faces(faces, fm)

    # Compare
    assert np.array_equal(mesh_vl.offsets, standalone_vl.offsets)
    assert np.array_equal(mesh_vl.data, standalone_vl.data)


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_vertex_link_mesh_assignment(index_dtype, real_dtype):
    """Test assigning vertex_link back to mesh."""
    faces, points = create_triangle_mesh(index_dtype, real_dtype)
    mesh = tf.Mesh(faces, points)

    # Get vertex_link from standalone function
    fm = tf.cell_membership(faces, mesh.number_of_points)
    standalone_vl = tf.vertex_link_faces(faces, fm)

    # Assign back to mesh
    mesh.vertex_link = standalone_vl

    # Get it back and verify
    retrieved_vl = mesh.vertex_link

    assert np.array_equal(retrieved_vl.offsets, standalone_vl.offsets)
    assert np.array_equal(retrieved_vl.data, standalone_vl.data)


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_vertex_link_mesh_larger(index_dtype, real_dtype):
    """Test vertex_link with larger mesh."""
    faces, points = create_larger_triangle_mesh(index_dtype, real_dtype)
    mesh = tf.Mesh(faces, points)

    # Get from mesh property
    mesh_vl = mesh.vertex_link

    # Get from standalone
    fm = tf.cell_membership(faces, mesh.number_of_points)
    standalone_vl = tf.vertex_link_faces(faces, fm)

    # Compare
    assert np.array_equal(mesh_vl.offsets, standalone_vl.offsets)
    assert np.array_equal(mesh_vl.data, standalone_vl.data)

    # Verify structure makes sense
    assert len(standalone_vl) == mesh.number_of_points


# ==============================================================================
# vertex_link Tests - EdgeMesh (vertex_link_edges)
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_vertex_link_edge_mesh(index_dtype, real_dtype):
    """Test vertex_link_edges matches EdgeMesh.vertex_link."""
    edges, points = create_edge_mesh(index_dtype, real_dtype)
    edge_mesh = tf.EdgeMesh(edges, points)

    # Get from edge mesh property
    mesh_vl = edge_mesh.vertex_link

    # Get vertex_link from standalone function
    standalone_vl = tf.vertex_link_edges(edges, edge_mesh.number_of_points)

    # Compare offsets and data
    assert np.array_equal(mesh_vl.offsets, standalone_vl.offsets), \
        "Offsets should match between edge mesh property and standalone function"
    assert np.array_equal(mesh_vl.data, standalone_vl.data), \
        "Data should match between edge mesh property and standalone function"


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_vertex_link_edge_mesh_assignment(index_dtype, real_dtype):
    """Test assigning vertex_link back to edge mesh."""
    edges, points = create_edge_mesh(index_dtype, real_dtype)
    edge_mesh = tf.EdgeMesh(edges, points)

    # Get vertex_link from standalone function
    standalone_vl = tf.vertex_link_edges(edges, edge_mesh.number_of_points)

    # Assign back to edge mesh
    edge_mesh.vertex_link = standalone_vl

    # Get it back and verify
    retrieved_vl = edge_mesh.vertex_link

    assert np.array_equal(retrieved_vl.offsets, standalone_vl.offsets)
    assert np.array_equal(retrieved_vl.data, standalone_vl.data)


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_vertex_link_edge_mesh_structure(index_dtype, real_dtype):
    """Test vertex_link structure for edge mesh."""
    edges, points = create_edge_mesh(index_dtype, real_dtype)

    # Get vertex_link
    vl = tf.vertex_link_edges(edges, len(points))

    # Verify structure
    assert len(vl) == len(points)

    # Vertex 0 should connect to vertex 1 only
    assert len(vl[0]) == 1
    assert 1 in vl[0]

    # Vertex 1 should connect to vertices 0 and 2
    assert len(vl[1]) == 2
    assert 0 in vl[1] and 2 in vl[1]

    # Vertex 2 should connect to vertices 1 and 3
    assert len(vl[2]) == 2
    assert 1 in vl[2] and 3 in vl[2]

    # Vertex 3 should connect to vertex 2 only
    assert len(vl[3]) == 1
    assert 2 in vl[3]


# ==============================================================================
# Combined Tests - All Topology in One Mesh
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_all_topology_mesh_triangles(index_dtype, real_dtype):
    """Test all topology structures together for triangle mesh."""
    faces, points = create_triangle_mesh(index_dtype, real_dtype)
    mesh = tf.Mesh(faces, points)

    # Get all from mesh properties
    mesh_fm = mesh.face_membership
    mesh_mel = mesh.manifold_edge_link
    mesh_fl = mesh.face_link
    mesh_vl = mesh.vertex_link

    # Get cell_membership for standalone functions
    standalone_fm = tf.cell_membership(faces, mesh.number_of_points)

    # Get all from standalone functions
    standalone_mel = tf.manifold_edge_link(faces, standalone_fm)
    standalone_fl = tf.face_link(faces, standalone_fm)
    standalone_vl = tf.vertex_link_faces(faces, standalone_fm)

    # Compare all
    assert np.array_equal(mesh_fm.offsets, standalone_fm.offsets)
    assert np.array_equal(mesh_fm.data, standalone_fm.data)
    assert np.array_equal(mesh_mel, standalone_mel)
    assert np.array_equal(mesh_fl.offsets, standalone_fl.offsets)
    assert np.array_equal(mesh_fl.data, standalone_fl.data)
    assert np.array_equal(mesh_vl.offsets, standalone_vl.offsets)
    assert np.array_equal(mesh_vl.data, standalone_vl.data)


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_all_topology_mesh_quads(index_dtype, real_dtype):
    """Test all topology structures together for quad mesh."""
    faces, points = create_quad_mesh(index_dtype, real_dtype)
    mesh = tf.Mesh(faces, points)

    # Get all from mesh properties
    mesh_fm = mesh.face_membership
    mesh_mel = mesh.manifold_edge_link
    mesh_fl = mesh.face_link
    mesh_vl = mesh.vertex_link

    # Get cell_membership for standalone functions
    standalone_fm = tf.cell_membership(faces, mesh.number_of_points)

    # Get all from standalone functions
    standalone_mel = tf.manifold_edge_link(faces, standalone_fm)
    standalone_fl = tf.face_link(faces, standalone_fm)
    standalone_vl = tf.vertex_link_faces(faces, standalone_fm)

    # Compare all
    assert np.array_equal(mesh_fm.offsets, standalone_fm.offsets)
    assert np.array_equal(mesh_fm.data, standalone_fm.data)
    assert np.array_equal(mesh_mel, standalone_mel)
    assert np.array_equal(mesh_fl.offsets, standalone_fl.offsets)
    assert np.array_equal(mesh_fl.data, standalone_fl.data)
    assert np.array_equal(mesh_vl.offsets, standalone_vl.offsets)
    assert np.array_equal(mesh_vl.data, standalone_vl.data)


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_all_topology_edge_mesh(index_dtype, real_dtype):
    """Test all topology structures together for edge mesh."""
    edges, points = create_edge_mesh(index_dtype, real_dtype)
    edge_mesh = tf.EdgeMesh(edges, points)

    # Get all from edge mesh properties
    mesh_em = edge_mesh.edge_membership
    mesh_vl = edge_mesh.vertex_link

    # Get from standalone functions
    standalone_em = tf.cell_membership(edges, edge_mesh.number_of_points)
    standalone_vl = tf.vertex_link_edges(edges, edge_mesh.number_of_points)

    # Compare all
    assert np.array_equal(mesh_em.offsets, standalone_em.offsets)
    assert np.array_equal(mesh_em.data, standalone_em.data)
    assert np.array_equal(mesh_vl.offsets, standalone_vl.offsets)
    assert np.array_equal(mesh_vl.data, standalone_vl.data)


# ==============================================================================
# Return Type Tests
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_manifold_edge_link_return_type(index_dtype):
    """Test manifold_edge_link returns ndarray with correct dtype."""
    faces = np.array([[0, 1, 2], [1, 2, 3]], dtype=index_dtype)
    fm = tf.cell_membership(faces, 4)

    result = tf.manifold_edge_link(faces, fm)

    assert isinstance(result, np.ndarray)
    assert result.dtype == index_dtype
    assert result.shape == faces.shape


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_face_link_return_type(index_dtype):
    """Test face_link returns OffsetBlockedArray."""
    faces = np.array([[0, 1, 2], [1, 2, 3]], dtype=index_dtype)
    fm = tf.cell_membership(faces, 4)

    result = tf.face_link(faces, fm)

    assert isinstance(result, tf.OffsetBlockedArray)
    assert result.offsets.dtype == index_dtype
    assert result.data.dtype == index_dtype


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_vertex_link_faces_return_type(index_dtype):
    """Test vertex_link_faces returns OffsetBlockedArray."""
    faces = np.array([[0, 1, 2], [1, 2, 3]], dtype=index_dtype)
    fm = tf.cell_membership(faces, 4)

    result = tf.vertex_link_faces(faces, fm)

    assert isinstance(result, tf.OffsetBlockedArray)
    assert result.offsets.dtype == index_dtype
    assert result.data.dtype == index_dtype


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_vertex_link_edges_return_type(index_dtype):
    """Test vertex_link_edges returns OffsetBlockedArray."""
    edges = np.array([[0, 1], [1, 2]], dtype=index_dtype)

    result = tf.vertex_link_edges(edges, 3)

    assert isinstance(result, tf.OffsetBlockedArray)
    assert result.offsets.dtype == index_dtype
    assert result.data.dtype == index_dtype


# ==============================================================================
# Error Validation Tests
# ==============================================================================

def test_manifold_edge_link_invalid_dtype():
    """Test error for invalid dtype."""
    faces = np.array([[0, 1, 2]], dtype=np.float32)
    fm = tf.cell_membership(np.array([[0, 1, 2]], dtype=np.int32), 3)

    with pytest.raises(TypeError, match="cells dtype must be int32 or int64"):
        tf.manifold_edge_link(faces, fm)


def test_manifold_edge_link_invalid_ngon():
    """Test error for invalid ngon (edges not supported)."""
    edges = np.array([[0, 1], [1, 2]], dtype=np.int32)
    fm = tf.cell_membership(edges, 3)

    with pytest.raises(ValueError, match="cells must have 3 or 4 vertices"):
        tf.manifold_edge_link(edges, fm)


def test_manifold_edge_link_invalid_fm_type():
    """Test error for invalid cell_membership type."""
    faces = np.array([[0, 1, 2]], dtype=np.int32)

    with pytest.raises(TypeError, match="cell_membership must be OffsetBlockedArray"):
        tf.manifold_edge_link(faces, "invalid")


def test_manifold_edge_link_dtype_mismatch():
    """Test error for dtype mismatch between cells and cell_membership."""
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    fm = tf.cell_membership(np.array([[0, 1, 2]], dtype=np.int32), 3)

    with pytest.raises(TypeError, match="cell_membership dtype.*must match.*cells dtype"):
        tf.manifold_edge_link(faces, fm)


def test_face_link_invalid_dtype():
    """Test error for invalid dtype."""
    faces = np.array([[0, 1, 2]], dtype=np.float32)
    fm = tf.cell_membership(np.array([[0, 1, 2]], dtype=np.int32), 3)

    with pytest.raises(TypeError, match="faces dtype must be int32 or int64"):
        tf.face_link(faces, fm)


def test_face_link_invalid_ngon():
    """Test error for invalid ngon (edges not supported)."""
    edges = np.array([[0, 1], [1, 2]], dtype=np.int32)
    fm = tf.cell_membership(edges, 3)

    with pytest.raises(ValueError, match="faces must have 3 or 4 vertices"):
        tf.face_link(edges, fm)


def test_face_link_invalid_fm_type():
    """Test error for invalid cell_membership type."""
    faces = np.array([[0, 1, 2]], dtype=np.int32)

    with pytest.raises(TypeError, match="cell_membership must be OffsetBlockedArray"):
        tf.face_link(faces, "invalid")


def test_face_link_dtype_mismatch():
    """Test error for dtype mismatch between faces and cell_membership."""
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    fm = tf.cell_membership(np.array([[0, 1, 2]], dtype=np.int32), 3)

    with pytest.raises(TypeError, match="cell_membership dtype.*must match.*faces dtype"):
        tf.face_link(faces, fm)


def test_vertex_link_faces_invalid_dtype():
    """Test error for invalid dtype."""
    faces = np.array([[0, 1, 2]], dtype=np.float32)
    fm = tf.cell_membership(np.array([[0, 1, 2]], dtype=np.int32), 3)

    with pytest.raises(TypeError, match="faces dtype must be int32 or int64"):
        tf.vertex_link_faces(faces, fm)


def test_vertex_link_faces_invalid_ngon():
    """Test error for invalid ngon (edges not supported)."""
    edges = np.array([[0, 1], [1, 2]], dtype=np.int32)
    fm = tf.cell_membership(edges, 3)

    with pytest.raises(ValueError, match="faces must have 3 or 4 vertices"):
        tf.vertex_link_faces(edges, fm)


def test_vertex_link_faces_dtype_mismatch():
    """Test error for dtype mismatch between faces and cell_membership."""
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    fm = tf.cell_membership(np.array([[0, 1, 2]], dtype=np.int32), 3)

    with pytest.raises(TypeError, match="cell_membership dtype.*must match.*faces dtype"):
        tf.vertex_link_faces(faces, fm)


def test_vertex_link_edges_invalid_dtype():
    """Test error for invalid dtype."""
    edges = np.array([[0, 1]], dtype=np.float32)

    with pytest.raises(TypeError, match="edges dtype must be int32 or int64"):
        tf.vertex_link_edges(edges, 2)


def test_vertex_link_edges_invalid_shape():
    """Test error for invalid shape."""
    edges = np.array([0, 1, 2], dtype=np.int32)  # 1D

    with pytest.raises(ValueError, match="edges must be 2D array"):
        tf.vertex_link_edges(edges, 3)


def test_vertex_link_edges_invalid_ngon():
    """Test error for invalid ngon (must be 2)."""
    faces = np.array([[0, 1, 2]], dtype=np.int32)  # triangles

    with pytest.raises(ValueError, match="edges must have 2 vertices"):
        tf.vertex_link_edges(faces, 3)


# ==============================================================================
# Main runner
# ==============================================================================

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
