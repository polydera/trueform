"""
Test split_into_components functionality

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
# Test data generators
# ==============================================================================

def create_test_mesh_with_labels(dims, ngon, index_dtype, real_dtype):
    """Create simple mesh with multiple components for splitting tests."""
    if dims == 2:
        points = np.array([
            [0.0, 0.0],  # 0
            [1.0, 0.0],  # 1
            [0.5, 1.0],  # 2
            [2.0, 0.0],  # 3
            [2.5, 1.0],  # 4
            [3.0, 0.0],  # 5
        ], dtype=real_dtype)

        faces = np.array([
            [0, 1, 2],  # Component 0
            [1, 3, 2],  # Component 0
            [3, 4, 5],  # Component 1
            [3, 5, 4],  # Component 1
        ], dtype=index_dtype)
        labels = np.array([0, 0, 1, 1], dtype=np.int32)  # Labels always int32
    else:  # dims == 3
        points = np.array([
            [0.0, 0.0, 0.0],  # 0
            [1.0, 0.0, 0.0],  # 1
            [0.5, 1.0, 0.0],  # 2
            [2.0, 0.0, 0.0],  # 3
            [2.5, 1.0, 0.0],  # 4
            [3.0, 0.0, 0.0],  # 5
        ], dtype=real_dtype)

        faces = np.array([
            [0, 1, 2],  # Component 0
            [1, 3, 2],  # Component 0
            [3, 4, 5],  # Component 1
            [3, 5, 4],  # Component 1
        ], dtype=index_dtype)
        labels = np.array([0, 0, 1, 1], dtype=np.int32)  # Labels always int32

    return faces, points, labels


def create_dynamic_mesh_with_labels(dims, index_dtype, real_dtype):
    """Create a dynamic (variable-sized polygon) mesh with labels for splitting tests."""
    if dims == 2:
        points = np.array([
            [0.0, 0.0],  # 0
            [1.0, 0.0],  # 1
            [0.5, 1.0],  # 2
            [2.0, 0.0],  # 3
            [2.5, 1.0],  # 4
            [3.0, 0.0],  # 5
        ], dtype=real_dtype)
    else:  # dims == 3
        points = np.array([
            [0.0, 0.0, 0.0],  # 0
            [1.0, 0.0, 0.0],  # 1
            [0.5, 1.0, 0.0],  # 2
            [2.0, 0.0, 0.0],  # 3
            [2.5, 1.0, 0.0],  # 4
            [3.0, 0.0, 0.0],  # 5
        ], dtype=real_dtype)

    # Four faces: triangle, triangle, triangle, triangle (same as fixed test)
    offsets = np.array([0, 3, 6, 9, 12], dtype=index_dtype)
    data = np.array([0, 1, 2, 1, 3, 2, 3, 4, 5, 3, 5, 4], dtype=index_dtype)
    faces = tf.OffsetBlockedArray(offsets, data)
    labels = np.array([0, 0, 1, 1], dtype=np.int32)

    return faces, points, labels


def create_test_edge_mesh_with_labels(dims, index_dtype, real_dtype):
    """Create simple edge mesh with multiple components for splitting tests."""
    if dims == 2:
        points = np.array([
            [0.0, 0.0],  # 0
            [1.0, 0.0],  # 1
            [1.0, 1.0],  # 2
            [2.0, 0.0],  # 3
            [3.0, 0.0],  # 4
        ], dtype=real_dtype)

        edges = np.array([
            [0, 1],  # Component 0
            [1, 2],  # Component 0
            [3, 4],  # Component 1
        ], dtype=index_dtype)
        labels = np.array([0, 0, 1], dtype=np.int32)  # Labels always int32
    else:  # dims == 3
        points = np.array([
            [0.0, 0.0, 0.0],  # 0
            [1.0, 0.0, 0.0],  # 1
            [1.0, 1.0, 0.0],  # 2
            [2.0, 0.0, 0.0],  # 3
            [3.0, 0.0, 0.0],  # 4
        ], dtype=real_dtype)

        edges = np.array([
            [0, 1],  # Component 0
            [1, 2],  # Component 0
            [3, 4],  # Component 1
        ], dtype=index_dtype)
        labels = np.array([0, 0, 1], dtype=np.int32)  # Labels always int32

    return edges, points, labels


# ==============================================================================
# Mesh Tests - split_into_components
# ==============================================================================

@pytest.mark.parametrize("dims", DIMS)
@pytest.mark.parametrize("ngon", NGONS)
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_split_into_components_mesh_basic(dims, ngon, index_dtype, real_dtype):
    """Test splitting mesh into components."""
    faces, points, labels = create_test_mesh_with_labels(dims, ngon, index_dtype, real_dtype)
    mesh = tf.Mesh(faces, points)

    components, comp_labels = tf.split_into_components(mesh, labels)

    # Should have 2 components
    assert len(components) == 2
    assert len(comp_labels) == 2

    # Check component labels (always int32)
    assert comp_labels.dtype == np.int32
    assert np.array_equal(comp_labels, np.array([0, 1], dtype=np.int32))

    # Check each component
    for i, (comp_faces, comp_points) in enumerate(components):
        assert comp_faces.dtype == index_dtype
        assert comp_points.dtype == real_dtype
        assert comp_faces.shape[1] == ngon
        assert comp_points.shape[1] == dims
        # All indices should be valid
        assert np.all(comp_faces >= 0)
        assert np.all(comp_faces < len(comp_points))


@pytest.mark.parametrize("dims", DIMS)
@pytest.mark.parametrize("ngon", NGONS)
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_split_into_components_mesh_single_component(dims, ngon, index_dtype, real_dtype):
    """Test splitting mesh with all same labels (single component)."""
    faces, points, labels = create_test_mesh_with_labels(dims, ngon, index_dtype, real_dtype)
    # All same label
    labels_single = np.zeros(len(faces), dtype=np.int32)
    mesh = tf.Mesh(faces, points)

    components, comp_labels = tf.split_into_components(mesh, labels_single)

    # Should have 1 component
    assert len(components) == 1
    assert len(comp_labels) == 1
    assert comp_labels[0] == 0

    comp_faces, comp_points = components[0]
    assert comp_faces.shape[0] == len(faces)


@pytest.mark.parametrize("dims", DIMS)
@pytest.mark.parametrize("ngon", NGONS)
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_split_into_components_mesh_all_unique(dims, ngon, index_dtype, real_dtype):
    """Test splitting mesh where each face is its own component."""
    faces, points, _ = create_test_mesh_with_labels(dims, ngon, index_dtype, real_dtype)
    # Unique label per face
    labels_unique = np.arange(len(faces), dtype=np.int32)
    mesh = tf.Mesh(faces, points)

    components, comp_labels = tf.split_into_components(mesh, labels_unique)

    # Should have one component per face
    assert len(components) == len(faces)
    assert len(comp_labels) == len(faces)

    for i, (comp_faces, comp_points) in enumerate(components):
        # Each component should have exactly 1 face
        assert comp_faces.shape[0] == 1
        assert comp_faces.shape[1] == ngon


@pytest.mark.parametrize("dims", DIMS)
@pytest.mark.parametrize("ngon", NGONS)
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_split_into_components_mesh_unsorted_labels(dims, ngon, index_dtype, real_dtype):
    """Test that components are sorted by label value."""
    faces, points, _ = create_test_mesh_with_labels(dims, ngon, index_dtype, real_dtype)
    # Unsorted labels: [2, 0, 1, 1]
    labels = np.array([2, 0, 1, 1], dtype=np.int32)

    mesh = tf.Mesh(faces, points)
    components, comp_labels = tf.split_into_components(mesh, labels)

    # Component labels should be sorted
    assert np.all(comp_labels[:-1] <= comp_labels[1:])


# ==============================================================================
# EdgeMesh Tests - split_into_components
# ==============================================================================

@pytest.mark.parametrize("dims", DIMS)
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_split_into_components_edgemesh_basic(dims, index_dtype, real_dtype):
    """Test splitting EdgeMesh into components."""
    edges, points, labels = create_test_edge_mesh_with_labels(dims, index_dtype, real_dtype)
    em = tf.EdgeMesh(edges, points)

    components, comp_labels = tf.split_into_components(em, labels)

    # Should have 2 components
    assert len(components) == 2
    assert len(comp_labels) == 2

    # Check component labels (always int32)
    assert comp_labels.dtype == np.int32
    assert np.array_equal(comp_labels, np.array([0, 1], dtype=np.int32))

    # Check each component
    for i, (comp_edges, comp_points) in enumerate(components):
        assert comp_edges.dtype == index_dtype
        assert comp_points.dtype == real_dtype
        assert comp_edges.shape[1] == 2
        assert comp_points.shape[1] == dims
        # All indices should be valid
        assert np.all(comp_edges >= 0)
        assert np.all(comp_edges < len(comp_points))


@pytest.mark.parametrize("dims", DIMS)
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_split_into_components_edgemesh_single_component(dims, index_dtype, real_dtype):
    """Test EdgeMesh with all same labels (single component)."""
    edges, points, labels = create_test_edge_mesh_with_labels(dims, index_dtype, real_dtype)
    # All same label
    labels_single = np.zeros(len(edges), dtype=np.int32)
    em = tf.EdgeMesh(edges, points)

    components, comp_labels = tf.split_into_components(em, labels_single)

    # Should have 1 component
    assert len(components) == 1
    assert len(comp_labels) == 1
    assert comp_labels[0] == 0

    comp_edges, comp_points = components[0]
    assert comp_edges.shape[0] == len(edges)


# ==============================================================================
# Tuple Input Tests - split_into_components
# ==============================================================================

@pytest.mark.parametrize("dims", DIMS)
@pytest.mark.parametrize("V", [2, 3])
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_split_into_components_tuple_input_basic(dims, V, index_dtype, real_dtype):
    """Test splitting with tuple input."""
    if V == 2:
        indices, points, labels = create_test_edge_mesh_with_labels(dims, index_dtype, real_dtype)
    else:
        indices, points, labels = create_test_mesh_with_labels(dims, V, index_dtype, real_dtype)

    components, comp_labels = tf.split_into_components((indices, points), labels)

    # Should have components
    assert len(components) > 0
    assert len(comp_labels) > 0
    assert len(components) == len(comp_labels)

    # Check each component
    for comp_indices, comp_points in components:
        assert comp_indices.shape[1] == V
        assert comp_indices.dtype == index_dtype
        assert comp_points.shape[1] == dims
        assert comp_points.dtype == real_dtype
        # All indices should be valid
        assert np.all(comp_indices >= 0)
        assert np.all(comp_indices < len(comp_points))


@pytest.mark.parametrize("dims", DIMS)
@pytest.mark.parametrize("V", [2, 3])
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_split_into_components_tuple_dtype_preservation(dims, V, index_dtype, real_dtype):
    """Test that dtypes are preserved in split components."""
    if V == 2:
        indices, points, labels = create_test_edge_mesh_with_labels(dims, index_dtype, real_dtype)
    else:
        indices, points, labels = create_test_mesh_with_labels(dims, V, index_dtype, real_dtype)

    components, comp_labels = tf.split_into_components((indices, points), labels)

    # Check dtype preservation
    for comp_indices, comp_points in components:
        assert comp_indices.dtype == index_dtype
        assert comp_points.dtype == real_dtype

    # Component labels are always int32
    assert comp_labels.dtype == np.int32


# ==============================================================================
# Dynamic Mesh Tests - split_into_components
# ==============================================================================

@pytest.mark.parametrize("dims", DIMS)
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_split_into_components_dynamic_mesh_basic(dims, index_dtype, real_dtype):
    """Test splitting dynamic mesh into components."""
    faces, points, labels = create_dynamic_mesh_with_labels(dims, index_dtype, real_dtype)
    mesh = tf.Mesh(faces, points)

    components, comp_labels = tf.split_into_components(mesh, labels)

    # Should have 2 components
    assert len(components) == 2
    assert len(comp_labels) == 2

    # Check component labels (always int32)
    assert comp_labels.dtype == np.int32
    assert np.array_equal(comp_labels, np.array([0, 1], dtype=np.int32))

    # Check each component
    for i, (comp_faces, comp_points) in enumerate(components):
        assert isinstance(comp_faces, tf.OffsetBlockedArray)
        assert comp_faces.dtype == index_dtype
        assert comp_points.dtype == real_dtype
        assert comp_points.shape[1] == dims


@pytest.mark.parametrize("dims", DIMS)
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_split_into_components_dynamic_mesh_single_component(dims, index_dtype, real_dtype):
    """Test splitting dynamic mesh with all same labels (single component)."""
    faces, points, labels = create_dynamic_mesh_with_labels(dims, index_dtype, real_dtype)
    # All same label
    labels_single = np.zeros(len(faces), dtype=np.int32)
    mesh = tf.Mesh(faces, points)

    components, comp_labels = tf.split_into_components(mesh, labels_single)

    # Should have 1 component
    assert len(components) == 1
    assert len(comp_labels) == 1
    assert comp_labels[0] == 0

    comp_faces, comp_points = components[0]
    assert isinstance(comp_faces, tf.OffsetBlockedArray)
    assert len(comp_faces) == len(faces)


@pytest.mark.parametrize("dims", DIMS)
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_split_into_components_dynamic_mesh_all_unique(dims, index_dtype, real_dtype):
    """Test splitting dynamic mesh where each face is its own component."""
    faces, points, _ = create_dynamic_mesh_with_labels(dims, index_dtype, real_dtype)
    # Unique label per face
    labels_unique = np.arange(len(faces), dtype=np.int32)
    mesh = tf.Mesh(faces, points)

    components, comp_labels = tf.split_into_components(mesh, labels_unique)

    # Should have one component per face
    assert len(components) == len(faces)
    assert len(comp_labels) == len(faces)

    for i, (comp_faces, comp_points) in enumerate(components):
        assert isinstance(comp_faces, tf.OffsetBlockedArray)
        # Each component should have exactly 1 face
        assert len(comp_faces) == 1


# ==============================================================================
# Dynamic Tuple Input Tests - split_into_components
# ==============================================================================

@pytest.mark.parametrize("dims", DIMS)
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_split_into_components_dynamic_tuple_basic(dims, index_dtype, real_dtype):
    """Test splitting with dynamic tuple input (OffsetBlockedArray)."""
    faces, points, labels = create_dynamic_mesh_with_labels(dims, index_dtype, real_dtype)

    components, comp_labels = tf.split_into_components((faces, points), labels)

    # Should have 2 components
    assert len(components) == 2
    assert len(comp_labels) == 2

    # Check each component
    for comp_faces, comp_points in components:
        assert isinstance(comp_faces, tf.OffsetBlockedArray)
        assert comp_faces.dtype == index_dtype
        assert comp_points.dtype == real_dtype
        assert comp_points.shape[1] == dims


@pytest.mark.parametrize("dims", DIMS)
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_split_into_components_dynamic_tuple_dtype_preservation(dims, index_dtype, real_dtype):
    """Test that dtypes are preserved in split components with dynamic input."""
    faces, points, labels = create_dynamic_mesh_with_labels(dims, index_dtype, real_dtype)

    components, comp_labels = tf.split_into_components((faces, points), labels)

    # Check dtype preservation
    for comp_faces, comp_points in components:
        assert isinstance(comp_faces, tf.OffsetBlockedArray)
        assert comp_faces.dtype == index_dtype
        assert comp_points.dtype == real_dtype

    # Component labels are always int32
    assert comp_labels.dtype == np.int32


# ==============================================================================
# Error Validation Tests
# ==============================================================================

def test_split_into_components_invalid_labels_dtype():
    """Test that non-int32 labels raise TypeError."""
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    labels = np.array([0.0], dtype=np.float32)

    with pytest.raises(TypeError, match="labels dtype must be int32"):
        tf.split_into_components((faces, points), labels)


def test_split_into_components_int64_labels_raises():
    """Test that int64 labels raise TypeError (only int32 supported)."""
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    labels = np.array([0], dtype=np.int64)  # int64 not supported

    with pytest.raises(TypeError, match="labels dtype must be int32"):
        tf.split_into_components((faces, points), labels)


def test_split_into_components_2d_labels():
    """Test that 2D labels array raises ValueError."""
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    labels = np.array([[0]], dtype=np.int32)

    with pytest.raises(ValueError, match="labels must be 1D array"):
        tf.split_into_components((faces, points), labels)


def test_split_into_components_wrong_size_labels():
    """Test that wrong-sized labels raises ValueError."""
    faces = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int32)
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float32)
    labels = np.array([0], dtype=np.int32)  # Wrong size

    with pytest.raises(ValueError, match="Number of labels.*must match"):
        tf.split_into_components((faces, points), labels)


def test_split_into_components_int64_indices_works():
    """Test that int64 indices work with int32 labels."""
    faces = np.array([[0, 1, 2]], dtype=np.int64)  # int64 indices
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    labels = np.array([0], dtype=np.int32)  # int32 labels (required)

    # Should work fine - indices can be int64, labels must be int32
    components, comp_labels = tf.split_into_components((faces, points), labels)

    assert len(components) == 1
    comp_faces, comp_points = components[0]
    assert comp_faces.dtype == np.int64  # Preserves int64


def test_split_into_components_pointcloud_not_supported():
    """Test that PointCloud raises TypeError."""
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    labels = np.array([0, 1, 0], dtype=np.int32)
    pc = tf.PointCloud(points)

    with pytest.raises(TypeError, match="PointCloud is not supported"):
        tf.split_into_components(pc, labels)


def test_split_into_components_invalid_V():
    """Test that invalid V value raises ValueError."""
    # V=5 is not supported
    indices = np.array([[0, 1, 2, 3, 4]], dtype=np.int32)
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0.5, 0.5, 1]], dtype=np.float32)
    labels = np.array([0], dtype=np.int32)

    with pytest.raises(ValueError, match="Fixed-size indices must have 2.*or 3.*columns"):
        tf.split_into_components((indices, points), labels)


def test_split_into_components_v4_not_supported():
    """Test that V=4 (quads) raises ValueError - use OffsetBlockedArray instead."""
    quads = np.array([[0, 1, 2, 3]], dtype=np.int32)
    points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32)
    labels = np.array([0], dtype=np.int32)

    with pytest.raises(ValueError, match="Fixed-size indices must have 2.*or 3.*columns"):
        tf.split_into_components((quads, points), labels)


def test_split_into_components_invalid_dims():
    """Test that invalid dims raises ValueError."""
    # 4D points not supported
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    points = np.array([[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
    labels = np.array([0], dtype=np.int32)

    with pytest.raises(ValueError, match="points must have 2 or 3 dimensions"):
        tf.split_into_components((faces, points), labels)


def test_split_into_components_invalid_type():
    """Test error for unsupported input type."""
    invalid_input = "not a valid input"
    labels = np.array([0], dtype=np.int32)

    with pytest.raises(TypeError, match="Expected tuple .* Mesh, or EdgeMesh"):
        tf.split_into_components(invalid_input, labels)


def test_split_into_components_mesh_invalid_ngon():
    """Test that Mesh with invalid NGon raises ValueError."""
    # Create a "mesh" with V=2 (should use EdgeMesh instead)
    edges = np.array([[0, 1]], dtype=np.int32)
    points = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)

    # Mesh only supports NGon=3 (triangles) for fixed-size or dynamic for variable-sized
    with pytest.raises(ValueError):
        # This will fail when constructing the Mesh
        _mesh = tf.Mesh(edges, points)


# ==============================================================================
# Component Reindexing Tests
# ==============================================================================

@pytest.mark.parametrize("dims", DIMS)
@pytest.mark.parametrize("ngon", NGONS)
def test_split_into_components_point_reindexing(dims, ngon):
    """Test that unused points are removed in each component."""
    if dims == 2:
        # Create mesh with unused point
        points = np.array([
            [0.0, 0.0],  # 0 - used by component 0
            [1.0, 0.0],  # 1 - used by component 0
            [0.5, 1.0],  # 2 - used by component 0
            [5.0, 5.0],  # 3 - UNUSED
            [2.0, 0.0],  # 4 - used by component 1
            [2.5, 1.0],  # 5 - used by component 1
            [3.0, 0.0],  # 6 - used by component 1
        ], dtype=np.float32)
    else:
        points = np.array([
            [0.0, 0.0, 0.0],  # 0 - used by component 0
            [1.0, 0.0, 0.0],  # 1 - used by component 0
            [0.5, 1.0, 0.0],  # 2 - used by component 0
            [5.0, 5.0, 5.0],  # 3 - UNUSED
            [2.0, 0.0, 0.0],  # 4 - used by component 1
            [2.5, 1.0, 0.0],  # 5 - used by component 1
            [3.0, 0.0, 0.0],  # 6 - used by component 1
        ], dtype=np.float32)

    faces = np.array([
        [0, 1, 2],  # Component 0 (uses points 0,1,2)
        [4, 5, 6],  # Component 1 (uses points 4,5,6)
    ], dtype=np.int32)

    labels = np.array([0, 1], dtype=np.int32)
    mesh = tf.Mesh(faces, points)

    components, comp_labels = tf.split_into_components(mesh, labels)

    # Each component should have exactly 3 points (unused point 3 removed)
    for comp_faces, comp_points in components:
        assert len(comp_points) == 3, f"Expected 3 points per component, got {len(comp_points)}"


# ==============================================================================
# Main runner
# ==============================================================================

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
