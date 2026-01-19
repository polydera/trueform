"""
Test k_rings and neighborhoods functionality

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


def create_linear_mesh(index_dtype, real_dtype):
    """Create a linear mesh where vertices are arranged in a line.

    This creates a strip of triangles along the x-axis for predictable
    neighborhood testing.
    """
    # 5 triangles in a strip: vertices 0-1-2, 1-3-2, 2-3-4, 3-5-4, 4-5-6
    faces = np.array([
        [0, 1, 2],
        [1, 3, 2],
        [2, 3, 4],
        [3, 5, 4],
        [4, 5, 6]
    ], dtype=index_dtype)
    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0],
        [1.5, 1.0, 0.0],
        [1.0, 2.0, 0.0],
        [2.0, 2.0, 0.0],
        [1.5, 3.0, 0.0]
    ], dtype=real_dtype)
    return faces, points


def build_connectivity(faces, n_points, index_dtype):
    """Build vertex connectivity (1-ring) from faces."""
    fm = tf.cell_membership(faces, n_points)
    vl = tf.vertex_link_faces(faces, fm)
    return vl


# ==============================================================================
# k_rings Tests - Basic Functionality
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_k_rings_basic(index_dtype, real_dtype):
    """Test k_rings returns correct structure."""
    faces, points = create_triangle_mesh(index_dtype, real_dtype)
    n_points = len(points)
    connectivity = build_connectivity(faces, n_points, index_dtype)

    result = tf.k_rings(connectivity, k=1)

    # Should return OffsetBlockedArray
    assert isinstance(result, tf.OffsetBlockedArray)
    # Should have one entry per vertex
    assert len(result) == n_points
    # Should preserve dtype
    assert result.dtype == index_dtype


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_k_rings_k1_equals_vertex_link(index_dtype, real_dtype):
    """Test k_rings with k=1 matches vertex_link (1-ring neighbors)."""
    faces, points = create_triangle_mesh(index_dtype, real_dtype)
    n_points = len(points)
    connectivity = build_connectivity(faces, n_points, index_dtype)

    k1 = tf.k_rings(connectivity, k=1)

    # k=1 should give the same neighbors as the input connectivity
    for i in range(n_points):
        assert set(k1[i]) == set(connectivity[i]), \
            f"k=1 neighbors for vertex {i} should match vertex_link"


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_k_rings_k2_superset_of_k1(index_dtype, real_dtype):
    """Test k_rings with k=2 is a superset of k=1."""
    faces, points = create_larger_triangle_mesh(index_dtype, real_dtype)
    n_points = len(points)
    connectivity = build_connectivity(faces, n_points, index_dtype)

    k1 = tf.k_rings(connectivity, k=1)
    k2 = tf.k_rings(connectivity, k=2)

    # k=2 should include all k=1 neighbors
    for i in range(n_points):
        k1_set = set(k1[i])
        k2_set = set(k2[i])
        assert k1_set.issubset(k2_set), \
            f"k=2 neighbors for vertex {i} should include all k=1 neighbors"


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_k_rings_correctness(index_dtype, real_dtype):
    """Test k_rings returns correct neighbors for specific mesh."""
    # Simple mesh: faces [0,1,2], [1,3,2]
    # Vertex connectivity:
    #   0 -> {1, 2}
    #   1 -> {0, 2, 3}
    #   2 -> {0, 1, 3}
    #   3 -> {1, 2}
    faces, points = create_triangle_mesh(index_dtype, real_dtype)
    n_points = len(points)
    connectivity = build_connectivity(faces, n_points, index_dtype)

    k1 = tf.k_rings(connectivity, k=1)
    k2 = tf.k_rings(connectivity, k=2)

    # k=1 from vertex 0: {1, 2}
    assert set(k1[0]) == {1, 2}

    # k=2 from vertex 0: neighbors of {1, 2} = {0, 2, 3} ∪ {0, 1, 3} - {0} = {1, 2, 3}
    # (without the seed vertex 0)
    assert set(k2[0]) == {1, 2, 3}


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_k_rings_inclusive(index_dtype, real_dtype):
    """Test k_rings with inclusive=True includes seed vertex."""
    faces, points = create_triangle_mesh(index_dtype, real_dtype)
    n_points = len(points)
    connectivity = build_connectivity(faces, n_points, index_dtype)

    k1_exclusive = tf.k_rings(connectivity, k=1, inclusive=False)
    k1_inclusive = tf.k_rings(connectivity, k=1, inclusive=True)

    for i in range(n_points):
        # Exclusive should not contain seed
        assert i not in k1_exclusive[i], \
            f"Exclusive k-ring should not contain seed vertex {i}"

        # Inclusive should contain seed
        assert i in k1_inclusive[i], \
            f"Inclusive k-ring should contain seed vertex {i}"

        # Inclusive should have exactly one more element (the seed)
        assert len(k1_inclusive[i]) == len(k1_exclusive[i]) + 1


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_k_rings_larger_k(index_dtype):
    """Test k_rings with larger k value."""
    faces, points = create_linear_mesh(index_dtype, np.float32)
    n_points = len(points)
    connectivity = build_connectivity(faces, n_points, index_dtype)

    # With k large enough, all vertices should be reachable
    k_large = tf.k_rings(connectivity, k=10)

    # In a connected mesh, k=10 should reach all vertices
    for i in range(n_points):
        # Should reach all other vertices
        assert len(k_large[i]) == n_points - 1, \
            f"Large k should reach all {n_points - 1} other vertices from vertex {i}"


# ==============================================================================
# k_rings Tests - Return Types
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_k_rings_return_type(index_dtype):
    """Test k_rings returns OffsetBlockedArray with correct dtype."""
    faces = np.array([[0, 1, 2], [1, 2, 3]], dtype=index_dtype)
    fm = tf.cell_membership(faces, 4)
    connectivity = tf.vertex_link_faces(faces, fm)

    result = tf.k_rings(connectivity, k=1)

    assert isinstance(result, tf.OffsetBlockedArray)
    assert result.offsets.dtype == index_dtype
    assert result.data.dtype == index_dtype


# ==============================================================================
# k_rings Tests - Error Validation
# ==============================================================================

def test_k_rings_invalid_connectivity_type():
    """Test k_rings rejects invalid connectivity type."""
    with pytest.raises(TypeError, match="connectivity must be OffsetBlockedArray"):
        tf.k_rings("invalid", k=1)


def test_k_rings_invalid_connectivity_array():
    """Test k_rings rejects numpy array as connectivity."""
    connectivity = np.array([[1, 2], [0, 2], [0, 1]], dtype=np.int32)
    with pytest.raises(TypeError, match="connectivity must be OffsetBlockedArray"):
        tf.k_rings(connectivity, k=1)


def test_k_rings_invalid_k_zero():
    """Test k_rings rejects k=0."""
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    fm = tf.cell_membership(faces, 3)
    connectivity = tf.vertex_link_faces(faces, fm)

    with pytest.raises(ValueError, match="k must be a positive integer"):
        tf.k_rings(connectivity, k=0)


def test_k_rings_invalid_k_negative():
    """Test k_rings rejects negative k."""
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    fm = tf.cell_membership(faces, 3)
    connectivity = tf.vertex_link_faces(faces, fm)

    with pytest.raises(ValueError, match="k must be a positive integer"):
        tf.k_rings(connectivity, k=-1)


def test_k_rings_invalid_k_float():
    """Test k_rings rejects float k."""
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    fm = tf.cell_membership(faces, 3)
    connectivity = tf.vertex_link_faces(faces, fm)

    with pytest.raises((TypeError, ValueError)):
        tf.k_rings(connectivity, k=1.5)


# ==============================================================================
# neighborhoods Tests - Basic Functionality
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_neighborhoods_basic(index_dtype, real_dtype):
    """Test neighborhoods returns correct structure."""
    faces, points = create_triangle_mesh(index_dtype, real_dtype)
    n_points = len(points)
    connectivity = build_connectivity(faces, n_points, index_dtype)

    result = tf.neighborhoods(connectivity, points, radius=2.0)

    # Should return OffsetBlockedArray
    assert isinstance(result, tf.OffsetBlockedArray)
    # Should have one entry per vertex
    assert len(result) == n_points
    # Should preserve index dtype
    assert result.dtype == index_dtype


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_neighborhoods_small_radius(index_dtype, real_dtype):
    """Test neighborhoods with small radius returns fewer neighbors."""
    faces, points = create_larger_triangle_mesh(index_dtype, real_dtype)
    n_points = len(points)
    connectivity = build_connectivity(faces, n_points, index_dtype)

    small = tf.neighborhoods(connectivity, points, radius=0.5)
    large = tf.neighborhoods(connectivity, points, radius=5.0)

    # Small radius should give fewer or equal neighbors
    for i in range(n_points):
        assert len(small[i]) <= len(large[i]), \
            f"Small radius should give fewer neighbors for vertex {i}"


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_neighborhoods_large_radius(index_dtype, real_dtype):
    """Test neighborhoods with large radius returns all neighbors."""
    faces, points = create_triangle_mesh(index_dtype, real_dtype)
    n_points = len(points)
    connectivity = build_connectivity(faces, n_points, index_dtype)

    # Very large radius should reach all connected vertices
    result = tf.neighborhoods(connectivity, points, radius=100.0)

    for i in range(n_points):
        # Should reach all other vertices in a connected mesh
        assert len(result[i]) == n_points - 1, \
            f"Large radius should reach all {n_points - 1} other vertices from {i}"


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_neighborhoods_correctness(index_dtype, real_dtype):
    """Test neighborhoods returns vertices within specified radius."""
    faces, points = create_triangle_mesh(index_dtype, real_dtype)
    n_points = len(points)
    connectivity = build_connectivity(faces, n_points, index_dtype)

    radius = 1.5
    result = tf.neighborhoods(connectivity, points, radius=radius)

    # Verify all returned neighbors are within radius
    for i in range(n_points):
        seed_pos = points[i]
        for neighbor in result[i]:
            neighbor_pos = points[neighbor]
            dist = np.linalg.norm(neighbor_pos - seed_pos)
            assert dist <= radius, \
                f"Neighbor {neighbor} of vertex {i} is at distance {dist}, exceeds radius {radius}"


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_neighborhoods_inclusive(index_dtype, real_dtype):
    """Test neighborhoods with inclusive=True includes seed vertex."""
    faces, points = create_triangle_mesh(index_dtype, real_dtype)
    n_points = len(points)
    connectivity = build_connectivity(faces, n_points, index_dtype)

    exclusive = tf.neighborhoods(connectivity, points, radius=2.0, inclusive=False)
    inclusive = tf.neighborhoods(connectivity, points, radius=2.0, inclusive=True)

    for i in range(n_points):
        # Exclusive should not contain seed
        assert i not in exclusive[i], \
            f"Exclusive neighborhood should not contain seed vertex {i}"

        # Inclusive should contain seed
        assert i in inclusive[i], \
            f"Inclusive neighborhood should contain seed vertex {i}"


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_neighborhoods_distance_monotonicity(index_dtype, real_dtype):
    """Test neighborhoods with increasing radius gives non-decreasing neighbor counts."""
    faces, points = create_larger_triangle_mesh(index_dtype, real_dtype)
    n_points = len(points)
    connectivity = build_connectivity(faces, n_points, index_dtype)

    radii = [0.5, 1.0, 1.5, 2.0, 3.0]
    prev_counts = [0] * n_points

    for radius in radii:
        result = tf.neighborhoods(connectivity, points, radius=radius)
        for i in range(n_points):
            current_count = len(result[i])
            assert current_count >= prev_counts[i], \
                f"Larger radius should not decrease neighbor count for vertex {i}"
            prev_counts[i] = current_count


# ==============================================================================
# neighborhoods Tests - Return Types
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_neighborhoods_return_type(index_dtype, real_dtype):
    """Test neighborhoods returns OffsetBlockedArray with correct dtype."""
    faces = np.array([[0, 1, 2], [1, 2, 3]], dtype=index_dtype)
    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0],
        [1.5, 1.0, 0.0]
    ], dtype=real_dtype)
    fm = tf.cell_membership(faces, 4)
    connectivity = tf.vertex_link_faces(faces, fm)

    result = tf.neighborhoods(connectivity, points, radius=1.0)

    assert isinstance(result, tf.OffsetBlockedArray)
    assert result.offsets.dtype == index_dtype
    assert result.data.dtype == index_dtype


# ==============================================================================
# neighborhoods Tests - Error Validation
# ==============================================================================

def test_neighborhoods_invalid_connectivity_type():
    """Test neighborhoods rejects invalid connectivity type."""
    points = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
    with pytest.raises(TypeError, match="connectivity must be OffsetBlockedArray"):
        tf.neighborhoods("invalid", points, radius=1.0)


def test_neighborhoods_invalid_points_type():
    """Test neighborhoods rejects invalid points type."""
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    fm = tf.cell_membership(faces, 3)
    connectivity = tf.vertex_link_faces(faces, fm)

    with pytest.raises(TypeError, match="points must be np.ndarray"):
        tf.neighborhoods(connectivity, "invalid", radius=1.0)


def test_neighborhoods_invalid_points_dtype():
    """Test neighborhoods rejects invalid points dtype."""
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    points = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0]], dtype=np.int32)
    fm = tf.cell_membership(faces, 3)
    connectivity = tf.vertex_link_faces(faces, fm)

    with pytest.raises(TypeError, match="points dtype must be float32 or float64"):
        tf.neighborhoods(connectivity, points, radius=1.0)


def test_neighborhoods_invalid_points_shape():
    """Test neighborhoods rejects 1D points array."""
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    points = np.array([0, 0, 0, 1, 0, 0], dtype=np.float32)  # 1D
    fm = tf.cell_membership(faces, 3)
    connectivity = tf.vertex_link_faces(faces, fm)

    with pytest.raises(ValueError, match="points must be 2D array"):
        tf.neighborhoods(connectivity, points, radius=1.0)


def test_neighborhoods_invalid_points_dims():
    """Test neighborhoods rejects non-3D points."""
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    points = np.array([[0, 0], [1, 0], [0.5, 1]], dtype=np.float32)  # 2D points
    fm = tf.cell_membership(faces, 3)
    connectivity = tf.vertex_link_faces(faces, fm)

    with pytest.raises(ValueError, match="points must have 3 dimensions"):
        tf.neighborhoods(connectivity, points, radius=1.0)


def test_neighborhoods_invalid_radius_zero():
    """Test neighborhoods rejects radius=0."""
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    points = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0]], dtype=np.float32)
    fm = tf.cell_membership(faces, 3)
    connectivity = tf.vertex_link_faces(faces, fm)

    with pytest.raises(ValueError, match="radius must be positive"):
        tf.neighborhoods(connectivity, points, radius=0.0)


def test_neighborhoods_invalid_radius_negative():
    """Test neighborhoods rejects negative radius."""
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    points = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0]], dtype=np.float32)
    fm = tf.cell_membership(faces, 3)
    connectivity = tf.vertex_link_faces(faces, fm)

    with pytest.raises(ValueError, match="radius must be positive"):
        tf.neighborhoods(connectivity, points, radius=-1.0)


# ==============================================================================
# Integration Tests - k_rings and neighborhoods together
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_k_rings_neighborhoods_consistency(index_dtype, real_dtype):
    """Test k_rings and neighborhoods give consistent results on unit mesh."""
    # Create a unit mesh where distances match hop counts
    faces = np.array([
        [0, 1, 2],
        [1, 3, 2]
    ], dtype=index_dtype)
    # Points on a unit grid
    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0],
        [1.5, 1.0, 0.0]
    ], dtype=real_dtype)
    n_points = len(points)
    connectivity = build_connectivity(faces, n_points, index_dtype)

    k1 = tf.k_rings(connectivity, k=1)
    # Use large radius to get all neighbors within 1 hop
    # Max distance in this mesh is about 1.5
    neigh_large = tf.neighborhoods(connectivity, points, radius=2.0)

    # k=1 should give subset of large-radius neighborhood (or equal)
    for i in range(n_points):
        k1_set = set(k1[i])
        neigh_set = set(neigh_large[i])
        # k=1 neighbors should be subset of large radius neighbors
        assert k1_set.issubset(neigh_set), \
            f"k=1 neighbors should be subset of large-radius neighbors for vertex {i}"


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_both_functions_from_mesh(index_dtype, real_dtype):
    """Test both functions work with connectivity from Mesh object."""
    faces, points = create_triangle_mesh(index_dtype, real_dtype)
    mesh = tf.Mesh(faces, points)

    # Get connectivity from mesh
    connectivity = mesh.vertex_link

    # Both should work
    k_result = tf.k_rings(connectivity, k=1)
    n_result = tf.neighborhoods(connectivity, points, radius=1.5)

    assert isinstance(k_result, tf.OffsetBlockedArray)
    assert isinstance(n_result, tf.OffsetBlockedArray)
    assert len(k_result) == mesh.number_of_points
    assert len(n_result) == mesh.number_of_points


# ==============================================================================
# Main runner
# ==============================================================================

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
