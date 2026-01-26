"""
Tests for Mesh × Mesh neighbor_search

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""
import sys
import pytest
import numpy as np
import trueform as tf


# Type combinations
INDEX_DTYPES = [np.int32, np.int64]
REAL_DTYPES = [np.float32, np.float64]
# ngon: 3 for triangles, 'dyn' for dynamic
NGONS = [3, 'dyn']


def create_square_mesh_2d(points, index_dtype, ngon):
    """Create a square mesh from 4 points (2D)"""
    if ngon == 3:
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=index_dtype)
        return tf.Mesh(faces, points)
    else:  # dynamic - create as one quad
        offsets = np.array([0, 4], dtype=index_dtype)
        data = np.array([0, 1, 2, 3], dtype=index_dtype)
        faces = tf.OffsetBlockedArray(offsets, data)
        return tf.Mesh(faces, points)


def create_square_mesh_3d(points, index_dtype, ngon):
    """Create a square mesh from 4 points (3D)"""
    if ngon == 3:
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=index_dtype)
        return tf.Mesh(faces, points)
    else:  # dynamic - create as one quad
        offsets = np.array([0, 4], dtype=index_dtype)
        data = np.array([0, 1, 2, 3], dtype=index_dtype)
        faces = tf.OffsetBlockedArray(offsets, data)
        return tf.Mesh(faces, points)


@pytest.mark.parametrize("mesh0_index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("mesh1_index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
@pytest.mark.parametrize("ngon0", NGONS)
@pytest.mark.parametrize("ngon1", NGONS)
def test_mesh_neighbor_search_mesh_2d_parallel_meshes(mesh0_index_dtype, mesh1_index_dtype, real_dtype, ngon0, ngon1):
    """Test 2D: two parallel meshes with known gap"""
    # Mesh 0: square at y=[0,1]
    points0 = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=real_dtype)
    mesh0 = create_square_mesh_2d(points0, mesh0_index_dtype, ngon0)

    # Mesh 1: square at y=[2,3] (gap of 1.0)
    points1 = np.array([[0, 2], [1, 2], [1, 3], [0, 3]], dtype=real_dtype)
    mesh1 = create_square_mesh_2d(points1, mesh1_index_dtype, ngon1)

    result = tf.neighbor_search(mesh0, mesh1)
    assert result is not None

    (idx0, idx1), (dist, pt0, pt1) = result

    # Closest distance should be 1.0 (gap between y=1 and y=2)
    # Returns squared distance
    expected_dist2 = 1.0**2
    assert abs(dist - expected_dist2) < 1e-4

    # pt0 should be on top edge of mesh0 (y=1)
    assert abs(pt0[1] - 1.0) < 1e-4
    # pt1 should be on bottom edge of mesh1 (y=2)
    assert abs(pt1[1] - 2.0) < 1e-4

    # Test symmetry
    result_sym = tf.neighbor_search(mesh1, mesh0)
    (idx1_sym, idx0_sym), (dist_sym, pt1_sym, pt0_sym) = result_sym

    assert abs(dist_sym - dist) < 1e-6
    assert np.allclose(pt0_sym, pt0, atol=1e-5)
    assert np.allclose(pt1_sym, pt1, atol=1e-5)


@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
@pytest.mark.parametrize("ngon0", NGONS)
@pytest.mark.parametrize("ngon1", NGONS)
def test_mesh_neighbor_search_mesh_3d_parallel_planes(real_dtype, ngon0, ngon1):
    """Test 3D: two parallel planar meshes at known z-separation"""
    # Mesh 0: square in xy-plane at z=0
    points0 = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=real_dtype)
    mesh0 = create_square_mesh_3d(points0, np.int32, ngon0)

    # Mesh 1: square in xy-plane at z=2
    points1 = np.array([[0, 0, 2], [1, 0, 2], [1, 1, 2], [0, 1, 2]], dtype=real_dtype)
    mesh1 = create_square_mesh_3d(points1, np.int32, ngon1)

    result = tf.neighbor_search(mesh0, mesh1)
    assert result is not None

    (idx0, idx1), (dist, pt0, pt1) = result

    # Distance should be 2.0 (z separation), returns squared
    assert abs(dist - 4.0) < 1e-5

    # pt0 should be at z=0
    assert abs(pt0[2]) < 1e-4
    # pt1 should be at z=2
    assert abs(pt1[2] - 2.0) < 1e-4

    # Test symmetry
    result_sym = tf.neighbor_search(mesh1, mesh0)
    ((_, _), (dist_sym, pt1_sym, pt0_sym)) = result_sym
    assert abs(dist_sym - 4.0) < 1e-6
    assert np.allclose(pt0_sym, pt0, atol=1e-5)
    assert np.allclose(pt1_sym, pt1, atol=1e-5)


@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_mesh_neighbor_search_mesh_2d_radius_hit(real_dtype):
    """Test 2D with radius - within radius"""
    # Triangle mesh 0
    points0 = np.array([[0, 0], [1, 0], [0.5, 1]], dtype=real_dtype)
    faces0 = np.array([[0, 1, 2]], dtype=np.int32)

    # Triangle mesh 1 nearby
    points1 = np.array([[0, 1.5], [1, 1.5], [0.5, 2.5]], dtype=real_dtype)
    faces1 = np.array([[0, 1, 2]], dtype=np.int32)

    mesh0 = tf.Mesh(faces0, points0)
    mesh1 = tf.Mesh(faces1, points1)

    # Closest distance should be around 0.5, so radius=1.0 should find it
    result = tf.neighbor_search(mesh0, mesh1, radius=1.0)
    assert result is not None

    ((_, _), (dist, _, _)) = result
    assert dist < 1.0


@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_mesh_neighbor_search_mesh_2d_radius_miss(real_dtype):
    """Test 2D with radius - outside radius"""
    # Triangle mesh 0
    points0 = np.array([[0, 0], [1, 0], [0.5, 1]], dtype=real_dtype)
    faces0 = np.array([[0, 1, 2]], dtype=np.int32)

    # Triangle mesh 1 far away
    points1 = np.array([[10, 10], [11, 10], [10.5, 11]], dtype=real_dtype)
    faces1 = np.array([[0, 1, 2]], dtype=np.int32)

    mesh0 = tf.Mesh(faces0, points0)
    mesh1 = tf.Mesh(faces1, points1)

    result = tf.neighbor_search(mesh0, mesh1, radius=1.0)
    assert result is None


@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_mesh_neighbor_search_mesh_with_transformation(real_dtype):
    """Test neighbor search with transformed mesh"""
    # Triangle mesh 0 at origin
    points0 = np.array([[0, 0], [1, 0], [0.5, 1]], dtype=real_dtype)
    faces0 = np.array([[0, 1, 2]], dtype=np.int32)

    # Triangle mesh 1
    points1 = np.array([[0, 0], [1, 0], [0.5, 1]], dtype=real_dtype)
    faces1 = np.array([[0, 1, 2]], dtype=np.int32)

    mesh0 = tf.Mesh(faces0, points0)
    mesh1 = tf.Mesh(faces1, points1)

    # Transform mesh1 by offset (0, 3)
    transform = np.eye(3, dtype=real_dtype)
    transform[1, 2] = 3.0  # y offset
    mesh1.transformation = transform

    result = tf.neighbor_search(mesh0, mesh1)
    assert result is not None

    ((_, _), (dist, pt0, pt1)) = result

    # Distance should be about 2.0 (from y=1 to y=3), returns squared
    assert abs(dist - 4.0) < 1e-4

    # pt0 should be at top of mesh0 (y around 1.0)
    assert pt0[1] > 0.9
    # pt1 should be at bottom of transformed mesh1 (y around 3.0)
    assert pt1[1] > 2.9


def test_mesh_neighbor_search_mesh_dimension_mismatch():
    """Test that dimension mismatch raises error"""
    points2d = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32)
    faces2d = np.array([[0, 1, 2]], dtype=np.int32)

    points3d = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    faces3d = np.array([[0, 1, 2]], dtype=np.int32)

    mesh2d = tf.Mesh(faces2d, points2d)
    mesh3d = tf.Mesh(faces3d, points3d)

    with pytest.raises(ValueError, match="Dimension mismatch"):
        tf.neighbor_search(mesh2d, mesh3d)


@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_mesh_neighbor_search_mesh_offset_squares_dynamic(real_dtype):
    """Test two offset dynamic squares with unambiguous closest point"""
    # Dynamic mesh 0: unit square at origin (as quad)
    points0 = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=real_dtype)
    offsets0 = np.array([0, 4], dtype=np.int32)
    data0 = np.array([0, 1, 2, 3], dtype=np.int32)
    faces0 = tf.OffsetBlockedArray(offsets0, data0)

    # Dynamic mesh 1: unit square offset by (2, 0) (as quad)
    points1 = np.array([[2, 0], [3, 0], [3, 1], [2, 1]], dtype=real_dtype)
    offsets1 = np.array([0, 4], dtype=np.int32)
    data1 = np.array([0, 1, 2, 3], dtype=np.int32)
    faces1 = tf.OffsetBlockedArray(offsets1, data1)

    mesh0 = tf.Mesh(faces0, points0)
    mesh1 = tf.Mesh(faces1, points1)

    result = tf.neighbor_search(mesh0, mesh1)
    assert result is not None

    ((_, _), (dist, pt0, pt1)) = result

    # Distance should be 1.0 (gap between x=1 and x=2), returns squared
    assert abs(dist - 1.0) < 1e-5

    # pt0 should be on right edge of mesh0 (x=1)
    assert abs(pt0[0] - 1.0) < 1e-4
    # pt1 should be on left edge of mesh1 (x=2)
    assert abs(pt1[0] - 2.0) < 1e-4


@pytest.mark.parametrize("mesh0_index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("mesh1_index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_mesh_neighbor_search_mesh_all_index_combinations(mesh0_index_dtype, mesh1_index_dtype, real_dtype):
    """Test all combinations of index types work correctly"""
    # Simple triangle mesh 0
    points0 = np.array([[0, 0], [1, 0], [0.5, 1]], dtype=real_dtype)
    faces0 = np.array([[0, 1, 2]], dtype=mesh0_index_dtype)

    # Simple triangle mesh 1 above
    points1 = np.array([[0, 2], [1, 2], [0.5, 3]], dtype=real_dtype)
    faces1 = np.array([[0, 1, 2]], dtype=mesh1_index_dtype)

    mesh0 = tf.Mesh(faces0, points0)
    mesh1 = tf.Mesh(faces1, points1)

    # Should work for all index type combinations
    result = tf.neighbor_search(mesh0, mesh1)
    assert result is not None

    # Test symmetry also works
    result_sym = tf.neighbor_search(mesh1, mesh0)
    assert result_sym is not None


@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_mesh_neighbor_search_mesh_3d_perpendicular_dynamic(real_dtype):
    """Test 3D: two perpendicular planar dynamic meshes"""
    # Mesh 0: square in xy-plane at z=0 (as quad)
    points0 = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=real_dtype)
    offsets0 = np.array([0, 4], dtype=np.int32)
    data0 = np.array([0, 1, 2, 3], dtype=np.int32)
    faces0 = tf.OffsetBlockedArray(offsets0, data0)

    # Mesh 1: square in xz-plane at y=2 (as quad)
    points1 = np.array([[0, 2, 0], [1, 2, 0], [1, 2, 1], [0, 2, 1]], dtype=real_dtype)
    offsets1 = np.array([0, 4], dtype=np.int32)
    data1 = np.array([0, 1, 2, 3], dtype=np.int32)
    faces1 = tf.OffsetBlockedArray(offsets1, data1)

    mesh0 = tf.Mesh(faces0, points0)
    mesh1 = tf.Mesh(faces1, points1)

    result = tf.neighbor_search(mesh0, mesh1)
    assert result is not None

    ((_, _), (dist, pt0, pt1)) = result

    # Closest distance should be 1.0 (from y=1 to y=2), returns squared
    assert abs(dist - 1.0) < 1e-5

    # pt0 should have y=1
    assert abs(pt0[1] - 1.0) < 1e-4
    # pt1 should have y=2, z=0
    assert abs(pt1[1] - 2.0) < 1e-4
    assert abs(pt1[2]) < 1e-4


@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_mesh_neighbor_search_mesh_mixed_tri_dynamic(real_dtype):
    """Test triangle mesh vs dynamic mesh"""
    # Triangle mesh 0
    points0 = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=real_dtype)
    faces0 = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    mesh0 = tf.Mesh(faces0, points0)

    # Dynamic mesh 1 (quad) offset by (2, 0)
    points1 = np.array([[2, 0], [3, 0], [3, 1], [2, 1]], dtype=real_dtype)
    offsets1 = np.array([0, 4], dtype=np.int32)
    data1 = np.array([0, 1, 2, 3], dtype=np.int32)
    faces1 = tf.OffsetBlockedArray(offsets1, data1)
    mesh1 = tf.Mesh(faces1, points1)

    result = tf.neighbor_search(mesh0, mesh1)
    assert result is not None

    ((_, _), (dist, pt0, pt1)) = result

    # Distance should be 1.0 (gap between x=1 and x=2), returns squared
    assert abs(dist - 1.0) < 1e-5


if __name__ == "__main__":
    # Run tests with verbose output
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
