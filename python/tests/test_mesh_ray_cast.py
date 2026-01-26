"""
Tests for mesh ray_cast functionality

Uses pytest parametrization to efficiently test all type combinations:
- Index types: int32, int64
- Real types: float32, float64
- Mesh types: triangles, dynamic (variable-size)
- Dims: 2D, 3D

Copyright (c) 2025 Žiga Sajovic, XLAB
"""

import sys
import numpy as np
import pytest
import trueform as tf

# Type combinations to test
INDEX_DTYPES = [np.int32, np.int64]
REAL_DTYPES = [np.float32, np.float64]
MESH_TYPES = ['triangle', 'dynamic']  # triangles and dynamic (variable-size)


def create_2d_triangle_mesh(index_dtype, real_dtype):
    """Create a simple 2D triangle mesh"""
    # Single triangle in xy-plane: vertices at [0,0], [1,0], [0.5,1]
    faces = np.array([[0, 1, 2]], dtype=index_dtype)
    points = np.array([[0, 0], [1, 0], [0.5, 1]], dtype=real_dtype)
    return tf.Mesh(faces, points)


def create_3d_triangle_mesh(index_dtype, real_dtype):
    """Create a simple 3D triangle mesh"""
    # Single triangle in xy-plane at z=0: vertices at [0,0,0], [1,0,0], [0.5,1,0]
    faces = np.array([[0, 1, 2]], dtype=index_dtype)
    points = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0]], dtype=real_dtype)
    return tf.Mesh(faces, points)


def create_2d_dynamic_mesh(index_dtype, real_dtype):
    """Create a simple 2D dynamic mesh with variable-size polygons"""
    # Single quad as dynamic: vertices at [0,0], [1,0], [1,1], [0,1]
    offsets = np.array([0, 4], dtype=index_dtype)
    data = np.array([0, 1, 2, 3], dtype=index_dtype)
    faces = tf.OffsetBlockedArray(offsets, data)
    points = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=real_dtype)
    return tf.Mesh(faces, points)


def create_3d_dynamic_mesh(index_dtype, real_dtype):
    """Create a simple 3D dynamic mesh with variable-size polygons"""
    # Single quad in xy-plane at z=0 as dynamic
    offsets = np.array([0, 4], dtype=index_dtype)
    data = np.array([0, 1, 2, 3], dtype=index_dtype)
    faces = tf.OffsetBlockedArray(offsets, data)
    points = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=real_dtype
    )
    return tf.Mesh(faces, points)


# Test fixtures for mesh creation
MESH_CREATORS = {
    (2, 'triangle'): create_2d_triangle_mesh,
    (3, 'triangle'): create_3d_triangle_mesh,
    (2, 'dynamic'): create_2d_dynamic_mesh,
    (3, 'dynamic'): create_3d_dynamic_mesh,
}


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
@pytest.mark.parametrize("dims", [2, 3])
@pytest.mark.parametrize("mesh_type", MESH_TYPES)
def test_mesh_ray_cast_hit(index_dtype, real_dtype, dims, mesh_type):
    """Test ray casting that should hit the mesh"""
    mesh_creator = MESH_CREATORS[(dims, mesh_type)]
    mesh = mesh_creator(index_dtype, real_dtype)

    if dims == 2:
        # Ray from left pointing right at center of mesh
        ray = tf.Ray(
            origin=np.array([-1.0, 0.5], dtype=real_dtype),
            direction=np.array([1.0, 0.0], dtype=real_dtype)
        )
    else:  # dims == 3
        # Ray from above pointing down at center of mesh
        ray = tf.Ray(
            origin=np.array([0.5, 0.5, 2.0], dtype=real_dtype),
            direction=np.array([0.0, 0.0, -1.0], dtype=real_dtype)
        )

    result = tf.ray_cast(ray, mesh)

    # Should hit the mesh
    assert result is not None, "Ray should intersect mesh"
    assert isinstance(result, tuple), "Result should be a tuple"
    assert len(result) == 2, "Result should be (face_idx, t)"

    face_idx, t = result
    assert isinstance(face_idx, (int, np.integer)), "Face index should be an integer"
    assert face_idx == 0, "Should hit face 0 (only face in mesh)"
    assert isinstance(t, (float, np.floating)), "t should be a float"
    assert t > 0, "t should be positive for ray traveling forward"


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
@pytest.mark.parametrize("dims", [2, 3])
@pytest.mark.parametrize("mesh_type", MESH_TYPES)
def test_mesh_ray_cast_miss(index_dtype, real_dtype, dims, mesh_type):
    """Test ray casting that should miss the mesh"""
    mesh_creator = MESH_CREATORS[(dims, mesh_type)]
    mesh = mesh_creator(index_dtype, real_dtype)

    if dims == 2:
        # Ray from left but above the mesh
        ray = tf.Ray(
            origin=np.array([-1.0, 5.0], dtype=real_dtype),
            direction=np.array([1.0, 0.0], dtype=real_dtype)
        )
    else:  # dims == 3
        # Ray from above but offset (should miss)
        ray = tf.Ray(
            origin=np.array([5.0, 5.0, 2.0], dtype=real_dtype),
            direction=np.array([0.0, 0.0, -1.0], dtype=real_dtype)
        )

    result = tf.ray_cast(ray, mesh)

    # Should miss the mesh
    assert result is None, "Ray should not intersect mesh"


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
@pytest.mark.parametrize("dims", [2, 3])
@pytest.mark.parametrize("mesh_type", MESH_TYPES)
def test_mesh_ray_cast_pointing_away(index_dtype, real_dtype, dims, mesh_type):
    """Test ray pointing away from mesh"""
    mesh_creator = MESH_CREATORS[(dims, mesh_type)]
    mesh = mesh_creator(index_dtype, real_dtype)

    if dims == 2:
        # Ray from left but pointing away
        ray = tf.Ray(
            origin=np.array([-1.0, 0.5], dtype=real_dtype),
            direction=np.array([-1.0, 0.0], dtype=real_dtype)
        )
    else:  # dims == 3
        # Ray from above but pointing up
        ray = tf.Ray(
            origin=np.array([0.5, 0.5, 2.0], dtype=real_dtype),
            direction=np.array([0.0, 0.0, 1.0], dtype=real_dtype)
        )

    result = tf.ray_cast(ray, mesh)

    # Should not hit (pointing away)
    assert result is None, "Ray pointing away should not intersect mesh"


@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_mesh_ray_cast_geometric_correctness_3d(real_dtype):
    """Test geometric correctness with known answers (3D triangle)"""
    # Triangle in xy-plane at z=0
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    points = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0]], dtype=real_dtype)
    mesh = tf.Mesh(faces, points)

    # Ray from above pointing down - should hit at t=2.0
    ray = tf.Ray(
        origin=np.array([0.5, 0.3, 2.0], dtype=real_dtype),
        direction=np.array([0.0, 0.0, -1.0], dtype=real_dtype)
    )

    result = tf.ray_cast(ray, mesh)
    assert result is not None, "Ray should hit triangle"

    face_idx, t = result
    assert face_idx == 0, "Should hit face 0"
    assert np.isclose(t, 2.0, atol=1e-5), f"Expected t=2.0, got {t}"

    # Verify hit point is on the plane
    hit_point = ray.origin + t * ray.direction
    assert np.isclose(hit_point[2], 0.0, atol=1e-5), "Hit point should be on z=0 plane"
    assert 0.0 <= hit_point[0] <= 1.0, "Hit point x should be in triangle bounds"
    assert 0.0 <= hit_point[1] <= 1.0, "Hit point y should be in triangle bounds"


@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_mesh_ray_cast_geometric_correctness_2d(real_dtype):
    """Test geometric correctness with known answers (2D triangle)"""
    # Triangle at [0,0], [1,0], [0.5,1]
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    points = np.array([[0, 0], [1, 0], [0.5, 1]], dtype=real_dtype)
    mesh = tf.Mesh(faces, points)

    # Ray from left pointing right through center - should hit
    ray = tf.Ray(
        origin=np.array([-2.0, 0.3], dtype=real_dtype),
        direction=np.array([1.0, 0.0], dtype=real_dtype)
    )

    result = tf.ray_cast(ray, mesh)
    assert result is not None, "Ray should hit triangle"

    face_idx, t = result
    assert face_idx == 0, "Should hit face 0"
    assert t > 0, "t should be positive"

    # Verify hit point is within triangle bounds
    hit_point = ray.origin + t * ray.direction
    assert 0.0 <= hit_point[0] <= 1.0, "Hit point x should be in triangle bounds"
    assert 0.0 <= hit_point[1] <= 1.0, "Hit point y should be in triangle bounds"


@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_mesh_ray_cast_with_transformation_2d(real_dtype):
    """Test ray casting with 2D transformation"""
    # Create triangle at origin
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    points = np.array([[0, 0], [1, 0], [0.5, 1]], dtype=real_dtype)
    mesh = tf.Mesh(faces, points)

    # Apply translation [5, 3]
    # Format: [r11 r12 tx]
    #         [r21 r22 ty]
    #         [  0   0  1]
    # After transformation, triangle is at [5,3], [6,3], [5.5,4]
    transformation = np.array([
        [1, 0, 5],
        [0, 1, 3],
        [0, 0, 1]
    ], dtype=real_dtype)

    mesh.transformation = transformation

    # Ray should hit transformed triangle
    ray = tf.Ray(
        origin=np.array([3.0, 3.5], dtype=real_dtype),  # To the left of transformed triangle
        direction=np.array([1.0, 0.0], dtype=real_dtype)
    )

    result = tf.ray_cast(ray, mesh)
    assert result is not None, "Ray should hit transformed mesh"

    face_idx, t = result
    assert face_idx == 0, "Should hit face 0"
    assert t > 0, "t should be positive"

    # Verify hit point is within transformed triangle bounds
    hit_point = ray.origin + t * ray.direction
    assert 5.0 <= hit_point[0] <= 6.0, f"Hit point x should be in transformed bounds, got {hit_point[0]}"


@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_mesh_ray_cast_with_transformation_3d(real_dtype):
    """Test ray casting with 3D transformation"""
    # Create triangle in xy-plane at z=0
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    points = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0]], dtype=real_dtype)
    mesh = tf.Mesh(faces, points)

    # Apply translation [10, 5, 2]
    # Format: [r11 r12 r13 tx]
    #         [r21 r22 r23 ty]
    #         [r31 r32 r33 tz]
    #         [  0   0   0  1]
    # After transformation, triangle is at z=2
    transformation = np.array([
        [1, 0, 0, 10],
        [0, 1, 0,  5],
        [0, 0, 1,  2],
        [0, 0, 0,  1]
    ], dtype=real_dtype)

    mesh.transformation = transformation

    # Ray from above transformed triangle pointing down
    ray = tf.Ray(
        origin=np.array([10.5, 5.3, 5.0], dtype=real_dtype),
        direction=np.array([0.0, 0.0, -1.0], dtype=real_dtype)
    )

    result = tf.ray_cast(ray, mesh)
    assert result is not None, "Ray should hit transformed mesh"

    face_idx, t = result
    assert face_idx == 0, "Should hit face 0"
    # Ray starts at z=5, mesh is at z=2, so t should be 3.0
    assert np.isclose(t, 3.0, atol=1e-4), f"Expected t=3.0, got {t}"

    # Verify hit point is at the transformed plane
    hit_point = ray.origin + t * ray.direction
    assert np.isclose(hit_point[2], 2.0, atol=1e-5), f"Hit point should be at z=2, got {hit_point[2]}"


@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_mesh_ray_cast_transformation_clear(real_dtype):
    """Test clearing transformation"""
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    points = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0]], dtype=real_dtype)
    mesh = tf.Mesh(faces, points)

    # Apply transformation
    transformation = np.array([
        [1, 0, 0, 10],
        [0, 1, 0,  5],
        [0, 0, 1,  2],
        [0, 0, 0,  1]
    ], dtype=real_dtype)
    mesh.transformation = transformation

    # Ray at original position should miss
    ray_original = tf.Ray(
        origin=np.array([0.5, 0.3, 2.0], dtype=real_dtype),
        direction=np.array([0.0, 0.0, -1.0], dtype=real_dtype)
    )
    result = tf.ray_cast(ray_original, mesh)
    assert result is None, "Ray at original position should miss transformed mesh"

    # Clear transformation
    mesh.transformation = None

    # Now ray at original position should hit
    result = tf.ray_cast(ray_original, mesh)
    assert result is not None, "Ray should hit after clearing transformation"

    face_idx, t = result
    assert face_idx == 0
    assert np.isclose(t, 2.0, atol=1e-5)


def test_mesh_ray_cast_dimension_mismatch():
    """Test that dimension mismatch raises an error"""
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    points = np.array([[0, 0], [1, 0], [0.5, 1]], dtype=np.float32)
    mesh = tf.Mesh(faces, points)  # 2D mesh

    # 3D ray
    ray_3d = tf.Ray(
        origin=np.array([0.5, 0.5, 2.0], dtype=np.float32),
        direction=np.array([0.0, 0.0, -1.0], dtype=np.float32)
    )

    with pytest.raises(ValueError, match="Dimension mismatch"):
        tf.ray_cast(ray_3d, mesh)


def test_mesh_ray_cast_dtype_mismatch():
    """Test that dtype mismatch raises an error"""
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    points = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0]], dtype=np.float32)
    mesh = tf.Mesh(faces, points)  # float32 mesh

    # float64 ray
    ray_f64 = tf.Ray(
        origin=np.array([0.5, 0.5, 2.0], dtype=np.float64),
        direction=np.array([0.0, 0.0, -1.0], dtype=np.float64)
    )

    with pytest.raises(TypeError, match="Dtype mismatch"):
        tf.ray_cast(ray_f64, mesh)


def test_mesh_ray_cast_multi_face():
    """Test ray casting on mesh with multiple faces"""
    # Create a mesh with 2 triangles
    faces = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int32)
    points = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 1]], dtype=np.float32
    )
    mesh = tf.Mesh(faces, points)

    # Ray should hit one of the faces
    ray = tf.Ray(
        origin=np.array([0.3, 0.3, 2.0], dtype=np.float32),
        direction=np.array([0.0, 0.0, -1.0], dtype=np.float32)
    )

    result = tf.ray_cast(ray, mesh)
    assert result is not None, "Ray should hit mesh"

    face_idx, t = result
    assert face_idx in [0, 1], "Should hit one of the two faces"
    assert t > 0, "t should be positive"


if __name__ == "__main__":
    # Run tests with verbose output
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
