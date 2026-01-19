"""
Tests for edge_mesh ray_cast functionality

Uses pytest parametrization to efficiently test all type combinations:
- Index types: int32, int64
- Real types: float32, float64
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


def create_2d_edge_mesh(index_dtype, real_dtype):
    """Create a simple 2D edge mesh with multiple edges"""
    # Multiple edges forming a path
    edges = np.array([[0, 1], [1, 2], [2, 3]], dtype=index_dtype)
    points = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=real_dtype)
    return tf.EdgeMesh(edges, points)


def create_3d_edge_mesh(index_dtype, real_dtype):
    """Create a simple 3D edge mesh with multiple edges"""
    # Multiple edges forming a 3D path
    edges = np.array([[0, 1], [1, 2], [2, 3]], dtype=index_dtype)
    points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 1]], dtype=real_dtype)
    return tf.EdgeMesh(edges, points)


def construct_ray_hitting_edge_mesh(edge_mesh, expected_t=1.0, interpolation_t=0.5, seed=None):
    """
    Construct a ray that will hit the edge mesh at a known point with known t value.

    Parameters
    ----------
    edge_mesh : EdgeMesh
        The edge mesh to construct a ray for
    expected_t : float
        The expected t value where the ray will hit (default 1.0)
    interpolation_t : float
        Parameter in [0, 1] for interpolating along the edge (0.5 = midpoint)
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    ray : Ray
        Ray that will hit the edge mesh at t=expected_t
    expected_edge_idx : int
        Index of the edge that should be hit
    expected_hit_point : ndarray
        Expected hit point on the edge
    """
    if seed is not None:
        np.random.seed(seed)

    # Pick a random edge
    edge_idx = np.random.choice(edge_mesh.edges.shape[0])
    pt0, pt1 = edge_mesh.points[edge_mesh.edges[edge_idx]]

    # Interpolate to get a point on the edge
    hit_point = pt0 + interpolation_t * (pt1 - pt0)

    # Generate a random direction (normalized)
    if edge_mesh.dims == 2:
        # For 2D, pick a random angle
        angle = np.random.uniform(0, 2 * np.pi)
        direction = np.array([np.cos(angle), np.sin(angle)], dtype=edge_mesh.dtype)
    else:  # 3D
        # For 3D, pick random direction on unit sphere
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.random.uniform(0, np.pi)
        direction = np.array([
            np.sin(phi) * np.cos(theta),
            np.sin(phi) * np.sin(theta),
            np.cos(phi)
        ], dtype=edge_mesh.dtype)

    # Create ray origin offset from hit point
    # origin = hit_point + direction * expected_t
    # Then: hit_point = origin + (-direction) * expected_t
    origin = hit_point + direction * expected_t
    ray_direction = -direction

    ray = tf.Ray(origin=origin, direction=ray_direction)

    return ray, edge_idx, hit_point


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
@pytest.mark.parametrize("dims", [2, 3])
def test_edge_mesh_ray_cast_hit(index_dtype, real_dtype, dims):
    """Test ray casting that should hit the edge mesh with known geometry"""
    if dims == 2:
        edge_mesh = create_2d_edge_mesh(index_dtype, real_dtype)
    else:  # dims == 3
        edge_mesh = create_3d_edge_mesh(index_dtype, real_dtype)

    # Construct ray with known hit point and t value
    ray, expected_edge_idx, expected_hit_point = construct_ray_hitting_edge_mesh(
        edge_mesh, expected_t=1.0, interpolation_t=0.5, seed=42
    )

    result = tf.ray_cast(ray, edge_mesh)

    # Should hit the edge mesh
    assert result is not None, "Ray should intersect edge mesh"
    assert isinstance(result, tuple), "Result should be a tuple"
    assert len(result) == 2, "Result should be (edge_idx, t)"

    edge_idx, t = result
    assert isinstance(edge_idx, (int, np.integer)), "Edge index should be an integer"
    assert edge_idx == expected_edge_idx, f"Should hit edge {expected_edge_idx}, got {edge_idx}"
    assert isinstance(t, (float, np.floating)), "t should be a float"
    assert np.isclose(t, 1.0, atol=1e-5), f"Expected t=1.0, got {t}"

    # Verify the hit point
    hit_point = ray.origin + t * ray.direction
    assert np.allclose(hit_point, expected_hit_point, atol=1e-4), \
        f"Hit point mismatch: expected {expected_hit_point}, got {hit_point}"


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
@pytest.mark.parametrize("dims", [2, 3])
def test_edge_mesh_ray_cast_miss(index_dtype, real_dtype, dims):
    """Test ray casting that should miss the edge mesh"""
    if dims == 2:
        edge_mesh = create_2d_edge_mesh(index_dtype, real_dtype)
        # Ray from above but offset (will miss the segment)
        ray = tf.Ray(
            origin=np.array([5.0, 3.0], dtype=real_dtype),
            direction=np.array([0.0, -1.0], dtype=real_dtype)
        )
    else:  # dims == 3
        edge_mesh = create_3d_edge_mesh(index_dtype, real_dtype)
        # Ray from the side but offset (will miss)
        ray = tf.Ray(
            origin=np.array([5.0, 3.0, 0.0], dtype=real_dtype),
            direction=np.array([0.0, -1.0, 0.0], dtype=real_dtype)
        )

    result = tf.ray_cast(ray, edge_mesh)

    # Should miss the edge mesh
    assert result is None, "Ray should not intersect edge mesh"


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
@pytest.mark.parametrize("dims", [2, 3])
def test_edge_mesh_ray_cast_pointing_away(index_dtype, real_dtype, dims):
    """Test ray pointing away from edge mesh"""
    if dims == 2:
        edge_mesh = create_2d_edge_mesh(index_dtype, real_dtype)
        # Ray from above but pointing up (away)
        ray = tf.Ray(
            origin=np.array([1.0, 3.0], dtype=real_dtype),
            direction=np.array([0.0, 1.0], dtype=real_dtype)
        )
    else:  # dims == 3
        edge_mesh = create_3d_edge_mesh(index_dtype, real_dtype)
        # Ray from the side but pointing away
        ray = tf.Ray(
            origin=np.array([1.0, 3.0, 0.0], dtype=real_dtype),
            direction=np.array([0.0, 1.0, 0.0], dtype=real_dtype)
        )

    result = tf.ray_cast(ray, edge_mesh)

    # Should not hit (pointing away)
    assert result is None, "Ray pointing away should not intersect edge mesh"


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
@pytest.mark.parametrize("dims", [2, 3])
def test_edge_mesh_ray_cast_multiple_t_values(index_dtype, real_dtype, dims):
    """Test ray casting with different expected t values"""
    if dims == 2:
        edge_mesh = create_2d_edge_mesh(index_dtype, real_dtype)
    else:
        edge_mesh = create_3d_edge_mesh(index_dtype, real_dtype)

    # Test different t values
    # Use different seeds to ensure each scenario has independent geometry
    test_cases = [(0.5, 42), (1.0, 43), (2.0, 44), (5.0, 45)]

    for expected_t, seed in test_cases:
        ray, expected_edge_idx, expected_hit_point = construct_ray_hitting_edge_mesh(
            edge_mesh, expected_t=expected_t, interpolation_t=0.5, seed=seed
        )

        result = tf.ray_cast(ray, edge_mesh)
        assert result is not None, f"Ray should hit edge mesh at t={expected_t}"

        edge_idx, t = result
        assert edge_idx == expected_edge_idx, f"Should hit edge {expected_edge_idx}, got {edge_idx}"
        assert np.isclose(t, expected_t, atol=1e-5), f"Expected t={expected_t}, got {t}"

        # Verify hit point
        hit_point = ray.origin + t * ray.direction
        assert np.allclose(hit_point, expected_hit_point, atol=1e-4), \
            f"Hit point mismatch at t={expected_t}"


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
@pytest.mark.parametrize("dims", [2, 3])
def test_edge_mesh_ray_cast_edge_interpolation(index_dtype, real_dtype, dims):
    """Test ray casting at different positions along edges"""
    if dims == 2:
        edge_mesh = create_2d_edge_mesh(index_dtype, real_dtype)
    else:
        edge_mesh = create_3d_edge_mesh(index_dtype, real_dtype)

    # Test hitting at different positions along the edge
    # Note: Use interior points (0.25, 0.5, 0.75) for strict edge index checking
    # Endpoints (0.0, 1.0) hit shared vertices and can return adjacent edges
    # Use different seeds to ensure independent geometry for each test
    test_cases = [(0.25, 123), (0.5, 124), (0.75, 125)]

    for interpolation_t, seed in test_cases:
        ray, expected_edge_idx, expected_hit_point = construct_ray_hitting_edge_mesh(
            edge_mesh, expected_t=1.0, interpolation_t=interpolation_t, seed=seed
        )

        result = tf.ray_cast(ray, edge_mesh)
        assert result is not None, f"Ray should hit at interpolation_t={interpolation_t}"

        edge_idx, t = result
        assert edge_idx == expected_edge_idx, \
            f"Should hit edge {expected_edge_idx}, got {edge_idx} at interpolation_t={interpolation_t}"
        assert np.isclose(t, 1.0, atol=1e-5), f"Expected t=1.0, got {t}"

        # Verify hit point
        hit_point = ray.origin + t * ray.direction
        assert np.allclose(hit_point, expected_hit_point, atol=1e-4), \
            f"Hit point mismatch at interpolation_t={interpolation_t}"


@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_edge_mesh_ray_cast_with_transformation_2d(real_dtype):
    """Test ray casting with 2D transformation"""
    # Create segment at origin: [0, 0] to [1, 0]
    edges = np.array([[0, 1]], dtype=np.int32)
    points = np.array([[0, 0], [1, 0]], dtype=real_dtype)
    edge_mesh = tf.EdgeMesh(edges, points)

    # First, construct a ray that hits the untransformed edge mesh
    ray_untransformed, expected_edge_idx, expected_hit_point = construct_ray_hitting_edge_mesh(
        edge_mesh, expected_t=1.5, interpolation_t=0.5, seed=200
    )

    # Verify it hits untransformed
    result_untransformed = tf.ray_cast(ray_untransformed, edge_mesh)
    assert result_untransformed is not None
    edge_idx_orig, t_orig = result_untransformed
    assert edge_idx_orig == expected_edge_idx
    assert np.isclose(t_orig, 1.5, atol=1e-5)

    # Now apply transformation [translation by (5, 3)]
    # Format: [r11 r12 tx]
    #         [r21 r22 ty]
    #         [  0   0  1]
    transformation = np.array([
        [1, 0, 5],
        [0, 1, 3],
        [0, 0, 1]
    ], dtype=real_dtype)

    edge_mesh.transformation = transformation

    # Transform the ray: apply same transformation to origin, direction stays same
    origin_homogeneous = np.array([ray_untransformed.origin[0], ray_untransformed.origin[1], 1.0], dtype=real_dtype)
    transformed_origin = (transformation @ origin_homogeneous)[:2]

    # Direction is only affected by rotation/scale part (upper-left 2x2)
    transformed_direction = (transformation[:2, :2] @ ray_untransformed.direction)

    ray_transformed = tf.Ray(
        origin=transformed_origin,
        direction=transformed_direction
    )

    # Cast transformed ray against transformed edge mesh
    result = tf.ray_cast(ray_transformed, edge_mesh)
    assert result is not None, "Ray should hit transformed edge mesh"

    edge_idx, t = result
    assert edge_idx == expected_edge_idx, f"Should hit edge {expected_edge_idx}, got {edge_idx}"
    # t value should be the same (distance is preserved under rigid transformations)
    assert np.isclose(t, 1.5, atol=1e-4), f"Expected t=1.5, got {t}"


@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_edge_mesh_ray_cast_with_transformation_3d(real_dtype):
    """Test ray casting with 3D transformation"""
    # Create segment along x-axis: [0, 0, 0] to [1, 0, 0]
    edges = np.array([[0, 1]], dtype=np.int32)
    points = np.array([[0, 0, 0], [1, 0, 0]], dtype=real_dtype)
    edge_mesh = tf.EdgeMesh(edges, points)

    # First, construct a ray that hits the untransformed edge mesh
    ray_untransformed, expected_edge_idx, expected_hit_point = construct_ray_hitting_edge_mesh(
        edge_mesh, expected_t=2.0, interpolation_t=0.5, seed=300
    )

    # Verify it hits untransformed
    result_untransformed = tf.ray_cast(ray_untransformed, edge_mesh)
    assert result_untransformed is not None
    edge_idx_orig, t_orig = result_untransformed
    assert edge_idx_orig == expected_edge_idx
    assert np.isclose(t_orig, 2.0, atol=1e-5)

    # Now apply transformation [translation by (10, 5, 2)]
    # Format: [r11 r12 r13 tx]
    #         [r21 r22 r23 ty]
    #         [r31 r32 r33 tz]
    #         [  0   0   0  1]
    transformation = np.array([
        [1, 0, 0, 10],
        [0, 1, 0,  5],
        [0, 0, 1,  2],
        [0, 0, 0,  1]
    ], dtype=real_dtype)

    edge_mesh.transformation = transformation

    # Transform the ray: apply same transformation to origin, direction stays same
    origin_homogeneous = np.array([
        ray_untransformed.origin[0],
        ray_untransformed.origin[1],
        ray_untransformed.origin[2],
        1.0
    ], dtype=real_dtype)
    transformed_origin = (transformation @ origin_homogeneous)[:3]

    # Direction is only affected by rotation/scale part (upper-left 3x3)
    transformed_direction = (transformation[:3, :3] @ ray_untransformed.direction)

    ray_transformed = tf.Ray(
        origin=transformed_origin,
        direction=transformed_direction
    )

    # Cast transformed ray against transformed edge mesh
    result = tf.ray_cast(ray_transformed, edge_mesh)
    assert result is not None, "Ray should hit transformed edge mesh"

    edge_idx, t = result
    assert edge_idx == expected_edge_idx, f"Should hit edge {expected_edge_idx}, got {edge_idx}"
    # t value should be the same (distance is preserved under rigid transformations)
    assert np.isclose(t, 2.0, atol=1e-4), f"Expected t=2.0, got {t}"


@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_edge_mesh_ray_cast_transformation_clear(real_dtype):
    """Test clearing transformation"""
    edges = np.array([[0, 1]], dtype=np.int32)
    points = np.array([[0, 0, 0], [1, 0, 0]], dtype=real_dtype)
    edge_mesh = tf.EdgeMesh(edges, points)

    # Construct a ray that hits the untransformed edge mesh
    ray_untransformed, expected_edge_idx, _ = construct_ray_hitting_edge_mesh(
        edge_mesh, expected_t=1.8, interpolation_t=0.5, seed=400
    )

    # Verify it hits untransformed
    result = tf.ray_cast(ray_untransformed, edge_mesh)
    assert result is not None
    edge_idx, t = result
    assert edge_idx == expected_edge_idx
    assert np.isclose(t, 1.8, atol=1e-5)

    # Apply transformation
    transformation = np.array([
        [1, 0, 0, 10],
        [0, 1, 0,  5],
        [0, 0, 1,  2],
        [0, 0, 0,  1]
    ], dtype=real_dtype)
    edge_mesh.transformation = transformation

    # Original ray should miss the transformed edge mesh
    result = tf.ray_cast(ray_untransformed, edge_mesh)
    assert result is None, "Untransformed ray should miss transformed edge mesh"

    # Clear transformation
    edge_mesh.transformation = None

    # Now original ray should hit again
    result = tf.ray_cast(ray_untransformed, edge_mesh)
    assert result is not None, "Ray should hit after clearing transformation"

    edge_idx, t = result
    assert edge_idx == expected_edge_idx
    assert np.isclose(t, 1.8, atol=1e-5)


def test_edge_mesh_ray_cast_dimension_mismatch():
    """Test that dimension mismatch raises an error"""
    edges = np.array([[0, 1]], dtype=np.int32)
    points = np.array([[0, 0], [1, 0]], dtype=np.float32)
    edge_mesh = tf.EdgeMesh(edges, points)  # 2D edge mesh

    # 3D ray
    ray_3d = tf.Ray(
        origin=np.array([0.5, 0.5, 2.0], dtype=np.float32),
        direction=np.array([0.0, 0.0, -1.0], dtype=np.float32)
    )

    with pytest.raises(ValueError, match="Dimension mismatch"):
        tf.ray_cast(ray_3d, edge_mesh)


def test_edge_mesh_ray_cast_dtype_mismatch():
    """Test that dtype mismatch raises an error"""
    edges = np.array([[0, 1]], dtype=np.int32)
    points = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
    edge_mesh = tf.EdgeMesh(edges, points)  # float32 edge mesh

    # float64 ray
    ray_f64 = tf.Ray(
        origin=np.array([0.5, 2.0, 0.0], dtype=np.float64),
        direction=np.array([0.0, -1.0, 0.0], dtype=np.float64)
    )

    with pytest.raises(TypeError, match="Dtype mismatch"):
        tf.ray_cast(ray_f64, edge_mesh)


def test_edge_mesh_ray_cast_multi_edge():
    """Test ray casting on edge mesh with multiple edges using known geometry"""
    # Create a mesh with 3 edges forming a path
    edges = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int32)
    points = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [2, 1, 0]], dtype=np.float32
    )
    edge_mesh = tf.EdgeMesh(edges, points)

    # Use helper to construct ray with known hit
    ray, expected_edge_idx, expected_hit_point = construct_ray_hitting_edge_mesh(
        edge_mesh, expected_t=1.5, interpolation_t=0.3, seed=999
    )

    result = tf.ray_cast(ray, edge_mesh)
    assert result is not None, "Ray should hit edge mesh"

    edge_idx, t = result
    assert edge_idx == expected_edge_idx, f"Should hit edge {expected_edge_idx}, got {edge_idx}"
    assert np.isclose(t, 1.5, atol=1e-5), f"Expected t=1.5, got {t}"

    # Verify hit point
    hit_point = ray.origin + t * ray.direction
    assert np.allclose(hit_point, expected_hit_point, atol=1e-4)


def test_edge_mesh_ray_cast_endpoint_hit():
    """Test ray hitting exactly at an edge endpoint (isolated edge, no ambiguity)"""
    # Single isolated segment from [0, 0] to [1, 0]
    edges = np.array([[0, 1]], dtype=np.int32)
    points = np.array([[0, 0], [1, 0]], dtype=np.float32)
    edge_mesh = tf.EdgeMesh(edges, points)

    # Use helper to hit at edge start (interpolation_t=0.0)
    # Since there's only one edge, no ambiguity about which edge is hit
    ray, expected_edge_idx, expected_hit_point = construct_ray_hitting_edge_mesh(
        edge_mesh, expected_t=2.0, interpolation_t=0.0, seed=111
    )

    result = tf.ray_cast(ray, edge_mesh)
    assert result is not None, "Ray should hit edge at endpoint"

    edge_idx, t = result
    assert edge_idx == expected_edge_idx, f"Should hit edge {expected_edge_idx}, got {edge_idx}"
    assert np.isclose(t, 2.0, atol=1e-5), f"Expected t=2.0, got {t}"

    # Verify we hit the expected point
    hit_point = ray.origin + t * ray.direction
    assert np.allclose(hit_point, expected_hit_point, atol=1e-4)


if __name__ == "__main__":
    # Run tests with verbose output
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
