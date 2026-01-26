"""
Test ensure_positive_orientation

Copyright (c) 2025 Ziga Sajovic, XLAB
"""

import sys

import pytest
import numpy as np
import trueform as tf

# Parameter sets
INDEX_DTYPES = [np.int32, np.int64]
REAL_DTYPES = [np.float32, np.float64]
MESH_TYPES = ['triangle', 'dynamic']


# ==============================================================================
# Helper functions
# ==============================================================================

def make_dynamic_faces(faces_array):
    """Convert triangle faces array to OffsetBlockedArray for dynamic mesh."""
    index_dtype = faces_array.dtype
    offsets = np.arange(0, len(faces_array) * 3 + 1, 3, dtype=index_dtype)
    data = faces_array.ravel()
    return tf.OffsetBlockedArray(offsets, data)


def compute_signed_volume(faces, points):
    """Compute signed volume of a closed mesh."""
    # Use trueform's signed_volume if available, otherwise compute manually
    total = 0.0
    for face in faces:
        p0 = points[face[0]]
        for i in range(1, len(face) - 1):
            p1 = points[face[i]]
            p2 = points[face[i + 1]]
            # Signed volume contribution: dot(p0, cross(p1, p2)) / 6
            cross = np.cross(p1, p2)
            total += np.dot(p0, cross)
    return total / 6.0


# ==============================================================================
# Test data generators
# ==============================================================================

def create_tetrahedron_positive(index_dtype, real_dtype):
    """
    Create a tetrahedron with positive signed volume (outward normals).

    Vertices:
    - 0: origin
    - 1: x-axis
    - 2: y-axis
    - 3: z-axis

    Faces oriented with outward normals (CCW from outside).
    """
    faces = np.array([
        [0, 2, 1],  # Base (z=0), normal pointing down
        [0, 1, 3],  # Front face
        [0, 3, 2],  # Left face
        [1, 2, 3],  # Back face
    ], dtype=index_dtype)

    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=real_dtype)

    return faces, points


def create_tetrahedron_negative(index_dtype, real_dtype):
    """
    Create a tetrahedron with negative signed volume (inward normals).
    Same as positive but with reversed winding.
    """
    faces = np.array([
        [0, 1, 2],  # Base reversed
        [0, 3, 1],  # Front reversed
        [0, 2, 3],  # Left reversed
        [1, 3, 2],  # Back reversed
    ], dtype=index_dtype)

    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=real_dtype)

    return faces, points


def create_tetrahedron_inconsistent(index_dtype, real_dtype):
    """
    Create a tetrahedron with inconsistent face orientations.
    Some faces point in, some point out.
    """
    faces = np.array([
        [0, 2, 1],  # Outward
        [0, 3, 1],  # Inward (flipped)
        [0, 3, 2],  # Outward
        [1, 3, 2],  # Inward (flipped)
    ], dtype=index_dtype)

    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=real_dtype)

    return faces, points


# ==============================================================================
# ensure_positive_orientation Tests
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_already_positive(index_dtype, real_dtype):
    """Test that already positive mesh remains unchanged."""
    faces, points = create_tetrahedron_positive(index_dtype, real_dtype)

    result = tf.ensure_positive_orientation((faces, points))

    # Check return type and shape
    assert isinstance(result, np.ndarray)
    assert result.shape == faces.shape
    assert result.dtype == index_dtype

    # Volume should be positive
    vol = compute_signed_volume(result, points)
    assert vol > 0, f"Expected positive volume, got {vol}"


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_negative_becomes_positive(index_dtype, real_dtype):
    """Test that negative volume mesh becomes positive."""
    faces, points = create_tetrahedron_negative(index_dtype, real_dtype)

    # Verify it's negative before
    vol_before = compute_signed_volume(faces, points)
    assert vol_before < 0, f"Test setup error: expected negative volume, got {vol_before}"

    result = tf.ensure_positive_orientation((faces, points))

    # Volume should be positive after
    vol_after = compute_signed_volume(result, points)
    assert vol_after > 0, f"Expected positive volume after, got {vol_after}"

    # Magnitude should be same
    assert abs(abs(vol_after) - abs(vol_before)) < 1e-5


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_inconsistent_becomes_positive(index_dtype, real_dtype):
    """Test that inconsistent mesh becomes consistently positive."""
    faces, points = create_tetrahedron_inconsistent(index_dtype, real_dtype)

    result = tf.ensure_positive_orientation((faces, points))

    # Volume should be positive
    vol = compute_signed_volume(result, points)
    assert vol > 0, f"Expected positive volume, got {vol}"


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_is_consistent_flag(index_dtype, real_dtype):
    """Test is_consistent flag skips orient_faces_consistently step."""
    faces, points = create_tetrahedron_negative(index_dtype, real_dtype)

    # With is_consistent=True, it only checks volume and flips if needed
    result = tf.ensure_positive_orientation((faces, points), is_consistent=True)

    vol = compute_signed_volume(result, points)
    assert vol > 0, f"Expected positive volume, got {vol}"


# ==============================================================================
# Dynamic mesh tests
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_dynamic_mesh_positive(index_dtype, real_dtype):
    """Test dynamic mesh with positive volume."""
    faces_array, points = create_tetrahedron_positive(index_dtype, real_dtype)
    faces = make_dynamic_faces(faces_array)
    mesh = tf.Mesh(faces, points)

    result = tf.ensure_positive_orientation(mesh)

    # Result should be OffsetBlockedArray
    assert isinstance(result, tf.OffsetBlockedArray)

    # Convert to array for volume check
    result_array = np.array([result.data[result.offsets[i]:result.offsets[i+1]] for i in range(len(result))])
    vol = compute_signed_volume(result_array, points)
    assert vol > 0


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_dynamic_mesh_negative(index_dtype, real_dtype):
    """Test dynamic mesh with negative volume becomes positive."""
    faces_array, points = create_tetrahedron_negative(index_dtype, real_dtype)
    faces = make_dynamic_faces(faces_array)
    mesh = tf.Mesh(faces, points)

    result = tf.ensure_positive_orientation(mesh)

    assert isinstance(result, tf.OffsetBlockedArray)
    result_array = np.array([result.data[result.offsets[i]:result.offsets[i+1]] for i in range(len(result))])
    vol = compute_signed_volume(result_array, points)
    assert vol > 0


# ==============================================================================
# Input type tests
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_with_mesh_input(index_dtype, real_dtype):
    """Test with Mesh input instead of tuple."""
    faces, points = create_tetrahedron_negative(index_dtype, real_dtype)
    mesh = tf.Mesh(faces, points)

    result = tf.ensure_positive_orientation(mesh)

    assert result.shape == faces.shape
    assert result.dtype == index_dtype
    vol = compute_signed_volume(result, points)
    assert vol > 0


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_with_prebuilt_manifold_edge_link(index_dtype, real_dtype):
    """Test with Mesh that has prebuilt manifold_edge_link."""
    faces, points = create_tetrahedron_negative(index_dtype, real_dtype)
    mesh = tf.Mesh(faces, points)

    # Force build manifold_edge_link
    _ = mesh.manifold_edge_link

    result = tf.ensure_positive_orientation(mesh)

    vol = compute_signed_volume(result, points)
    assert vol > 0


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_does_not_modify_input(index_dtype, real_dtype):
    """Test that input is not modified."""
    faces, points = create_tetrahedron_negative(index_dtype, real_dtype)
    faces_original = faces.copy()

    _result = tf.ensure_positive_orientation((faces, points))

    # Input should be unchanged
    np.testing.assert_array_equal(faces, faces_original)


# ==============================================================================
# Error validation tests
# ==============================================================================

def test_invalid_input_type():
    """Test with invalid input type."""
    with pytest.raises(TypeError, match="Expected tuple.*or Mesh"):
        tf.ensure_positive_orientation("not valid")


def test_invalid_tuple_length():
    """Test with tuple of wrong length."""
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    with pytest.raises(ValueError, match="2 elements"):
        tf.ensure_positive_orientation((faces,))


def test_2d_mesh_rejected():
    """Test that 2D mesh is rejected."""
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    points = np.array([[0, 0], [1, 0], [0.5, 1]], dtype=np.float32)

    with pytest.raises(ValueError, match="3D"):
        tf.ensure_positive_orientation((faces, points))


def test_invalid_ngon():
    """Test with invalid ngon (not 3 and not dynamic)."""
    faces = np.array([[0, 1, 2, 3]], dtype=np.int32)
    points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32)

    with pytest.raises((ValueError, TypeError)):
        tf.ensure_positive_orientation((faces, points))


# ==============================================================================
# Return type tests
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_preserves_index_dtype(index_dtype):
    """Test that index dtype is preserved."""
    faces, points = create_tetrahedron_negative(index_dtype, np.float32)

    result = tf.ensure_positive_orientation((faces, points))
    assert result.dtype == index_dtype


@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_works_with_real_dtype(real_dtype):
    """Test that different real dtypes work."""
    faces, points = create_tetrahedron_negative(np.int32, real_dtype)

    result = tf.ensure_positive_orientation((faces, points))
    assert result.shape == faces.shape


# ==============================================================================
# Main runner
# ==============================================================================

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
