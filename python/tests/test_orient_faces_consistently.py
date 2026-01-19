"""
Test orient_faces_consistently

Copyright (c) 2025 Ziga Sajovic, XLAB
"""

import sys

import pytest
import numpy as np
import trueform as tf

# Parameter sets
INDEX_DTYPES = [np.int32, np.int64]
REAL_DTYPES = [np.float32, np.float64]
DIMS = [2, 3]
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


def faces_are_consistent(faces, points):
    """
    Check if all adjacent faces have consistent orientation.

    For manifold edges (shared by exactly 2 faces), the edge should appear
    in opposite directions in the two faces.
    """
    # Build edge to face map
    edge_faces = {}
    for fi, face in enumerate(faces):
        n = len(face)
        for i in range(n):
            v0, v1 = face[i], face[(i + 1) % n]
            edge = (v0, v1)
            if edge not in edge_faces:
                edge_faces[edge] = []
            edge_faces[edge].append(fi)

    # Check each edge
    for edge, face_list in edge_faces.items():
        if len(face_list) == 1:
            # Boundary edge, ok
            continue
        elif len(face_list) == 2:
            # Manifold edge - check if opposite edge exists
            opposite = (edge[1], edge[0])
            if opposite in edge_faces and len(edge_faces[opposite]) > 0:
                # Good - edges in opposite directions
                pass
            else:
                # Both faces have edge in same direction - inconsistent
                return False
        else:
            # Non-manifold edge - skip consistency check
            pass

    return True


def canonicalize_face(face):
    """Canonicalize a face for comparison (rotation and direction independent)."""
    n = len(face)
    # Find all rotations and both directions
    candidates = []
    for start in range(n):
        # Forward
        forward = tuple(face[(start + i) % n] for i in range(n))
        candidates.append(forward)
        # Reverse
        reverse = tuple(face[(start - i) % n] for i in range(n))
        candidates.append(reverse)
    return min(candidates)


# ==============================================================================
# Test data generators
# ==============================================================================

def create_two_triangles_consistent(index_dtype, real_dtype, dims):
    """
    Create two triangles with consistent orientation.

    Triangle 0: [0, 1, 2] - CCW
    Triangle 1: [1, 3, 2] - shares edge (1,2), appears as (3,2) -> (2,1) in reverse

    For consistent orientation with edge (1,2) in face 0,
    face 1 should have edge (2,1) which means vertices [1,3,2] -> edge (2,1) exists.
    """
    faces = np.array([
        [0, 1, 2],
        [1, 3, 2]
    ], dtype=index_dtype)

    if dims == 2:
        points = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, 1.0],
            [1.5, 1.0]
        ], dtype=real_dtype)
    else:
        points = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [1.5, 1.0, 0.0]
        ], dtype=real_dtype)

    return faces, points


def create_two_triangles_inconsistent(index_dtype, real_dtype, dims):
    """
    Create two triangles with inconsistent orientation.

    Triangle 0: [0, 1, 2] - CCW
    Triangle 1: [1, 2, 3] - shares edge (1,2), but has it in same direction -> inconsistent
    """
    faces = np.array([
        [0, 1, 2],
        [1, 2, 3]  # Flipped - edge (1,2) in same direction as face 0
    ], dtype=index_dtype)

    if dims == 2:
        points = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, 1.0],
            [1.5, 1.0]
        ], dtype=real_dtype)
    else:
        points = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [1.5, 1.0, 0.0]
        ], dtype=real_dtype)

    return faces, points


def create_four_triangles_one_flipped(index_dtype, real_dtype, dims):
    r"""
    Create four triangles forming a larger shape, with one flipped.

    Layout (quad split into 4 triangles around center point 4)::

        3 --- 2
        |\ 2 /|
        | \ / |
        |3 4 1|
        | / \ |
        |/ 0 \|
        0 --- 1

    Faces (CCW): [0,1,4], [1,2,4], [2,3,4], [3,0,4]
    Flip face 2: [2,3,4] -> [4,3,2]

    Since 3 faces are correct and 1 is flipped, the result should
    have consistent orientation matching the majority.
    """
    faces = np.array([
        [0, 1, 4],  # CCW
        [1, 2, 4],  # CCW
        [4, 3, 2],  # Flipped (should be [2, 3, 4])
        [3, 0, 4],  # CCW
    ], dtype=index_dtype)

    if dims == 2:
        points = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.5, 0.5]  # Center
        ], dtype=real_dtype)
    else:
        points = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, 0.0]  # Center
        ], dtype=real_dtype)

    return faces, points


def create_single_triangle(index_dtype, real_dtype, dims):
    """Create a single triangle (trivially consistent)."""
    faces = np.array([
        [0, 1, 2]
    ], dtype=index_dtype)

    if dims == 2:
        points = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, 1.0]
        ], dtype=real_dtype)
    else:
        points = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0]
        ], dtype=real_dtype)

    return faces, points


def create_two_separate_triangles(index_dtype, real_dtype, dims):
    """
    Create two triangles that don't share any edge (separate components).
    Each should be oriented independently.
    """
    faces = np.array([
        [0, 1, 2],
        [3, 4, 5]
    ], dtype=index_dtype)

    if dims == 2:
        points = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, 1.0],
            [3.0, 0.0],
            [4.0, 0.0],
            [3.5, 1.0]
        ], dtype=real_dtype)
    else:
        points = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [3.5, 1.0, 0.0]
        ], dtype=real_dtype)

    return faces, points




# ==============================================================================
# orient_faces_consistently Tests
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
@pytest.mark.parametrize("dims", DIMS)
def test_already_consistent(index_dtype, real_dtype, dims):
    """Test that already consistent faces remain unchanged."""
    faces, points = create_two_triangles_consistent(index_dtype, real_dtype, dims)

    print("faces")
    print(faces)
    print("points")
    print(points)
    result = tf.orient_faces_consistently((faces, points))
    print("result")
    print(result)

    # Check return type and shape
    assert isinstance(result, np.ndarray)
    assert result.shape == faces.shape
    assert result.dtype == index_dtype

    # Check consistency
    assert faces_are_consistent(result, points)

    # Should be identical to input (already consistent)
    np.testing.assert_array_equal(result, faces)


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
@pytest.mark.parametrize("dims", DIMS)
def test_inconsistent_two_triangles(index_dtype, real_dtype, dims):
    """Test that inconsistent faces are corrected."""
    faces, points = create_two_triangles_inconsistent(index_dtype, real_dtype, dims)

    result = tf.orient_faces_consistently((faces, points))

    # Check return type
    assert result.shape == faces.shape
    assert result.dtype == index_dtype

    # Must be consistent after orientation
    assert faces_are_consistent(result, points)


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
@pytest.mark.parametrize("dims", DIMS)
def test_four_triangles_one_flipped(index_dtype, real_dtype, dims):
    """Test that one flipped face among four is corrected."""
    faces, points = create_four_triangles_one_flipped(index_dtype, real_dtype, dims)

    result = tf.orient_faces_consistently((faces, points))

    # Check consistency
    assert faces_are_consistent(result, points)

    # Check dtype preserved
    assert result.dtype == index_dtype


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
@pytest.mark.parametrize("dims", DIMS)
def test_single_triangle(index_dtype, real_dtype, dims):
    """Test single triangle (trivially consistent)."""
    faces, points = create_single_triangle(index_dtype, real_dtype, dims)

    result = tf.orient_faces_consistently((faces, points))

    # Single face is always consistent
    assert result.shape == faces.shape
    assert faces_are_consistent(result, points)

    # Should be unchanged
    np.testing.assert_array_equal(result, faces)


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
@pytest.mark.parametrize("dims", DIMS)
def test_separate_components(index_dtype, real_dtype, dims):
    """Test that separate components are handled independently."""
    faces, points = create_two_separate_triangles(index_dtype, real_dtype, dims)

    result = tf.orient_faces_consistently((faces, points))

    # Each component is a single triangle, so trivially consistent
    assert result.shape == faces.shape
    assert faces_are_consistent(result, points)


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
@pytest.mark.parametrize("dims", DIMS)
def test_dynamic_mesh_consistent(index_dtype, real_dtype, dims):
    """Test dynamic mesh already consistent."""
    faces_array, points = create_two_triangles_consistent(index_dtype, real_dtype, dims)
    faces = make_dynamic_faces(faces_array)
    mesh = tf.Mesh(faces, points)

    result = tf.orient_faces_consistently(mesh)

    # Result should be OffsetBlockedArray
    assert isinstance(result, tf.OffsetBlockedArray)
    # Convert to array for consistency check
    result_array = np.array([result.data[result.offsets[i]:result.offsets[i+1]] for i in range(len(result))])
    assert faces_are_consistent(result_array, points)


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
@pytest.mark.parametrize("dims", DIMS)
def test_dynamic_mesh_inconsistent(index_dtype, real_dtype, dims):
    """Test dynamic mesh with inconsistent orientation."""
    faces_array, points = create_two_triangles_inconsistent(index_dtype, real_dtype, dims)
    faces = make_dynamic_faces(faces_array)
    mesh = tf.Mesh(faces, points)

    result = tf.orient_faces_consistently(mesh)

    # Result should be OffsetBlockedArray
    assert isinstance(result, tf.OffsetBlockedArray)
    # Convert to array for consistency check
    result_array = np.array([result.data[result.offsets[i]:result.offsets[i+1]] for i in range(len(result))])
    assert faces_are_consistent(result_array, points)


# ==============================================================================
# Input type tests
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_with_mesh_input(index_dtype, real_dtype):
    """Test with Mesh input instead of tuple."""
    faces, points = create_two_triangles_inconsistent(index_dtype, real_dtype, 3)
    mesh = tf.Mesh(faces, points)

    result = tf.orient_faces_consistently(mesh)

    assert result.shape == faces.shape
    assert result.dtype == index_dtype
    assert faces_are_consistent(result, points)


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_with_prebuilt_manifold_edge_link(index_dtype, real_dtype):
    """Test with Mesh that has prebuilt manifold_edge_link."""
    faces, points = create_two_triangles_inconsistent(index_dtype, real_dtype, 3)
    mesh = tf.Mesh(faces, points)

    # Force build manifold_edge_link
    _ = mesh.manifold_edge_link

    result = tf.orient_faces_consistently(mesh)

    assert faces_are_consistent(result, points)


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_does_not_modify_input(index_dtype, real_dtype):
    """Test that input is not modified."""
    faces, points = create_two_triangles_inconsistent(index_dtype, real_dtype, 3)
    faces_original = faces.copy()

    result = tf.orient_faces_consistently((faces, points))

    # Input should be unchanged
    np.testing.assert_array_equal(faces, faces_original)

    # Result should be different (one face flipped)
    assert not np.array_equal(result, faces_original)


# ==============================================================================
# Error validation tests
# ==============================================================================

def test_invalid_input_type():
    """Test with invalid input type."""
    with pytest.raises(TypeError, match="Expected tuple.*or Mesh"):
        tf.orient_faces_consistently("not valid")


def test_invalid_tuple_length():
    """Test with tuple of wrong length."""
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    with pytest.raises(ValueError, match="2 elements"):
        tf.orient_faces_consistently((faces,))


def test_invalid_ngon():
    """Test with invalid ngon (not 3 and not dynamic)."""
    # Mesh with 4 vertices per face should fail since we don't support quads anymore
    faces = np.array([[0, 1, 2, 3]], dtype=np.int32)
    points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32)

    with pytest.raises((ValueError, TypeError)):  # Mesh constructor or orient check
        tf.orient_faces_consistently((faces, points))


# ==============================================================================
# Return type tests
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_preserves_index_dtype(index_dtype):
    """Test that index dtype is preserved."""
    faces = np.array([[0, 1, 2], [1, 2, 3]], dtype=index_dtype)
    points = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0], [1.5, 1, 0]], dtype=np.float32)

    result = tf.orient_faces_consistently((faces, points))
    assert result.dtype == index_dtype


@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_works_with_real_dtype(real_dtype):
    """Test that different real dtypes work."""
    faces = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int32)
    points = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0], [1.5, 1, 0]], dtype=real_dtype)

    result = tf.orient_faces_consistently((faces, points))
    assert result.shape == faces.shape


# ==============================================================================
# Main runner
# ==============================================================================

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

