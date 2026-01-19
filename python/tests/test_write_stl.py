"""
Test write_stl functionality

Copyright (c) 2025 Žiga Sajovic, XLAB
"""

import sys
import os
import pytest
import numpy as np
import trueform as tf
import tempfile

# Parametrize index dtypes
INDEX_DTYPES = [np.int32, np.int64]


def canonicalize_mesh(faces, points):
    """
    Create canonical ordering of mesh for comparison.

    1. For each face, rotate indices so it starts with the lexicographically smallest point
    2. Sort faces lexicographically by their point coordinates

    Returns canonical (faces, points) suitable for comparison.
    """
    canonical_faces = []

    for face in faces:
        # Get the three points of this face
        face_points = points[face]

        # Find the index (0, 1, or 2) of the lexicographically smallest point
        # Compare as tuples for lexicographic ordering
        point_tuples = [tuple(p) for p in face_points]
        min_idx = min(range(3), key=lambda i: point_tuples[i])

        # Rotate face so it starts with the smallest point
        rotated_face = np.roll(face, -min_idx)
        canonical_faces.append(rotated_face)

    canonical_faces = np.array(canonical_faces)

    # Now sort faces by their point coordinates
    # Create sort key: concatenate all three points of each face
    sort_keys = []
    for face in canonical_faces:
        face_points = points[face]
        # Flatten to single tuple for sorting
        sort_key = tuple(face_points.flatten())
        sort_keys.append(sort_key)

    # Get sort order
    sort_order = sorted(range(len(sort_keys)), key=lambda i: sort_keys[i])

    # Apply sort order
    canonical_faces = canonical_faces[sort_order]

    return canonical_faces, points


def assert_meshes_equal(faces1, points1, faces2, points2, atol=1e-5):
    """
    Assert that two meshes are geometrically equivalent.

    Handles the fact that STL is unordered and vertices may be reindexed.
    Compares meshes by canonicalizing them first.
    """
    # Check shapes match
    assert faces1.shape == faces2.shape, \
        f"Face count mismatch: {faces1.shape[0]} vs {faces2.shape[0]}"
    assert points1.shape == points2.shape, \
        f"Point count mismatch: {points1.shape[0]} vs {points2.shape[0]}"

    # Canonicalize both meshes
    faces1_canon, points1_canon = canonicalize_mesh(faces1, points1)
    faces2_canon, points2_canon = canonicalize_mesh(faces2, points2)

    # Compare canonicalized faces by their point coordinates
    for i, (face1, face2) in enumerate(zip(faces1_canon, faces2_canon)):
        points1_face = points1_canon[face1]
        points2_face = points2_canon[face2]

        assert np.allclose(points1_face, points2_face, atol=atol), \
            f"Face {i} points mismatch:\n" \
            f"  Mesh 1: {points1_face}\n" \
            f"  Mesh 2: {points2_face}"


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_write_stl_simple_triangle(index_dtype):
    """Test writing a simple triangle mesh"""
    with tempfile.TemporaryDirectory() as tmpdir:
        stl_file = os.path.join(tmpdir, "triangle.stl")

        # Create a simple triangle
        faces = np.array([[0, 1, 2]], dtype=index_dtype)
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)

        # Write STL
        success = tf.write_stl((faces, points), stl_file)
        assert success, "write_stl should return True on success"

        # Verify file was created
        assert os.path.exists(stl_file), "STL file should be created"
        assert os.path.getsize(stl_file) > 0, "STL file should not be empty"


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_write_stl_round_trip(index_dtype):
    """Test writing and reading back produces same data (after canonical ordering)"""
    with tempfile.TemporaryDirectory() as tmpdir:
        stl_file = os.path.join(tmpdir, "round_trip.stl")

        # Create a cube (8 points, 12 triangular faces)
        points_orig = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
        ], dtype=np.float32)

        faces_orig = np.array([
            [0, 1, 2], [0, 2, 3],  # bottom
            [4, 6, 5], [4, 7, 6],  # top
            [0, 5, 1], [0, 4, 5],  # front
            [2, 7, 3], [2, 6, 7],  # back
            [0, 3, 7], [0, 7, 4],  # left
            [1, 6, 2], [1, 5, 6],  # right
        ], dtype=index_dtype)

        # Write STL
        success = tf.write_stl((faces_orig, points_orig), stl_file)
        assert success, "write_stl should succeed"

        # Read it back
        faces_read, points_read = tf.read_stl(stl_file, index_dtype=index_dtype)

        # Check shapes
        assert faces_read.shape == faces_orig.shape, "Faces shape should match"
        assert points_read.shape == points_orig.shape, "Points shape should match"

        # Check dtypes
        assert faces_read.dtype == index_dtype, f"Faces dtype should be {index_dtype}"
        assert points_read.dtype == np.float32, "Points dtype should be float32"

        # Compare meshes geometrically (handles reordering/reindexing)
        assert_meshes_equal(faces_orig, points_orig, faces_read, points_read)


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_write_stl_with_transformation(index_dtype):
    """Test writing with a transformation matrix"""
    with tempfile.TemporaryDirectory() as tmpdir:
        stl_file = os.path.join(tmpdir, "transformed.stl")

        # Create a simple triangle at origin
        faces = np.array([[0, 1, 2]], dtype=index_dtype)
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)

        # Create transformation: translate by (5, 0, 10)
        transformation = np.eye(4, dtype=np.float32)
        transformation[0, 3] = 5.0  # x translation
        transformation[2, 3] = 10.0  # z translation

        # Write with transformation
        success = tf.write_stl((faces, points), stl_file, transformation=transformation)
        assert success, "write_stl with transformation should succeed"

        # Read back
        faces_read, points_read = tf.read_stl(stl_file, index_dtype=index_dtype)

        # Expected transformed mesh
        expected_faces = np.array([[0, 1, 2]], dtype=index_dtype)
        expected_points = np.array([
            [5, 0, 10], [6, 0, 10], [5, 1, 10]
        ], dtype=np.float32)

        # Compare meshes geometrically
        assert_meshes_equal(expected_faces, expected_points, faces_read, points_read)


def test_write_stl_filename_extension():
    """Test that .stl extension is appended if not present"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write without .stl extension
        filename_base = os.path.join(tmpdir, "test")

        faces = np.array([[0, 1, 2]], dtype=np.int32)
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)

        success = tf.write_stl((faces, points), filename_base)
        assert success, "write_stl should succeed"

        # Check that file with .stl extension was created
        expected_file = filename_base + ".stl"
        assert os.path.exists(expected_file), \
            ".stl extension should be appended automatically"


def test_write_stl_invalid_faces_shape():
    """Test that invalid faces shape raises error"""
    with tempfile.TemporaryDirectory() as tmpdir:
        stl_file = os.path.join(tmpdir, "test.stl")

        # Invalid faces: shape (N, 4) instead of (N, 3)
        faces = np.array([[0, 1, 2, 3]], dtype=np.int32)
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]],
                         dtype=np.float32)

        with pytest.raises(ValueError, match="faces must have shape"):
            tf.write_stl((faces, points), stl_file)


def test_write_stl_invalid_faces_dtype():
    """Test that invalid faces dtype raises error"""
    with tempfile.TemporaryDirectory() as tmpdir:
        stl_file = os.path.join(tmpdir, "test.stl")

        # Invalid dtype: float instead of int
        faces = np.array([[0, 1, 2]], dtype=np.float32)
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)

        with pytest.raises(ValueError, match="faces dtype must be"):
            tf.write_stl((faces, points), stl_file)


def test_write_stl_invalid_points_shape():
    """Test that invalid points shape raises error"""
    with tempfile.TemporaryDirectory() as tmpdir:
        stl_file = os.path.join(tmpdir, "test.stl")

        # Invalid points: shape (N, 2) instead of (N, 3)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        points = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32)

        with pytest.raises(ValueError, match="points must have shape"):
            tf.write_stl((faces, points), stl_file)


def test_write_stl_invalid_points_dtype():
    """Test that invalid points dtype raises error"""
    with tempfile.TemporaryDirectory() as tmpdir:
        stl_file = os.path.join(tmpdir, "test.stl")

        # Invalid dtype: float64 instead of float32
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)

        with pytest.raises(ValueError, match="points dtype must be"):
            tf.write_stl((faces, points), stl_file)


def test_write_stl_invalid_transformation_shape():
    """Test that invalid transformation shape raises error"""
    with tempfile.TemporaryDirectory() as tmpdir:
        stl_file = os.path.join(tmpdir, "test.stl")

        faces = np.array([[0, 1, 2]], dtype=np.int32)
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)

        # Invalid transformation: shape (3, 3) instead of (4, 4)
        transformation = np.eye(3, dtype=np.float32)

        with pytest.raises(ValueError, match="transformation must have shape"):
            tf.write_stl((faces, points), stl_file, transformation=transformation)


def test_write_stl_invalid_transformation_dtype():
    """Test that invalid transformation dtype raises error"""
    with tempfile.TemporaryDirectory() as tmpdir:
        stl_file = os.path.join(tmpdir, "test.stl")

        faces = np.array([[0, 1, 2]], dtype=np.int32)
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)

        # Invalid dtype: float64 instead of float32
        transformation = np.eye(4, dtype=np.float64)

        with pytest.raises(ValueError, match="transformation dtype must be"):
            tf.write_stl((faces, points), stl_file, transformation=transformation)


def test_write_stl_non_contiguous_arrays():
    """Test that non-contiguous arrays are handled correctly"""
    with tempfile.TemporaryDirectory() as tmpdir:
        stl_file = os.path.join(tmpdir, "test.stl")

        # Create non-contiguous arrays via slicing
        faces_full = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int32)
        points_full = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]],
                              dtype=np.float32)

        # Take non-contiguous slices
        faces = faces_full[::1, :]  # This might not be contiguous depending on numpy
        points = points_full[::1, :]

        # Should work even if arrays are not contiguous
        success = tf.write_stl((faces, points), stl_file)
        assert success, "write_stl should handle non-contiguous arrays"

        # Verify by reading back
        faces_read, points_read = tf.read_stl(stl_file)
        assert faces_read.shape[0] == 2, "Should have 2 faces"
        assert points_read.shape[0] == 4, "Should have 4 points"


def test_write_stl_from_mesh_object():
    """Test writing from a Mesh object"""
    with tempfile.TemporaryDirectory() as tmpdir:
        stl_file = os.path.join(tmpdir, "mesh.stl")

        # Create a Mesh object
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        mesh = tf.Mesh(faces, points)

        # Write from mesh
        success = tf.write_stl(mesh, stl_file)
        assert success, "write_stl should succeed with Mesh object"

        # Read back and verify
        faces_read, points_read = tf.read_stl(stl_file)

        # Create expected mesh for comparison
        expected_faces = np.array([[0, 1, 2]], dtype=np.int32)
        expected_points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)

        assert_meshes_equal(expected_faces, expected_points, faces_read, points_read)


def test_write_stl_mesh_with_transformation():
    """Test writing a Mesh with transformation property set"""
    with tempfile.TemporaryDirectory() as tmpdir:
        stl_file = os.path.join(tmpdir, "mesh_transformed.stl")

        # Create a Mesh at origin
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        mesh = tf.Mesh(faces, points)

        # Set transformation on mesh
        transformation = np.eye(4, dtype=np.float32)
        transformation[0, 3] = 2.0  # Translate 2 in X
        transformation[1, 3] = 3.0  # Translate 3 in Y
        mesh.transformation = transformation

        # Write mesh (should apply transformation)
        success = tf.write_stl(mesh, stl_file)
        assert success, "write_stl should succeed with transformed Mesh"

        # Read back
        faces_read, points_read = tf.read_stl(stl_file)

        # Expected transformed mesh
        expected_faces = np.array([[0, 1, 2]], dtype=np.int32)
        expected_points = np.array([[2, 3, 0], [3, 3, 0], [2, 4, 0]], dtype=np.float32)

        assert_meshes_equal(expected_faces, expected_points, faces_read, points_read)


def test_write_stl_mesh_transformation_override():
    """Test that explicit transformation kwarg overrides mesh transformation"""
    with tempfile.TemporaryDirectory() as tmpdir:
        stl_file = os.path.join(tmpdir, "mesh_override.stl")

        # Create a Mesh with one transformation
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        mesh = tf.Mesh(faces, points)

        # Set mesh transformation
        mesh_transform = np.eye(4, dtype=np.float32)
        mesh_transform[0, 3] = 10.0  # This should be ignored
        mesh.transformation = mesh_transform

        # Override with different transformation
        override_transform = np.eye(4, dtype=np.float32)
        override_transform[2, 3] = 5.0  # Translate 5 in Z
        success = tf.write_stl(mesh, stl_file, transformation=override_transform)
        assert success, "write_stl should succeed"

        # Read back
        faces_read, points_read = tf.read_stl(stl_file)

        # Expected: override transformation applied, not mesh transformation
        expected_faces = np.array([[0, 1, 2]], dtype=np.int32)
        expected_points = np.array([[0, 0, 5], [1, 0, 5], [0, 1, 5]], dtype=np.float32)

        assert_meshes_equal(expected_faces, expected_points, faces_read, points_read)


def test_write_stl_mesh_dynamic_error():
    """Test that dynamic mesh raises error"""
    with tempfile.TemporaryDirectory() as tmpdir:
        stl_file = os.path.join(tmpdir, "dynamic.stl")

        # Create a dynamic mesh
        offsets = np.array([0, 3, 7], dtype=np.int32)
        data = np.array([0, 1, 2, 0, 2, 3, 4], dtype=np.int32)
        faces = tf.OffsetBlockedArray(offsets, data)
        points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.5, 0.5, 0]], dtype=np.float32)
        mesh = tf.Mesh(faces, points)

        # Should raise error for dynamic mesh
        with pytest.raises(ValueError, match="STL format only supports triangular meshes"):
            tf.write_stl(mesh, stl_file)


def test_write_stl_mesh_2d_error():
    """Test that 2D mesh raises error"""
    with tempfile.TemporaryDirectory() as tmpdir:
        stl_file = os.path.join(tmpdir, "2d.stl")

        # Create a 2D mesh
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        points = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32)
        mesh = tf.Mesh(faces, points)

        # Should raise error for 2D mesh
        with pytest.raises(ValueError, match="STL format only supports 3D meshes"):
            tf.write_stl(mesh, stl_file)


if __name__ == "__main__":
    # Run tests with pytest
    sys.exit(pytest.main([__file__, "-v"]))
