"""
Test write_obj functionality

Copyright (c) 2025 Žiga Sajovic, XLAB
"""

import sys
import os
import pytest
import numpy as np
import trueform as tf
import tempfile

# Parametrize index dtypes, point dtypes, and ngons
INDEX_DTYPES = [np.int32, np.int64]
POINT_DTYPES = [np.float32, np.float64]
NGONS = [3]


def canonicalize_mesh(faces, points, ngon):
    """
    Create canonical ordering of mesh for comparison.

    1. For each face, rotate indices so it starts with the lexicographically smallest point
    2. Sort faces lexicographically by their point coordinates

    Returns canonical (faces, points) suitable for comparison.
    """
    canonical_faces = []

    for face in faces:
        # Get the points of this face
        face_points = points[face]

        # Find the index of the lexicographically smallest point
        point_tuples = [tuple(p) for p in face_points]
        min_idx = min(range(ngon), key=lambda i: point_tuples[i])

        # Rotate face so it starts with the smallest point
        rotated_face = np.roll(face, -min_idx)
        canonical_faces.append(rotated_face)

    canonical_faces = np.array(canonical_faces)

    # Now sort faces by their point coordinates
    sort_keys = []
    for face in canonical_faces:
        face_points = points[face]
        sort_key = tuple(face_points.flatten())
        sort_keys.append(sort_key)

    sort_order = sorted(range(len(sort_keys)), key=lambda i: sort_keys[i])
    canonical_faces = canonical_faces[sort_order]

    return canonical_faces, points


def assert_meshes_equal(faces1, points1, faces2, points2, ngon, atol=1e-5):
    """
    Assert that two meshes are geometrically equivalent.

    Handles the fact that vertices may be reordered.
    Compares meshes by canonicalizing them first.
    """
    # Check shapes match
    assert faces1.shape == faces2.shape, \
        f"Face count mismatch: {faces1.shape[0]} vs {faces2.shape[0]}"
    assert points1.shape == points2.shape, \
        f"Point count mismatch: {points1.shape[0]} vs {points2.shape[0]}"

    # Canonicalize both meshes
    faces1_canon, points1_canon = canonicalize_mesh(faces1, points1, ngon)
    faces2_canon, points2_canon = canonicalize_mesh(faces2, points2, ngon)

    # Compare canonicalized faces by their point coordinates
    for i, (face1, face2) in enumerate(zip(faces1_canon, faces2_canon)):
        points1_face = points1_canon[face1]
        points2_face = points2_canon[face2]

        assert np.allclose(points1_face, points2_face, atol=atol), \
            f"Face {i} points mismatch:\n" \
            f"  Mesh 1: {points1_face}\n" \
            f"  Mesh 2: {points2_face}"


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("point_dtype", POINT_DTYPES)
@pytest.mark.parametrize("ngon", NGONS)
def test_write_obj_simple(index_dtype, point_dtype, ngon):
    """Test writing a simple mesh"""
    with tempfile.TemporaryDirectory() as tmpdir:
        obj_file = os.path.join(tmpdir, "test.obj")

        # Create a simple face
        if ngon == 3:
            faces = np.array([[0, 1, 2]], dtype=index_dtype)
            points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=point_dtype)
        else:
            faces = np.array([[0, 1, 2, 3]], dtype=index_dtype)
            points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=point_dtype)

        # Write OBJ
        success = tf.write_obj((faces, points), obj_file)
        assert success, "write_obj should return True on success"

        # Verify file was created
        assert os.path.exists(obj_file), "OBJ file should be created"
        assert os.path.getsize(obj_file) > 0, "OBJ file should not be empty"


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("ngon", NGONS)
def test_write_obj_round_trip(index_dtype, ngon):
    """Test writing and reading back produces same data"""
    with tempfile.TemporaryDirectory() as tmpdir:
        obj_file = os.path.join(tmpdir, "round_trip.obj")

        # Create a cube
        points_orig = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
        ], dtype=np.float32)

        if ngon == 3:
            # 12 triangles
            faces_orig = np.array([
                [0, 1, 2], [0, 2, 3],  # bottom
                [4, 6, 5], [4, 7, 6],  # top
                [0, 5, 1], [0, 4, 5],  # front
                [2, 7, 3], [2, 6, 7],  # back
                [0, 3, 7], [0, 7, 4],  # left
                [1, 6, 2], [1, 5, 6],  # right
            ], dtype=index_dtype)
        else:
            # 6 quads
            faces_orig = np.array([
                [0, 1, 2, 3],  # bottom
                [4, 7, 6, 5],  # top
                [0, 4, 5, 1],  # front
                [2, 6, 7, 3],  # back
                [0, 3, 7, 4],  # left
                [1, 5, 6, 2],  # right
            ], dtype=index_dtype)

        # Write OBJ
        success = tf.write_obj((faces_orig, points_orig), obj_file)
        assert success, "write_obj should succeed"

        # Read it back
        faces_read, points_read = tf.read_obj(obj_file, ngon=ngon, index_dtype=index_dtype)

        # Check shapes
        assert faces_read.shape == faces_orig.shape, "Faces shape should match"
        assert points_read.shape == points_orig.shape, "Points shape should match"

        # Check dtypes
        assert faces_read.dtype == index_dtype, f"Faces dtype should be {index_dtype}"
        assert points_read.dtype == np.float32, "Points dtype should be float32"

        # Compare meshes geometrically
        assert_meshes_equal(faces_orig, points_orig, faces_read, points_read, ngon)


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("ngon", NGONS)
def test_write_obj_with_transformation(index_dtype, ngon):
    """Test writing with a transformation matrix"""
    with tempfile.TemporaryDirectory() as tmpdir:
        obj_file = os.path.join(tmpdir, "transformed.obj")

        # Create a simple face at origin
        if ngon == 3:
            faces = np.array([[0, 1, 2]], dtype=index_dtype)
            points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        else:
            faces = np.array([[0, 1, 2, 3]], dtype=index_dtype)
            points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32)

        # Create transformation: translate by (5, 0, 10)
        transformation = np.eye(4, dtype=np.float32)
        transformation[0, 3] = 5.0  # x translation
        transformation[2, 3] = 10.0  # z translation

        # Write with transformation
        success = tf.write_obj((faces, points), obj_file, transformation=transformation)
        assert success, "write_obj with transformation should succeed"

        # Read back
        faces_read, points_read = tf.read_obj(obj_file, ngon=ngon, index_dtype=index_dtype)

        # Expected transformed mesh
        if ngon == 3:
            expected_faces = np.array([[0, 1, 2]], dtype=index_dtype)
            expected_points = np.array([
                [5, 0, 10], [6, 0, 10], [5, 1, 10]
            ], dtype=np.float32)
        else:
            expected_faces = np.array([[0, 1, 2, 3]], dtype=index_dtype)
            expected_points = np.array([
                [5, 0, 10], [6, 0, 10], [6, 1, 10], [5, 1, 10]
            ], dtype=np.float32)

        # Compare meshes geometrically
        assert_meshes_equal(expected_faces, expected_points, faces_read, points_read, ngon)


def test_write_obj_filename_extension():
    """Test that .obj extension is appended if not present"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write without .obj extension
        filename_base = os.path.join(tmpdir, "test")

        faces = np.array([[0, 1, 2]], dtype=np.int32)
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)

        success = tf.write_obj((faces, points), filename_base)
        assert success, "write_obj should succeed"

        # Check that file with .obj extension was created
        expected_file = filename_base + ".obj"
        assert os.path.exists(expected_file), \
            ".obj extension should be appended automatically"


def test_write_obj_invalid_faces_shape():
    """Test that invalid faces shape raises error"""
    with tempfile.TemporaryDirectory() as tmpdir:
        obj_file = os.path.join(tmpdir, "test.obj")

        # Invalid faces: shape (N, 5) - not 3 or 4
        faces = np.array([[0, 1, 2, 3, 4]], dtype=np.int32)
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0.5, 0.5, 0]],
                         dtype=np.float32)

        with pytest.raises(ValueError, match="faces must have shape"):
            tf.write_obj((faces, points), obj_file)


def test_write_obj_invalid_faces_dtype():
    """Test that invalid faces dtype raises error"""
    with tempfile.TemporaryDirectory() as tmpdir:
        obj_file = os.path.join(tmpdir, "test.obj")

        # Invalid dtype: float instead of int
        faces = np.array([[0, 1, 2]], dtype=np.float32)
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)

        with pytest.raises(ValueError, match="faces dtype must be"):
            tf.write_obj((faces, points), obj_file)


def test_write_obj_invalid_points_shape():
    """Test that invalid points shape raises error"""
    with tempfile.TemporaryDirectory() as tmpdir:
        obj_file = os.path.join(tmpdir, "test.obj")

        # Invalid points: shape (N, 2) instead of (N, 3)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        points = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32)

        with pytest.raises(ValueError, match="points must have shape"):
            tf.write_obj((faces, points), obj_file)


def test_write_obj_invalid_points_dtype():
    """Test that invalid points dtype raises error"""
    with tempfile.TemporaryDirectory() as tmpdir:
        obj_file = os.path.join(tmpdir, "test.obj")

        # Invalid dtype: int instead of float
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.int32)

        with pytest.raises(ValueError, match="points dtype must be"):
            tf.write_obj((faces, points), obj_file)


def test_write_obj_invalid_transformation_shape():
    """Test that invalid transformation shape raises error"""
    with tempfile.TemporaryDirectory() as tmpdir:
        obj_file = os.path.join(tmpdir, "test.obj")

        faces = np.array([[0, 1, 2]], dtype=np.int32)
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)

        # Invalid transformation: shape (3, 3) instead of (4, 4)
        transformation = np.eye(3, dtype=np.float32)

        with pytest.raises(ValueError, match="transformation must have shape"):
            tf.write_obj((faces, points), obj_file, transformation=transformation)


def test_write_obj_invalid_transformation_dtype():
    """Test that invalid transformation dtype raises error"""
    with tempfile.TemporaryDirectory() as tmpdir:
        obj_file = os.path.join(tmpdir, "test.obj")

        faces = np.array([[0, 1, 2]], dtype=np.int32)
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)

        # Invalid dtype: float64 instead of float32
        transformation = np.eye(4, dtype=np.float64)

        with pytest.raises(ValueError, match="transformation dtype must be"):
            tf.write_obj((faces, points), obj_file, transformation=transformation)


def test_write_obj_non_contiguous_arrays():
    """Test that non-contiguous arrays are handled correctly"""
    with tempfile.TemporaryDirectory() as tmpdir:
        obj_file = os.path.join(tmpdir, "test.obj")

        # Create non-contiguous arrays via slicing
        faces_full = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int32)
        points_full = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]],
                              dtype=np.float32)

        # Take slices
        faces = faces_full[::1, :]
        points = points_full[::1, :]

        # Should work even if arrays are not contiguous
        success = tf.write_obj((faces, points), obj_file)
        assert success, "write_obj should handle non-contiguous arrays"

        # Verify by reading back
        faces_read, points_read = tf.read_obj(obj_file, ngon=3)
        assert faces_read.shape[0] == 2, "Should have 2 faces"
        assert points_read.shape[0] == 4, "Should have 4 points"


@pytest.mark.parametrize("ngon", NGONS)
def test_write_obj_from_mesh_object(ngon):
    """Test writing from a Mesh object"""
    with tempfile.TemporaryDirectory() as tmpdir:
        obj_file = os.path.join(tmpdir, "mesh.obj")

        # Create a Mesh object
        if ngon == 3:
            faces = np.array([[0, 1, 2]], dtype=np.int32)
            points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        else:
            faces = np.array([[0, 1, 2, 3]], dtype=np.int32)
            points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32)

        mesh = tf.Mesh(faces, points)

        # Write from mesh
        success = tf.write_obj(mesh, obj_file)
        assert success, "write_obj should succeed with Mesh object"

        # Read back and verify
        faces_read, points_read = tf.read_obj(obj_file, ngon=ngon)

        assert_meshes_equal(faces, points, faces_read, points_read, ngon)


@pytest.mark.parametrize("ngon", NGONS)
def test_write_obj_mesh_with_transformation(ngon):
    """Test writing a Mesh with transformation property set"""
    with tempfile.TemporaryDirectory() as tmpdir:
        obj_file = os.path.join(tmpdir, "mesh_transformed.obj")

        # Create a Mesh at origin
        if ngon == 3:
            faces = np.array([[0, 1, 2]], dtype=np.int32)
            points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        else:
            faces = np.array([[0, 1, 2, 3]], dtype=np.int32)
            points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32)

        mesh = tf.Mesh(faces, points)

        # Set transformation on mesh
        transformation = np.eye(4, dtype=np.float32)
        transformation[0, 3] = 2.0  # Translate 2 in X
        transformation[1, 3] = 3.0  # Translate 3 in Y
        mesh.transformation = transformation

        # Write mesh (should apply transformation)
        success = tf.write_obj(mesh, obj_file)
        assert success, "write_obj should succeed with transformed Mesh"

        # Read back
        faces_read, points_read = tf.read_obj(obj_file, ngon=ngon)

        # Expected transformed mesh
        if ngon == 3:
            expected_faces = np.array([[0, 1, 2]], dtype=np.int32)
            expected_points = np.array([[2, 3, 0], [3, 3, 0], [2, 4, 0]], dtype=np.float32)
        else:
            expected_faces = np.array([[0, 1, 2, 3]], dtype=np.int32)
            expected_points = np.array([[2, 3, 0], [3, 3, 0], [3, 4, 0], [2, 4, 0]], dtype=np.float32)

        assert_meshes_equal(expected_faces, expected_points, faces_read, points_read, ngon)


def test_write_obj_mesh_2d_error():
    """Test that 2D mesh raises error"""
    with tempfile.TemporaryDirectory() as tmpdir:
        obj_file = os.path.join(tmpdir, "2d.obj")

        # Create a 2D mesh
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        points = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32)
        mesh = tf.Mesh(faces, points)

        # Should raise error for 2D mesh
        with pytest.raises(ValueError, match="OBJ format only supports 3D meshes"):
            tf.write_obj(mesh, obj_file)


def test_write_obj_mesh_dynamic_error():
    """Test that dynamic mesh raises error"""
    with tempfile.TemporaryDirectory() as tmpdir:
        obj_file = os.path.join(tmpdir, "dynamic.obj")

        # Create a dynamic mesh
        offsets = np.array([0, 3, 7], dtype=np.int32)
        data = np.array([0, 1, 2, 0, 2, 3, 4], dtype=np.int32)
        faces = tf.OffsetBlockedArray(offsets, data)
        points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.5, 0.5, 0]], dtype=np.float32)
        mesh = tf.Mesh(faces, points)

        # Should raise error for dynamic mesh
        with pytest.raises(ValueError, match="dynamic"):
            tf.write_obj(mesh, obj_file)


if __name__ == "__main__":
    # Run tests with pytest
    sys.exit(pytest.main([__file__, "-v"]))
