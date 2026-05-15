"""
Test read_obj functionality

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
NGONS = [3, 4]


def create_simple_triangle_obj(filename):
    """Create a simple OBJ file with a single triangle"""
    obj_content = """# Simple triangle
v 0 0 0
v 1 0 0
v 0 1 0
f 1 2 3
"""
    with open(filename, 'w') as f:
        f.write(obj_content)


def create_simple_quad_obj(filename):
    """Create a simple OBJ file with a single quad"""
    obj_content = """# Simple quad
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
f 1 2 3 4
"""
    with open(filename, 'w') as f:
        f.write(obj_content)


def create_cube_triangles_obj(filename):
    """Create an OBJ file with a cube (12 triangles)"""
    obj_content = """# Cube with triangles
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
v 0 0 1
v 1 0 1
v 1 1 1
v 0 1 1
f 1 2 3
f 1 3 4
f 5 7 6
f 5 8 7
f 1 6 2
f 1 5 6
f 3 8 4
f 3 7 8
f 1 4 8
f 1 8 5
f 2 7 3
f 2 6 7
"""
    with open(filename, 'w') as f:
        f.write(obj_content)


def create_cube_quads_obj(filename):
    """Create an OBJ file with a cube (6 quads)"""
    obj_content = """# Cube with quads
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
v 0 0 1
v 1 0 1
v 1 1 1
v 0 1 1
f 1 2 3 4
f 5 8 7 6
f 1 5 6 2
f 3 7 8 4
f 1 4 8 5
f 2 6 7 3
"""
    with open(filename, 'w') as f:
        f.write(obj_content)


def create_simple_obj(filename, ngon):
    """Create a simple OBJ file with the given ngon"""
    if ngon == 3:
        create_simple_triangle_obj(filename)
    else:
        create_simple_quad_obj(filename)


def create_cube_obj(filename, ngon):
    """Create a cube OBJ file with the given ngon"""
    if ngon == 3:
        create_cube_triangles_obj(filename)
    else:
        create_cube_quads_obj(filename)


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("point_dtype", POINT_DTYPES)
@pytest.mark.parametrize("ngon", NGONS)
def test_read_obj_simple(index_dtype, point_dtype, ngon):
    """Test reading a simple OBJ with one face"""
    with tempfile.TemporaryDirectory() as tmpdir:
        obj_file = os.path.join(tmpdir, "test.obj")
        create_simple_obj(obj_file, ngon)

        faces, points = tf.read_obj(
            obj_file, ngon=ngon, index_dtype=index_dtype, dtype=point_dtype
        )

        # Check shapes
        assert faces.shape[1] == ngon, f"Faces should have {ngon} columns"
        assert points.shape[1] == 3, "Points should have 3 columns"

        # Check dtypes
        assert faces.dtype == index_dtype, f"Faces should be {index_dtype}"
        assert points.dtype == point_dtype, f"Points should be {point_dtype}"

        # Check that we have at least one face
        assert faces.shape[0] >= 1, "Should have at least one face"

        # Check that face indices are valid (0-based)
        assert np.all(faces >= 0), "Face indices should be non-negative"
        assert np.all(faces < points.shape[0]), "Face indices should be within points range"

        # Check that points are 3D
        assert points.shape[0] >= ngon, f"Should have at least {ngon} points"


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("point_dtype", POINT_DTYPES)
@pytest.mark.parametrize("ngon", NGONS)
def test_read_obj_cube(index_dtype, point_dtype, ngon):
    """Test reading a cube OBJ"""
    with tempfile.TemporaryDirectory() as tmpdir:
        obj_file = os.path.join(tmpdir, "cube.obj")
        create_cube_obj(obj_file, ngon)

        faces, points = tf.read_obj(
            obj_file, ngon=ngon, index_dtype=index_dtype, dtype=point_dtype
        )

        # Check dtypes
        assert faces.dtype == index_dtype, f"Faces should be {index_dtype}"
        assert points.dtype == point_dtype, f"Points should be {point_dtype}"

        # Cube has 8 vertices
        assert points.shape[0] == 8, f"Cube should have 8 points, got {points.shape[0]}"

        # Cube has 12 triangles or 6 quads
        expected_faces = 12 if ngon == 3 else 6
        assert faces.shape[0] == expected_faces, \
            f"Cube should have {expected_faces} faces, got {faces.shape[0]}"

        # Check face indices are valid
        assert np.all(faces >= 0), "Face indices should be non-negative"
        assert np.all(faces < 8), "Face indices should be < 8 for a cube"

        # Check that points are within expected range [0, 1]
        assert np.all(points >= 0.0), "Points should be >= 0"
        assert np.all(points <= 1.0), "Points should be <= 1"


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("ngon", NGONS)
def test_read_obj_index_conversion(index_dtype, ngon):
    """Test that OBJ 1-based indices are converted to 0-based"""
    with tempfile.TemporaryDirectory() as tmpdir:
        obj_file = os.path.join(tmpdir, "test.obj")
        create_simple_obj(obj_file, ngon)

        faces, points = tf.read_obj(obj_file, ngon=ngon, index_dtype=index_dtype)

        # First face should reference indices 0, 1, 2 (or 0, 1, 2, 3 for quads)
        # since OBJ uses 1-based indexing
        assert faces[0, 0] == 0, "First index should be 0 (converted from 1)"
        assert faces[0, 1] == 1, "Second index should be 1 (converted from 2)"
        assert faces[0, 2] == 2, "Third index should be 2 (converted from 3)"
        if ngon == 4:
            assert faces[0, 3] == 3, "Fourth index should be 3 (converted from 4)"


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("ngon", NGONS)
def test_read_obj_face_formats(index_dtype, ngon):
    """Test reading OBJ with various face formats (v/vt/vn)"""
    with tempfile.TemporaryDirectory() as tmpdir:
        obj_file = os.path.join(tmpdir, "test.obj")

        if ngon == 3:
            # OBJ with texture coords and normals
            obj_content = """# Triangle with various face formats
v 0 0 0
v 1 0 0
v 0 1 0
vt 0 0
vt 1 0
vt 0 1
vn 0 0 1
f 1/1/1 2/2/1 3/3/1
"""
        else:
            # Quad with texture coords and normals
            obj_content = """# Quad with various face formats
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
vt 0 0
vt 1 0
vt 1 1
vt 0 1
vn 0 0 1
f 1/1/1 2/2/1 3/3/1 4/4/1
"""
        with open(obj_file, 'w') as f:
            f.write(obj_content)

        faces, points = tf.read_obj(obj_file, ngon=ngon, index_dtype=index_dtype)

        # Check that we got the face correctly (ignoring tex coords and normals)
        assert faces.shape[0] == 1, "Should have 1 face"
        assert faces.shape[1] == ngon, f"Face should have {ngon} vertices"

        # Indices should be 0-based
        expected_indices = list(range(ngon))
        for i, expected in enumerate(expected_indices):
            assert faces[0, i] == expected, \
                f"Index {i} should be {expected}, got {faces[0, i]}"


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("ngon", NGONS)
def test_read_obj_memory_ownership(index_dtype, ngon):
    """Test that numpy arrays properly own their memory"""
    with tempfile.TemporaryDirectory() as tmpdir:
        obj_file = os.path.join(tmpdir, "test.obj")
        create_simple_obj(obj_file, ngon)

        faces, points = tf.read_obj(obj_file, ngon=ngon, index_dtype=index_dtype)

        # Check that arrays are writable
        assert faces.flags['WRITEABLE'], "Faces array should be writeable"
        assert points.flags['WRITEABLE'], "Points array should be writeable"

        # Test that we can modify the arrays
        original_point = points[0].copy()
        points[0] = [999.0, 999.0, 999.0]
        assert not np.array_equal(points[0], original_point), "Should be able to modify points"


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_read_obj_with_comments(index_dtype):
    """Test reading OBJ with comments"""
    with tempfile.TemporaryDirectory() as tmpdir:
        obj_file = os.path.join(tmpdir, "test.obj")
        obj_content = """# This is a comment
# Another comment
v 0 0 0
v 1 0 0
v 0 1 0
# Comment between data
f 1 2 3
# Trailing comment
"""
        with open(obj_file, 'w') as f:
            f.write(obj_content)

        faces, points = tf.read_obj(obj_file, ngon=3, index_dtype=index_dtype)

        assert faces.shape[0] == 1, "Should have 1 face"
        assert points.shape[0] == 3, "Should have 3 points"


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_read_obj_double_precision_preserved(index_dtype):
    """A double-precision-only coordinate must round-trip through read_obj."""
    with tempfile.TemporaryDirectory() as tmpdir:
        obj_file = os.path.join(tmpdir, "precision.obj")
        # 1.0000000000000002 is the smallest double > 1.0; float32 cannot represent it.
        with open(obj_file, "w") as f:
            f.write("v 1.0000000000000002 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")

        faces, points = tf.read_obj(
            obj_file, ngon=3, index_dtype=index_dtype, dtype=np.float64
        )

        assert points.dtype == np.float64
        # The exact double value survived; a float32 read would collapse to 1.0.
        assert points[0, 0] != 1.0
        assert points[0, 0] == np.float64(1.0000000000000002)


def test_read_obj_invalid_dtype_real():
    """dtype must be float32 or float64."""
    with tempfile.TemporaryDirectory() as tmpdir:
        obj_file = os.path.join(tmpdir, "test.obj")
        create_simple_triangle_obj(obj_file)
        with pytest.raises(ValueError, match="dtype must be"):
            tf.read_obj(obj_file, ngon=3, dtype=np.int32)


def test_read_obj_invalid_ngon():
    """Test that invalid ngon raises error"""
    with tempfile.TemporaryDirectory() as tmpdir:
        obj_file = os.path.join(tmpdir, "test.obj")
        create_simple_triangle_obj(obj_file)

        with pytest.raises(ValueError, match="ngon must be 3, 4, or None"):
            tf.read_obj(obj_file, ngon=5)


def test_read_obj_invalid_dtype():
    """Test that invalid index_dtype raises error"""
    with tempfile.TemporaryDirectory() as tmpdir:
        obj_file = os.path.join(tmpdir, "test.obj")
        create_simple_triangle_obj(obj_file)

        with pytest.raises(ValueError, match="index_dtype must be"):
            tf.read_obj(obj_file, ngon=3, index_dtype=np.float32)


# =============================================================================
# Dynamic read tests
# =============================================================================


def create_mixed_obj(filename):
    """Create an OBJ file with mixed triangles and quads"""
    obj_content = """# Mixed: triangles and quads
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
v 2 0 0
f 1 2 3
f 1 3 4
f 1 2 5 3
"""
    with open(filename, 'w') as f:
        f.write(obj_content)


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("point_dtype", POINT_DTYPES)
def test_read_obj_dynamic_default(index_dtype, point_dtype):
    """Test that default read (ngon=None) returns OffsetBlockedArray"""
    with tempfile.TemporaryDirectory() as tmpdir:
        obj_file = os.path.join(tmpdir, "test.obj")
        create_simple_triangle_obj(obj_file)

        faces, points = tf.read_obj(
            obj_file, index_dtype=index_dtype, dtype=point_dtype
        )

        assert isinstance(faces, tf.OffsetBlockedArray), \
            f"Expected OffsetBlockedArray, got {type(faces)}"
        assert isinstance(points, np.ndarray)
        assert points.dtype == point_dtype
        assert len(faces) == 1
        assert points.shape == (3, 3)


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_read_obj_dynamic_triangles(index_dtype):
    """Test dynamic read of pure triangles"""
    with tempfile.TemporaryDirectory() as tmpdir:
        obj_file = os.path.join(tmpdir, "cube.obj")
        create_cube_triangles_obj(obj_file)

        faces, points = tf.read_obj(obj_file, index_dtype=index_dtype)

        assert isinstance(faces, tf.OffsetBlockedArray)
        assert len(faces) == 12
        assert points.shape[0] == 8

        # Each face should be a triangle
        for i in range(len(faces)):
            assert len(faces[i]) == 3, f"Face {i} should have 3 vertices"


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_read_obj_dynamic_quads(index_dtype):
    """Test dynamic read of pure quads"""
    with tempfile.TemporaryDirectory() as tmpdir:
        obj_file = os.path.join(tmpdir, "cube.obj")
        create_cube_quads_obj(obj_file)

        faces, points = tf.read_obj(obj_file, index_dtype=index_dtype)

        assert isinstance(faces, tf.OffsetBlockedArray)
        assert len(faces) == 6
        assert points.shape[0] == 8

        # Each face should be a quad
        for i in range(len(faces)):
            assert len(faces[i]) == 4, f"Face {i} should have 4 vertices"


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_read_obj_dynamic_mixed(index_dtype):
    """Test dynamic read of mixed polygons (triangles + quads)"""
    with tempfile.TemporaryDirectory() as tmpdir:
        obj_file = os.path.join(tmpdir, "mixed.obj")
        create_mixed_obj(obj_file)

        faces, points = tf.read_obj(obj_file, index_dtype=index_dtype)

        assert isinstance(faces, tf.OffsetBlockedArray)
        assert len(faces) == 3
        assert points.shape[0] == 5

        # First two faces are triangles, third is a quad
        assert len(faces[0]) == 3
        assert len(faces[1]) == 3
        assert len(faces[2]) == 4

        # Check 0-based indices of first face
        assert faces[0][0] == 0
        assert faces[0][1] == 1
        assert faces[0][2] == 2


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_read_obj_dynamic_index_dtype(index_dtype):
    """Test that dynamic read respects index_dtype"""
    with tempfile.TemporaryDirectory() as tmpdir:
        obj_file = os.path.join(tmpdir, "test.obj")
        create_simple_triangle_obj(obj_file)

        faces, points = tf.read_obj(obj_file, index_dtype=index_dtype)

        assert faces.dtype == index_dtype, f"Expected dtype {index_dtype}, got {faces.dtype}"


if __name__ == "__main__":
    # Run tests with pytest
    sys.exit(pytest.main([__file__, "-v"]))
