"""
Tests for concatenated function.

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

import sys

import pytest
import numpy as np
import trueform as tf


class TestConcatenatedMesh:
    """Tests for concatenating Mesh objects."""

    def test_concatenate_two_triangle_meshes(self):
        """Test concatenating two triangle meshes."""
        # First mesh: single triangle
        faces1 = np.array([[0, 1, 2]], dtype=np.int32)
        points1 = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        mesh1 = tf.Mesh(faces1, points1)

        # Second mesh: single triangle
        faces2 = np.array([[0, 1, 2]], dtype=np.int32)
        points2 = np.array([[5, 5, 5], [6, 5, 5], [5, 6, 5]], dtype=np.float32)
        mesh2 = tf.Mesh(faces2, points2)

        # Concatenate
        result_faces, result_points = tf.concatenated([mesh1, mesh2])

        # Verify faces are offset correctly
        expected_faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
        np.testing.assert_array_equal(result_faces, expected_faces)

        # Verify points are concatenated
        expected_points = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0],
            [5, 5, 5], [6, 5, 5], [5, 6, 5]
        ], dtype=np.float32)
        np.testing.assert_array_equal(result_points, expected_points)

    def test_concatenate_multiple_triangle_meshes(self):
        """Test concatenating three triangle meshes."""
        mesh1 = tf.Mesh(
            np.array([[0, 1, 2]], dtype=np.int32),
            np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        )
        mesh2 = tf.Mesh(
            np.array([[0, 1, 2]], dtype=np.int32),
            np.array([[2, 0, 0], [3, 0, 0], [2, 1, 0]], dtype=np.float32)
        )
        mesh3 = tf.Mesh(
            np.array([[0, 1, 2]], dtype=np.int32),
            np.array([[4, 0, 0], [5, 0, 0], [4, 1, 0]], dtype=np.float32)
        )

        result_faces, result_points = tf.concatenated([mesh1, mesh2, mesh3])

        # Verify faces
        expected_faces = np.array([
            [0, 1, 2],  # mesh1
            [3, 4, 5],  # mesh2 (offset by 3)
            [6, 7, 8],  # mesh3 (offset by 6)
        ], dtype=np.int32)
        np.testing.assert_array_equal(result_faces, expected_faces)

        # Verify points
        assert len(result_points) == 9

    def test_concatenate_dynamic_meshes(self):
        """Test concatenating dynamic meshes (variable-sized polygons)."""
        # First dynamic mesh: one triangle and one quad
        offsets1 = np.array([0, 3, 7], dtype=np.int32)
        data1 = np.array([0, 1, 2, 0, 2, 3, 4], dtype=np.int32)
        faces1 = tf.OffsetBlockedArray(offsets1, data1)
        points1 = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.5, 0.5, 0]], dtype=np.float32)
        mesh1 = tf.Mesh(faces1, points1)

        # Second dynamic mesh: two triangles
        offsets2 = np.array([0, 3, 6], dtype=np.int32)
        data2 = np.array([0, 1, 2, 1, 2, 3], dtype=np.int32)
        faces2 = tf.OffsetBlockedArray(offsets2, data2)
        points2 = np.array([[10, 0, 0], [11, 0, 0], [10, 1, 0], [11, 1, 0]], dtype=np.float32)
        mesh2 = tf.Mesh(faces2, points2)

        # Concatenate
        result_faces, result_points = tf.concatenated([mesh1, mesh2])

        # Verify result type
        assert isinstance(result_faces, tf.OffsetBlockedArray)

        # Verify number of faces
        assert len(result_faces) == 4  # 2 from mesh1 + 2 from mesh2

        # Verify points concatenated
        assert len(result_points) == 9  # 5 from mesh1 + 4 from mesh2

        # Verify face indices are correctly offset
        # mesh1 faces: [0,1,2], [0,2,3,4] -> no offset
        # mesh2 faces: [0,1,2], [1,2,3] -> offset by 5 -> [5,6,7], [6,7,8]
        np.testing.assert_array_equal(result_faces[0], [0, 1, 2])
        np.testing.assert_array_equal(result_faces[1], [0, 2, 3, 4])
        np.testing.assert_array_equal(result_faces[2], [5, 6, 7])
        np.testing.assert_array_equal(result_faces[3], [6, 7, 8])

    def test_concatenate_2d_meshes(self):
        """Test concatenating 2D triangle meshes."""
        mesh1 = tf.Mesh(
            np.array([[0, 1, 2]], dtype=np.int32),
            np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32)
        )
        mesh2 = tf.Mesh(
            np.array([[0, 1, 2]], dtype=np.int32),
            np.array([[2, 0], [3, 0], [2, 1]], dtype=np.float32)
        )

        result_faces, result_points = tf.concatenated([mesh1, mesh2])

        # Verify dimensions
        assert result_faces.shape == (2, 3)
        assert result_points.shape == (6, 2)


class TestConcatenatedEdgeMesh:
    """Tests for concatenating EdgeMesh objects."""

    def test_concatenate_two_edge_meshes_2d(self):
        """Test concatenating two 2D edge meshes."""
        edges1 = np.array([[0, 1]], dtype=np.int32)
        points1 = np.array([[0, 0], [1, 0]], dtype=np.float32)
        edgemesh1 = tf.EdgeMesh(edges1, points1)

        edges2 = np.array([[0, 1]], dtype=np.int32)
        points2 = np.array([[2, 0], [3, 0]], dtype=np.float32)
        edgemesh2 = tf.EdgeMesh(edges2, points2)

        result_edges, result_points = tf.concatenated([edgemesh1, edgemesh2])

        # Verify edges are offset correctly
        expected_edges = np.array([[0, 1], [2, 3]], dtype=np.int32)
        np.testing.assert_array_equal(result_edges, expected_edges)

        # Verify points
        expected_points = np.array([[0, 0], [1, 0], [2, 0], [3, 0]], dtype=np.float32)
        np.testing.assert_array_equal(result_points, expected_points)

    def test_concatenate_three_edge_meshes_3d(self):
        """Test concatenating three 3D edge meshes."""
        edgemesh1 = tf.EdgeMesh(
            np.array([[0, 1]], dtype=np.int32),
            np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
        )
        edgemesh2 = tf.EdgeMesh(
            np.array([[0, 1]], dtype=np.int32),
            np.array([[2, 0, 0], [3, 0, 0]], dtype=np.float32)
        )
        edgemesh3 = tf.EdgeMesh(
            np.array([[0, 1]], dtype=np.int32),
            np.array([[4, 0, 0], [5, 0, 0]], dtype=np.float32)
        )

        result_edges, result_points = tf.concatenated([edgemesh1, edgemesh2, edgemesh3])

        # Verify edges
        expected_edges = np.array([[0, 1], [2, 3], [4, 5]], dtype=np.int32)
        np.testing.assert_array_equal(result_edges, expected_edges)
        assert len(result_points) == 6


class TestConcatenatedTuples:
    """Tests for concatenating tuples of (indices, points)."""

    def test_concatenate_tuples(self):
        """Test concatenating raw (indices, points) tuples."""
        data1 = (
            np.array([[0, 1, 2]], dtype=np.int32),
            np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        )
        data2 = (
            np.array([[0, 1, 2]], dtype=np.int32),
            np.array([[5, 5, 5], [6, 5, 5], [5, 6, 5]], dtype=np.float32)
        )

        result_indices, result_points = tf.concatenated([data1, data2])

        expected_indices = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
        np.testing.assert_array_equal(result_indices, expected_indices)
        assert len(result_points) == 6

    def test_concatenate_dynamic_tuples(self):
        """Test concatenating tuples with OffsetBlockedArray indices."""
        # First tuple: one triangle, one quad
        offsets1 = np.array([0, 3, 7], dtype=np.int32)
        data1_arr = np.array([0, 1, 2, 0, 2, 3, 4], dtype=np.int32)
        indices1 = tf.OffsetBlockedArray(offsets1, data1_arr)
        points1 = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.5, 0.5, 0]], dtype=np.float32)
        data1 = (indices1, points1)

        # Second tuple: two triangles (same dtype as first)
        offsets2 = np.array([0, 3, 6], dtype=np.int32)
        data2_arr = np.array([0, 1, 2, 1, 2, 3], dtype=np.int32)
        indices2 = tf.OffsetBlockedArray(offsets2, data2_arr)
        points2 = np.array([[10, 0, 0], [11, 0, 0], [10, 1, 0], [11, 1, 0]], dtype=np.float32)
        data2 = (indices2, points2)

        result_indices, result_points = tf.concatenated([data1, data2])

        # Verify result is OffsetBlockedArray
        assert isinstance(result_indices, tf.OffsetBlockedArray)

        # Verify face count
        assert len(result_indices) == 4

        # Verify point count
        assert len(result_points) == 9

        # Verify face indices are correctly offset
        np.testing.assert_array_equal(result_indices[0], [0, 1, 2])
        np.testing.assert_array_equal(result_indices[1], [0, 2, 3, 4])
        np.testing.assert_array_equal(result_indices[2], [5, 6, 7])
        np.testing.assert_array_equal(result_indices[3], [6, 7, 8])


class TestConcatenatedMixedDtypes:
    """Tests for concatenating with mixed dtypes (numpy auto-conversion)."""

    def test_mixed_index_dtypes(self):
        """Test concatenating meshes with different index dtypes."""
        mesh1 = tf.Mesh(
            np.array([[0, 1, 2]], dtype=np.int32),
            np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        )
        mesh2 = tf.Mesh(
            np.array([[0, 1, 2]], dtype=np.int64),  # Different index dtype
            np.array([[5, 5, 5], [6, 5, 5], [5, 6, 5]], dtype=np.float32)
        )

        result_faces, result_points = tf.concatenated([mesh1, mesh2])

        # Numpy should handle type promotion
        assert result_faces.shape == (2, 3)
        assert result_points.shape == (6, 3)
        # Result dtype should be promoted (int64 takes precedence)
        assert result_faces.dtype == np.int64

    def test_mixed_real_dtypes(self):
        """Test concatenating meshes with different point dtypes."""
        mesh1 = tf.Mesh(
            np.array([[0, 1, 2]], dtype=np.int32),
            np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        )
        mesh2 = tf.Mesh(
            np.array([[0, 1, 2]], dtype=np.int32),
            np.array([[5, 5, 5], [6, 5, 5], [5, 6, 5]], dtype=np.float64)  # Different point dtype
        )

        result_faces, result_points = tf.concatenated([mesh1, mesh2])

        # Numpy should handle type promotion
        assert result_faces.shape == (2, 3)
        assert result_points.shape == (6, 3)
        # Result dtype should be promoted (float64 takes precedence)
        assert result_points.dtype == np.float64


class TestConcatenatedValidation:
    """Tests for validation and error handling."""

    def test_empty_list(self):
        """Test that empty list raises ValueError."""
        with pytest.raises(ValueError, match="Cannot concatenate empty list"):
            tf.concatenated([])

    def test_mixed_fixed_and_dynamic(self):
        """Test mixing fixed-size and dynamic meshes produces dynamic output."""
        # Fixed-size mesh (triangles)
        mesh1 = tf.Mesh(
            np.array([[0, 1, 2]], dtype=np.int32),
            np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        )
        # Dynamic mesh
        offsets = np.array([0, 3, 7], dtype=np.int32)
        data = np.array([0, 1, 2, 0, 2, 3, 4], dtype=np.int32)
        faces = tf.OffsetBlockedArray(offsets, data)
        mesh2 = tf.Mesh(faces, np.array([[5, 5, 5], [6, 5, 5], [5, 6, 5], [6, 6, 5], [5.5, 5.5, 5]], dtype=np.float32))

        result_faces, result_points = tf.concatenated([mesh1, mesh2])

        # Result should be dynamic
        assert isinstance(result_faces, tf.OffsetBlockedArray)
        assert len(result_faces) == 3  # 1 tri from mesh1 + 2 faces from mesh2
        assert len(result_points) == 8  # 3 + 5

        # Check indices are correctly offset
        np.testing.assert_array_equal(result_faces[0], [0, 1, 2])  # mesh1 tri
        np.testing.assert_array_equal(result_faces[1], [3, 4, 5])  # mesh2 tri (offset by 3)
        np.testing.assert_array_equal(result_faces[2], [3, 5, 6, 7])  # mesh2 quad (offset by 3)

    def test_mixed_ngon(self):
        """Test mixing triangles and dynamic (quad) produces dynamic output."""
        # Triangle mesh
        mesh1 = tf.Mesh(
            np.array([[0, 1, 2]], dtype=np.int32),
            np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        )
        # Quad mesh using OffsetBlockedArray (dynamic)
        quad_offsets = np.array([0, 4], dtype=np.int32)
        quad_data = np.array([0, 1, 2, 3], dtype=np.int32)
        mesh2 = tf.Mesh(
            tf.OffsetBlockedArray(quad_offsets, quad_data),
            np.array([[5, 5, 5], [6, 5, 5], [6, 6, 5], [5, 6, 5]], dtype=np.float32)
        )

        result_faces, result_points = tf.concatenated([mesh1, mesh2])

        # Result should be dynamic (mixed ngon)
        assert isinstance(result_faces, tf.OffsetBlockedArray)
        assert len(result_faces) == 2  # 1 tri + 1 quad
        assert len(result_points) == 7  # 3 + 4

        # Check indices
        np.testing.assert_array_equal(result_faces[0], [0, 1, 2])  # tri
        np.testing.assert_array_equal(result_faces[1], [3, 4, 5, 6])  # quad (offset by 3)

    def test_mismatched_dims(self):
        """Test that mismatched dims raises ValueError."""
        mesh1 = tf.Mesh(
            np.array([[0, 1, 2]], dtype=np.int32),
            np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32)  # 2D
        )
        mesh2 = tf.Mesh(
            np.array([[0, 1, 2]], dtype=np.int32),
            np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)  # 3D
        )

        with pytest.raises(ValueError, match="All points must have same dims"):
            tf.concatenated([mesh1, mesh2])

    def test_mixed_types_mesh_edgemesh(self):
        """Test that mixing Mesh and EdgeMesh raises TypeError."""
        mesh = tf.Mesh(
            np.array([[0, 1, 2]], dtype=np.int32),
            np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        )
        edgemesh = tf.EdgeMesh(
            np.array([[0, 1]], dtype=np.int32),
            np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
        )

        with pytest.raises(TypeError, match="All items must be Mesh objects"):
            tf.concatenated([mesh, edgemesh])

    def test_mixed_types_edgemesh_mesh(self):
        """Test that mixing EdgeMesh and Mesh raises TypeError."""
        edgemesh = tf.EdgeMesh(
            np.array([[0, 1]], dtype=np.int32),
            np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
        )
        mesh = tf.Mesh(
            np.array([[0, 1, 2]], dtype=np.int32),
            np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        )

        with pytest.raises(TypeError, match="All items must be EdgeMesh objects"):
            tf.concatenated([edgemesh, mesh])

    def test_single_mesh(self):
        """Test concatenating single mesh (should work)."""
        mesh = tf.Mesh(
            np.array([[0, 1, 2]], dtype=np.int32),
            np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        )

        result_faces, result_points = tf.concatenated([mesh])

        # Should return unchanged data
        np.testing.assert_array_equal(result_faces, mesh.faces)
        np.testing.assert_array_equal(result_points, mesh.points)

    def test_invalid_tuple_structure(self):
        """Test that invalid tuple structure raises TypeError."""
        data = [
            (np.array([[0, 1, 2]]), np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])),
            (np.array([[0, 1, 2]]),)  # Invalid: only one element
        ]

        with pytest.raises(TypeError, match="All items must be \\(indices, points\\) tuples"):
            tf.concatenated(data)


class TestConcatenatedReferentialIntegrity:
    """Tests to verify referential integrity is maintained."""

    def test_indices_reference_correct_points(self):
        """Test that indices correctly reference points after concatenation."""
        # Create two meshes with specific point values
        mesh1 = tf.Mesh(
            np.array([[0, 1, 2]], dtype=np.int32),
            np.array([[10, 10, 10], [20, 20, 20], [30, 30, 30]], dtype=np.float32)
        )
        mesh2 = tf.Mesh(
            np.array([[0, 1, 2]], dtype=np.int32),
            np.array([[100, 100, 100], [200, 200, 200], [300, 300, 300]], dtype=np.float32)
        )

        result_faces, result_points = tf.concatenated([mesh1, mesh2])

        # Verify first face references first three points
        face0 = result_faces[0]
        points0 = result_points[face0]
        np.testing.assert_array_equal(points0[0], [10, 10, 10])
        np.testing.assert_array_equal(points0[1], [20, 20, 20])
        np.testing.assert_array_equal(points0[2], [30, 30, 30])

        # Verify second face references last three points
        face1 = result_faces[1]
        points1 = result_points[face1]
        np.testing.assert_array_equal(points1[0], [100, 100, 100])
        np.testing.assert_array_equal(points1[1], [200, 200, 200])
        np.testing.assert_array_equal(points1[2], [300, 300, 300])

    def test_complex_offset_chain(self):
        """Test offset calculation with varying point counts."""
        # Three meshes with different numbers of points
        mesh1 = tf.Mesh(
            np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int32),  # 2 faces, 4 points
            np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32)
        )
        mesh2 = tf.Mesh(
            np.array([[0, 1, 2]], dtype=np.int32),  # 1 face, 3 points
            np.array([[10, 0, 0], [11, 0, 0], [10, 1, 0]], dtype=np.float32)
        )
        mesh3 = tf.Mesh(
            np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32),  # 2 faces, 4 points
            np.array([[20, 0, 0], [21, 0, 0], [21, 1, 0], [20, 1, 0]], dtype=np.float32)
        )

        result_faces, result_points = tf.concatenated([mesh1, mesh2, mesh3])

        # Expected faces:
        # mesh1: [[0, 1, 2], [1, 2, 3]] -> offset 0
        # mesh2: [[0, 1, 2]] -> offset 4 -> [[4, 5, 6]]
        # mesh3: [[0, 1, 2], [0, 2, 3]] -> offset 7 -> [[7, 8, 9], [7, 9, 10]]
        expected_faces = np.array([
            [0, 1, 2], [1, 2, 3],  # mesh1
            [4, 5, 6],              # mesh2
            [7, 8, 9], [7, 9, 10]   # mesh3
        ], dtype=np.int32)

        np.testing.assert_array_equal(result_faces, expected_faces)
        assert len(result_points) == 11  # 4 + 3 + 4


# ==============================================================================
# Transformation Tests
# ==============================================================================

def create_translation_3d(tx, ty, tz, dtype):
    """Create a 3D translation matrix."""
    return np.array([
        [1, 0, 0, tx],
        [0, 1, 0, ty],
        [0, 0, 1, tz],
        [0, 0, 0, 1]
    ], dtype=dtype)


def create_rotation_z_3d(angle_degrees, dtype):
    """Create a 3D rotation matrix around Z-axis."""
    angle = np.radians(angle_degrees)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    return np.array([
        [cos_a, -sin_a, 0, 0],
        [sin_a,  cos_a, 0, 0],
        [0,      0,     1, 0],
        [0,      0,     0, 1]
    ], dtype=dtype)


def create_translation_2d(tx, ty, dtype):
    """Create a 2D translation matrix."""
    return np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
    ], dtype=dtype)


class TestConcatenatedTransformations:
    """Tests for concatenated with transformations applied to meshes."""

    def test_mesh_with_translation(self):
        """Test concatenating mesh with translation transformation."""
        # Mesh at origin
        mesh1 = tf.Mesh(
            np.array([[0, 1, 2]], dtype=np.int32),
            np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        )
        # Mesh at origin with translation to (10, 0, 0)
        mesh2 = tf.Mesh(
            np.array([[0, 1, 2]], dtype=np.int32),
            np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        )
        mesh2.transformation = create_translation_3d(10, 0, 0, np.float32)

        result_faces, result_points = tf.concatenated([mesh1, mesh2])

        # mesh1 points unchanged
        np.testing.assert_array_almost_equal(result_points[0], [0, 0, 0])
        np.testing.assert_array_almost_equal(result_points[1], [1, 0, 0])
        np.testing.assert_array_almost_equal(result_points[2], [0, 1, 0])

        # mesh2 points translated by (10, 0, 0)
        np.testing.assert_array_almost_equal(result_points[3], [10, 0, 0])
        np.testing.assert_array_almost_equal(result_points[4], [11, 0, 0])
        np.testing.assert_array_almost_equal(result_points[5], [10, 1, 0])

    def test_mesh_with_rotation(self):
        """Test concatenating mesh with rotation transformation."""
        # Mesh with point at (1, 0, 0)
        mesh1 = tf.Mesh(
            np.array([[0, 1, 2]], dtype=np.int32),
            np.array([[0, 0, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32)
        )
        # Mesh rotated 90° around Z - point at (1, 0, 0) should become (0, 1, 0)
        mesh2 = tf.Mesh(
            np.array([[0, 1, 2]], dtype=np.int32),
            np.array([[0, 0, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32)
        )
        mesh2.transformation = create_rotation_z_3d(90, np.float32)

        result_faces, result_points = tf.concatenated([mesh1, mesh2])

        # mesh1 points unchanged
        np.testing.assert_array_almost_equal(result_points[1], [1, 0, 0])

        # mesh2 point at (1, 0, 0) rotated 90° around Z -> (0, 1, 0)
        np.testing.assert_array_almost_equal(result_points[4], [0, 1, 0], decimal=5)

    def test_multiple_meshes_with_transformations(self):
        """Test concatenating multiple meshes with different transformations."""
        base_faces = np.array([[0, 1, 2]], dtype=np.int32)
        base_points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)

        mesh1 = tf.Mesh(base_faces.copy(), base_points.copy())
        # No transformation

        mesh2 = tf.Mesh(base_faces.copy(), base_points.copy())
        mesh2.transformation = create_translation_3d(5, 0, 0, np.float32)

        mesh3 = tf.Mesh(base_faces.copy(), base_points.copy())
        mesh3.transformation = create_translation_3d(10, 0, 0, np.float32)

        result_faces, result_points = tf.concatenated([mesh1, mesh2, mesh3])

        # Verify correct number of faces and points
        assert result_faces.shape == (3, 3)
        assert result_points.shape == (9, 3)

        # Verify transformations applied
        # mesh1: origin
        np.testing.assert_array_almost_equal(result_points[0], [0, 0, 0])
        # mesh2: translated by (5, 0, 0)
        np.testing.assert_array_almost_equal(result_points[3], [5, 0, 0])
        # mesh3: translated by (10, 0, 0)
        np.testing.assert_array_almost_equal(result_points[6], [10, 0, 0])

    def test_mesh_no_transformation_returns_original(self):
        """Test that mesh without transformation returns unchanged points."""
        mesh = tf.Mesh(
            np.array([[0, 1, 2]], dtype=np.int32),
            np.array([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5], [7.5, 8.5, 9.5]], dtype=np.float32)
        )
        # No transformation set

        result_faces, result_points = tf.concatenated([mesh])

        np.testing.assert_array_equal(result_points, mesh.points)

    def test_edgemesh_with_translation(self):
        """Test concatenating edge meshes with translation transformation."""
        # EdgeMesh at origin
        em1 = tf.EdgeMesh(
            np.array([[0, 1]], dtype=np.int32),
            np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
        )
        # EdgeMesh with translation
        em2 = tf.EdgeMesh(
            np.array([[0, 1]], dtype=np.int32),
            np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
        )
        em2.transformation = create_translation_3d(0, 5, 0, np.float32)

        result_edges, result_points = tf.concatenated([em1, em2])

        # em1 points unchanged
        np.testing.assert_array_almost_equal(result_points[0], [0, 0, 0])
        np.testing.assert_array_almost_equal(result_points[1], [1, 0, 0])

        # em2 points translated by (0, 5, 0)
        np.testing.assert_array_almost_equal(result_points[2], [0, 5, 0])
        np.testing.assert_array_almost_equal(result_points[3], [1, 5, 0])

    def test_2d_mesh_with_translation(self):
        """Test concatenating 2D meshes with translation transformation."""
        mesh1 = tf.Mesh(
            np.array([[0, 1, 2]], dtype=np.int32),
            np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32)
        )
        mesh2 = tf.Mesh(
            np.array([[0, 1, 2]], dtype=np.int32),
            np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32)
        )
        mesh2.transformation = create_translation_2d(3, 4, np.float32)

        result_faces, result_points = tf.concatenated([mesh1, mesh2])

        # mesh1 points unchanged
        np.testing.assert_array_almost_equal(result_points[0], [0, 0])

        # mesh2 points translated by (3, 4)
        np.testing.assert_array_almost_equal(result_points[3], [3, 4])
        np.testing.assert_array_almost_equal(result_points[4], [4, 4])
        np.testing.assert_array_almost_equal(result_points[5], [3, 5])

    def test_dynamic_mesh_with_transformation(self):
        """Test concatenating dynamic mesh with transformation."""
        # Fixed mesh
        mesh1 = tf.Mesh(
            np.array([[0, 1, 2]], dtype=np.int32),
            np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        )

        # Dynamic mesh with transformation
        offsets = np.array([0, 3, 7], dtype=np.int32)
        data = np.array([0, 1, 2, 0, 2, 3, 4], dtype=np.int32)
        mesh2 = tf.Mesh(
            tf.OffsetBlockedArray(offsets, data),
            np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.5, 0.5, 0]], dtype=np.float32)
        )
        mesh2.transformation = create_translation_3d(10, 10, 10, np.float32)

        result_faces, result_points = tf.concatenated([mesh1, mesh2])

        # Result should be dynamic
        assert isinstance(result_faces, tf.OffsetBlockedArray)

        # mesh1 points unchanged
        np.testing.assert_array_almost_equal(result_points[0], [0, 0, 0])

        # mesh2 points translated by (10, 10, 10)
        np.testing.assert_array_almost_equal(result_points[3], [10, 10, 10])
        np.testing.assert_array_almost_equal(result_points[7], [10.5, 10.5, 10])

    def test_transformation_preserves_dtype(self):
        """Test that transformation preserves point dtype."""
        for dtype in [np.float32, np.float64]:
            mesh1 = tf.Mesh(
                np.array([[0, 1, 2]], dtype=np.int32),
                np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=dtype)
            )
            mesh2 = tf.Mesh(
                np.array([[0, 1, 2]], dtype=np.int32),
                np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=dtype)
            )
            mesh2.transformation = create_translation_3d(5, 0, 0, dtype)

            _, result_points = tf.concatenated([mesh1, mesh2])

            assert result_points.dtype == dtype

    def test_transformation_with_referential_integrity(self):
        """Test that face indices correctly reference transformed points."""
        mesh1 = tf.Mesh(
            np.array([[0, 1, 2]], dtype=np.int32),
            np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0]], dtype=np.float32)
        )
        mesh2 = tf.Mesh(
            np.array([[0, 1, 2]], dtype=np.int32),
            np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0]], dtype=np.float32)
        )
        mesh2.transformation = create_translation_3d(100, 100, 100, np.float32)

        result_faces, result_points = tf.concatenated([mesh1, mesh2])

        # Verify first face references mesh1 points (untransformed)
        face0_pts = result_points[result_faces[0]]
        np.testing.assert_array_almost_equal(face0_pts[0], [0, 0, 0])
        np.testing.assert_array_almost_equal(face0_pts[1], [10, 0, 0])
        np.testing.assert_array_almost_equal(face0_pts[2], [0, 10, 0])

        # Verify second face references mesh2 points (transformed)
        face1_pts = result_points[result_faces[1]]
        np.testing.assert_array_almost_equal(face1_pts[0], [100, 100, 100])
        np.testing.assert_array_almost_equal(face1_pts[1], [110, 100, 100])
        np.testing.assert_array_almost_equal(face1_pts[2], [100, 110, 100])


# ==============================================================================
# Main runner
# ==============================================================================

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
