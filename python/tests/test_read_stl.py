"""
Test read_stl functionality

Copyright (c) 2025 Žiga Sajovic, XLAB
"""

import sys
import os

# Add parent directory to path so we can import trueform
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import trueform as tf
import tempfile


def create_simple_stl(filename):
    """Create a simple ASCII STL file with a single triangle"""
    stl_content = """solid test
  facet normal 0 0 1
    outer loop
      vertex 0 0 0
      vertex 1 0 0
      vertex 0 1 0
    endloop
  endfacet
endsolid test
"""
    with open(filename, 'w') as f:
        f.write(stl_content)


def create_cube_stl(filename):
    """Create a simple ASCII STL file with a cube (12 triangles)"""
    stl_content = """solid cube
  facet normal 0 0 -1
    outer loop
      vertex 0 0 0
      vertex 1 0 0
      vertex 1 1 0
    endloop
  endfacet
  facet normal 0 0 -1
    outer loop
      vertex 0 0 0
      vertex 1 1 0
      vertex 0 1 0
    endloop
  endfacet
  facet normal 0 0 1
    outer loop
      vertex 0 0 1
      vertex 1 1 1
      vertex 1 0 1
    endloop
  endfacet
  facet normal 0 0 1
    outer loop
      vertex 0 0 1
      vertex 0 1 1
      vertex 1 1 1
    endloop
  endfacet
  facet normal 0 -1 0
    outer loop
      vertex 0 0 0
      vertex 1 0 1
      vertex 1 0 0
    endloop
  endfacet
  facet normal 0 -1 0
    outer loop
      vertex 0 0 0
      vertex 0 0 1
      vertex 1 0 1
    endloop
  endfacet
  facet normal 0 1 0
    outer loop
      vertex 0 1 0
      vertex 1 1 0
      vertex 1 1 1
    endloop
  endfacet
  facet normal 0 1 0
    outer loop
      vertex 0 1 0
      vertex 1 1 1
      vertex 0 1 1
    endloop
  endfacet
  facet normal -1 0 0
    outer loop
      vertex 0 0 0
      vertex 0 1 0
      vertex 0 1 1
    endloop
  endfacet
  facet normal -1 0 0
    outer loop
      vertex 0 0 0
      vertex 0 1 1
      vertex 0 0 1
    endloop
  endfacet
  facet normal 1 0 0
    outer loop
      vertex 1 0 0
      vertex 1 1 1
      vertex 1 1 0
    endloop
  endfacet
  facet normal 1 0 0
    outer loop
      vertex 1 0 0
      vertex 1 0 1
      vertex 1 1 1
    endloop
  endfacet
endsolid cube
"""
    with open(filename, 'w') as f:
        f.write(stl_content)


def test_read_stl_simple():
    """Test reading a simple STL with one triangle"""
    print("\n=== Test: Read Simple STL ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        stl_file = os.path.join(tmpdir, "test.stl")
        create_simple_stl(stl_file)

        faces, points = tf.read_stl(stl_file)

        print(f"  Faces shape: {faces.shape}, dtype: {faces.dtype}")
        print(f"  Points shape: {points.shape}, dtype: {points.dtype}")
        print(f"  Faces:\n{faces}")
        print(f"  Points:\n{points}")

        # Check shapes
        assert faces.shape[1] == 3, "Faces should have 3 columns"
        assert points.shape[1] == 3, "Points should have 3 columns"

        # Check dtypes
        assert faces.dtype == np.int32, "Faces should be int32"
        assert points.dtype == np.float32, "Points should be float32"

        # Check that we have at least one face
        assert faces.shape[0] >= 1, "Should have at least one face"

        # Check that face indices are valid
        assert np.all(faces >= 0), "Face indices should be non-negative"
        assert np.all(faces < points.shape[0]), "Face indices should be within points range"

        # Check that points are 3D
        assert points.shape[0] >= 3, "Should have at least 3 points"

        print("✓ Simple STL test passed")


def test_read_stl_cube():
    """Test reading a cube STL"""
    print("\n=== Test: Read Cube STL ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        stl_file = os.path.join(tmpdir, "cube.stl")
        create_cube_stl(stl_file)

        faces, points = tf.read_stl(stl_file)

        print(f"  Faces shape: {faces.shape}, dtype: {faces.dtype}")
        print(f"  Points shape: {points.shape}, dtype: {points.dtype}")
        print(f"  Number of faces: {faces.shape[0]}")
        print(f"  Number of points: {points.shape[0]}")

        # Cube has 12 triangles (2 per face, 6 faces)
        assert faces.shape[0] == 12, f"Cube should have 12 faces, got {faces.shape[0]}"

        # Cube has 8 unique vertices (after cleaning duplicates)
        assert points.shape[0] == 8, f"Cube should have 8 points, got {points.shape[0]}"

        # Check face indices are valid
        assert np.all(faces >= 0), "Face indices should be non-negative"
        assert np.all(faces < 8), "Face indices should be < 8 for a cube"

        # Check that points are within expected range [0, 1]
        assert np.all(points >= 0.0), "Points should be >= 0"
        assert np.all(points <= 1.0), "Points should be <= 1"

        print("✓ Cube STL test passed")


def test_read_stl_vertex_deduplication():
    """Test that duplicate vertices are merged"""
    print("\n=== Test: Vertex Deduplication ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        stl_file = os.path.join(tmpdir, "test.stl")
        # Create STL with duplicate vertices
        stl_content = """solid test
  facet normal 0 0 1
    outer loop
      vertex 0 0 0
      vertex 1 0 0
      vertex 0 1 0
    endloop
  endfacet
  facet normal 0 0 1
    outer loop
      vertex 1 0 0
      vertex 1 1 0
      vertex 0 1 0
    endloop
  endfacet
endsolid test
"""
        with open(stl_file, 'w') as f:
            f.write(stl_content)

        faces, points = tf.read_stl(stl_file)

        print(f"  Faces shape: {faces.shape}")
        print(f"  Points shape: {points.shape}")
        print(f"  Faces:\n{faces}")
        print(f"  Points:\n{points}")

        # Two triangles sharing vertices should result in 4 unique points
        # (0,0,0), (1,0,0), (0,1,0), (1,1,0)
        assert points.shape[0] == 4, f"Should have 4 unique points, got {points.shape[0]}"
        assert faces.shape[0] == 2, f"Should have 2 faces, got {faces.shape[0]}"

        # Check that faces reference the deduplicated points
        # Face 0 should be [0, 1, 2]
        # Face 1 should be [1, 3, 2] (shared vertices 1 and 2)
        shared_vertices = 0
        for i in range(3):
            if faces[0, i] in faces[1]:
                shared_vertices += 1

        assert shared_vertices == 2, f"Faces should share 2 vertices, found {shared_vertices}"

        print("✓ Vertex deduplication test passed")


def test_read_stl_memory_ownership():
    """Test that numpy arrays properly own their memory"""
    print("\n=== Test: Memory Ownership ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        stl_file = os.path.join(tmpdir, "test.stl")
        create_simple_stl(stl_file)

        faces, points = tf.read_stl(stl_file)

        # Check that arrays are writable
        # Note: OWNDATA may not be set with nanobind capsules, but memory is still managed
        assert faces.flags['WRITEABLE'], "Faces array should be writeable"
        assert points.flags['WRITEABLE'], "Points array should be writeable"

        # Test that we can modify the arrays (they're not views)
        original_point = points[0].copy()
        points[0] = [999.0, 999.0, 999.0]
        assert not np.array_equal(points[0], original_point), "Should be able to modify points"

        print("✓ Memory ownership test passed")


if __name__ == "__main__":
    print("Testing read_stl functionality\n")
    print("=" * 60)

    try:
        test_read_stl_simple()
        test_read_stl_cube()
        test_read_stl_vertex_deduplication()
        test_read_stl_memory_ownership()

        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
