"""
Test connect_edges_to_paths

Copyright (c) 2025 Ziga Sajovic, XLAB
"""

import sys

import pytest
import numpy as np
import trueform as tf

# Parameter sets
INDEX_DTYPES = [np.int32, np.int64]


# ==============================================================================
# Canonicalization helpers
# ==============================================================================

def canonicalize_path(path):
    """
    Canonicalize a path for comparison (handles both open and closed paths).

    For closed paths (first == last), strips duplicate and finds canonical rotation.
    For open paths, considers both forward and reverse directions.
    """
    if len(path) == 0:
        return tuple()

    path = list(path)

    # Check if closed (first == last)
    is_closed = len(path) > 1 and path[0] == path[-1]

    if is_closed:
        # Strip last element for closed path
        path = path[:-1]

    n = len(path)
    if n == 0:
        return tuple()

    # Find minimum element
    min_val = min(path)
    min_positions = [i for i, v in enumerate(path) if v == min_val]

    candidates = []

    if is_closed:
        # For closed paths, try all rotations starting from min positions
        for pos in min_positions:
            # Forward direction
            forward = tuple(path[(pos + i) % n] for i in range(n))
            candidates.append(forward)

            # Reverse direction
            reverse = tuple(path[(pos - i) % n] for i in range(n))
            candidates.append(reverse)
    else:
        # For open paths, try forward and reverse from start
        candidates.append(tuple(path))
        candidates.append(tuple(reversed(path)))

    return min(candidates)


def canonicalize_paths(paths):
    """Canonicalize all paths in an OffsetBlockedArray for comparison."""
    canonical = []
    for path in paths:
        canonical.append(canonicalize_path(path))
    return sorted(canonical)


# ==============================================================================
# Test data generators
# ==============================================================================

def create_simple_line(index_dtype):
    """
    Create a simple line: 0-1-2-3
    Expected: 1 path with 4 vertices
    """
    edges = np.array([
        [0, 1],
        [1, 2],
        [2, 3]
    ], dtype=index_dtype)
    expected_paths = [(0, 1, 2, 3)]
    return edges, expected_paths


def create_closed_loop(index_dtype):
    """
    Create a closed loop: 0-1-2-3-0
    Expected: 1 closed path with 5 vertices (0 appears twice)
    """
    edges = np.array([
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0]
    ], dtype=index_dtype)
    # Closed path, so 0 appears at start and end
    expected_paths = [(0, 1, 2, 3, 0)]
    return edges, expected_paths


def create_two_separate_lines(index_dtype):
    """
    Create two separate lines: 0-1-2 and 3-4-5
    Expected: 2 paths
    """
    edges = np.array([
        [0, 1],
        [1, 2],
        [3, 4],
        [4, 5]
    ], dtype=index_dtype)
    expected_paths = [(0, 1, 2), (3, 4, 5)]
    return edges, expected_paths


def create_unordered_edges(index_dtype):
    """
    Create edges in random order that form a line: 0-1-2-3
    Expected: 1 path with 4 vertices
    """
    edges = np.array([
        [2, 3],
        [0, 1],
        [1, 2]
    ], dtype=index_dtype)
    expected_paths = [(0, 1, 2, 3)]
    return edges, expected_paths


def create_branching_structure(index_dtype):
    r"""
    Create a branching structure::

         2
        /
       1
      / \
     0   3

    Expected: Multiple paths (exact decomposition may vary)
    """
    edges = np.array([
        [0, 1],
        [1, 2],
        [1, 3]
    ], dtype=index_dtype)
    # Branching structures may decompose into multiple paths
    # We just verify all vertices are covered
    return edges, None  # No specific expected paths


def create_single_edge(index_dtype):
    """
    Create a single edge: 0-1
    Expected: 1 path with 2 vertices
    """
    edges = np.array([
        [0, 1]
    ], dtype=index_dtype)
    expected_paths = [(0, 1)]
    return edges, expected_paths


def create_triangle(index_dtype):
    """
    Create a triangle loop: 0-1-2-0
    Expected: 1 closed path
    """
    edges = np.array([
        [0, 1],
        [1, 2],
        [2, 0]
    ], dtype=index_dtype)
    expected_paths = [(0, 1, 2, 0)]
    return edges, expected_paths


def create_figure_eight(index_dtype):
    r"""
    Create a figure-eight: two loops sharing a vertex::

       1     3
      / \   / \
     0   2-4   5

    Expected: Paths covering all edges
    """
    edges = np.array([
        [0, 1],
        [1, 2],
        [2, 0],
        [2, 3],
        [3, 4],
        [4, 2]
    ], dtype=index_dtype)
    # Complex topology - just verify structure
    return edges, None


# ==============================================================================
# connect_edges_to_paths Tests
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_simple_line(index_dtype):
    """Test connecting edges forming a simple line."""
    edges, expected = create_simple_line(index_dtype)

    paths = tf.connect_edges_to_paths(edges)

    # Check return type
    assert isinstance(paths, tf.OffsetBlockedArray)
    assert len(paths) == 1

    # Check path content
    result = canonicalize_paths(paths)
    expected_canonical = sorted([canonicalize_path(p) for p in expected])
    assert result == expected_canonical


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_closed_loop(index_dtype):
    """Test connecting edges forming a closed loop."""
    edges, expected = create_closed_loop(index_dtype)

    paths = tf.connect_edges_to_paths(edges)

    assert len(paths) == 1

    # Verify it's a closed path (first == last)
    path = paths[0]
    assert path[0] == path[-1]

    # Check canonical form
    result = canonicalize_paths(paths)
    expected_canonical = sorted([canonicalize_path(p) for p in expected])
    assert result == expected_canonical


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_two_separate_lines(index_dtype):
    """Test two disconnected lines."""
    edges, expected = create_two_separate_lines(index_dtype)

    paths = tf.connect_edges_to_paths(edges)

    assert len(paths) == 2

    # Check paths
    result = canonicalize_paths(paths)
    expected_canonical = sorted([canonicalize_path(p) for p in expected])
    assert result == expected_canonical


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_unordered_edges(index_dtype):
    """Test that edge order doesn't matter."""
    edges, expected = create_unordered_edges(index_dtype)

    paths = tf.connect_edges_to_paths(edges)

    assert len(paths) == 1

    result = canonicalize_paths(paths)
    expected_canonical = sorted([canonicalize_path(p) for p in expected])
    assert result == expected_canonical


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_single_edge(index_dtype):
    """Test a single edge."""
    edges, expected = create_single_edge(index_dtype)

    paths = tf.connect_edges_to_paths(edges)

    assert len(paths) == 1
    assert len(paths[0]) == 2

    result = canonicalize_paths(paths)
    expected_canonical = sorted([canonicalize_path(p) for p in expected])
    assert result == expected_canonical


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_triangle(index_dtype):
    """Test a triangle (closed loop)."""
    edges, expected = create_triangle(index_dtype)

    paths = tf.connect_edges_to_paths(edges)

    assert len(paths) == 1

    # Should be closed (first == last)
    path = paths[0]
    assert path[0] == path[-1]

    result = canonicalize_paths(paths)
    expected_canonical = sorted([canonicalize_path(p) for p in expected])
    assert result == expected_canonical


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_empty_input(index_dtype):
    """Test with empty edge array."""
    edges = np.array([], dtype=index_dtype).reshape(0, 2)

    paths = tf.connect_edges_to_paths(edges)

    assert isinstance(paths, tf.OffsetBlockedArray)
    assert len(paths) == 0


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_branching_structure(index_dtype):
    """Test branching structure (multiple paths)."""
    edges, _ = create_branching_structure(index_dtype)

    paths = tf.connect_edges_to_paths(edges)

    # Should have multiple paths
    assert len(paths) > 0

    # Verify all edges are covered in some path
    all_vertices = set()
    for path in paths:
        all_vertices.update(path)

    # Should cover vertices 0, 1, 2, 3
    assert all_vertices == {0, 1, 2, 3}


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_figure_eight(index_dtype):
    """Test figure-eight structure (complex topology)."""
    edges, _ = create_figure_eight(index_dtype)

    paths = tf.connect_edges_to_paths(edges)

    # Should produce some paths
    assert len(paths) > 0

    # Verify all vertices are covered
    all_vertices = set()
    for path in paths:
        all_vertices.update(path)

    # Should cover vertices 0-4
    assert all_vertices == {0, 1, 2, 3, 4}


# ==============================================================================
# Error Validation Tests
# ==============================================================================

def test_invalid_input_type():
    """Test with invalid input type."""
    with pytest.raises(TypeError, match="edges must be np.ndarray"):
        tf.connect_edges_to_paths("not an array")


def test_invalid_shape():
    """Test with wrong array shape."""
    edges = np.array([0, 1, 2], dtype=np.int32)  # 1D array
    with pytest.raises(ValueError, match="2D array"):
        tf.connect_edges_to_paths(edges)


def test_invalid_edge_width():
    """Test with wrong number of columns."""
    edges = np.array([[0, 1, 2]], dtype=np.int32)  # 3 columns instead of 2
    with pytest.raises(ValueError, match="2 columns"):
        tf.connect_edges_to_paths(edges)


def test_invalid_dtype():
    """Test with unsupported dtype."""
    edges = np.array([[0, 1], [1, 2]], dtype=np.float32)
    with pytest.raises(TypeError, match="int32 or int64"):
        tf.connect_edges_to_paths(edges)


# ==============================================================================
# Return Type Tests
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_preserves_dtype(index_dtype):
    """Test that output preserves input dtype."""
    edges = np.array([[0, 1], [1, 2]], dtype=index_dtype)

    paths = tf.connect_edges_to_paths(edges)

    # Check that path data has correct dtype
    assert paths.data.dtype == index_dtype
    assert paths.offsets.dtype == index_dtype


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_offset_blocked_array_structure(index_dtype):
    """Test that result is valid OffsetBlockedArray."""
    edges = np.array([[0, 1], [1, 2], [3, 4]], dtype=index_dtype)

    paths = tf.connect_edges_to_paths(edges)

    # Verify OffsetBlockedArray structure
    assert hasattr(paths, 'offsets')
    assert hasattr(paths, 'data')
    assert paths.offsets[0] == 0
    assert paths.offsets[-1] == len(paths.data)

    # Verify offsets are non-decreasing
    for i in range(len(paths.offsets) - 1):
        assert paths.offsets[i] <= paths.offsets[i + 1]


# ==============================================================================
# Main runner
# ==============================================================================

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
