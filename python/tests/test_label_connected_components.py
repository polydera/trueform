"""
Test label_connected_components functionality

Copyright (c) 2025 Žiga Sajovic, XLAB
"""

import sys

import pytest
import numpy as np
import trueform as tf

# Parameter sets
INDEX_DTYPES = [np.int32, np.int64]

# ==============================================================================
# Test data generators
# ==============================================================================

def create_single_component_ndarray(dtype):
    """Create fixed-width connectivity with all nodes in one component."""
    # 4 nodes forming a linear chain: 0-1-2-3
    connectivity = np.array([
        [1, -1, -1],  # Node 0 connects to node 1
        [0, 2, -1],   # Node 1 connects to nodes 0 and 2
        [1, 3, -1],   # Node 2 connects to nodes 1 and 3
        [2, -1, -1],  # Node 3 connects to node 2
    ], dtype=dtype)
    return connectivity


def create_single_component_oba(dtype):
    """Create OffsetBlockedArray with all nodes in one component."""
    # 4 nodes forming a linear chain: 0-1-2-3
    # Node 0: [1]
    # Node 1: [0, 2]
    # Node 2: [1, 3]
    # Node 3: [2]
    offsets = np.array([0, 1, 3, 5, 6], dtype=dtype)
    data = np.array([1, 0, 2, 1, 3, 2], dtype=dtype)
    return tf.OffsetBlockedArray(offsets, data)


def create_two_components_ndarray(dtype):
    """Create fixed-width connectivity with two separate components."""
    # 6 nodes: component 0 = {0,1,2}, component 1 = {3,4,5}
    connectivity = np.array([
        [1, -1, -1],  # Node 0 connects to node 1 (component 0)
        [0, 2, -1],   # Node 1 connects to nodes 0,2 (component 0)
        [1, -1, -1],  # Node 2 connects to node 1 (component 0)
        [4, -1, -1],  # Node 3 connects to node 4 (component 1)
        [3, 5, -1],   # Node 4 connects to nodes 3,5 (component 1)
        [4, -1, -1],  # Node 5 connects to node 4 (component 1)
    ], dtype=dtype)
    return connectivity


def create_two_components_oba(dtype):
    """Create OffsetBlockedArray with two separate components."""
    # 6 nodes: component 0 = {0,1,2}, component 1 = {3,4,5}
    # Node 0: [1]
    # Node 1: [0, 2]
    # Node 2: [1]
    # Node 3: [4]
    # Node 4: [3, 5]
    # Node 5: [4]
    offsets = np.array([0, 1, 3, 4, 5, 7, 8], dtype=dtype)
    data = np.array([1, 0, 2, 1, 4, 3, 5, 4], dtype=dtype)
    return tf.OffsetBlockedArray(offsets, data)


def create_many_components_ndarray(dtype):
    """Create connectivity where each node is its own component."""
    # 5 isolated nodes
    connectivity = np.array([
        [-1, -1],
        [-1, -1],
        [-1, -1],
        [-1, -1],
        [-1, -1],
    ], dtype=dtype)
    return connectivity


def create_many_components_oba(dtype):
    """Create OffsetBlockedArray where each node is its own component."""
    # 5 isolated nodes (empty neighbor lists)
    # All nodes have empty connectivity
    offsets = np.array([0, 0, 0, 0, 0, 0], dtype=dtype)
    data = np.array([], dtype=dtype)
    return tf.OffsetBlockedArray(offsets, data)


def create_fully_connected_ndarray(dtype):
    """Create fully connected graph."""
    # 4 nodes all connected to each other
    connectivity = np.array([
        [1, 2, 3],
        [0, 2, 3],
        [0, 1, 3],
        [0, 1, 2],
    ], dtype=dtype)
    return connectivity


def create_fully_connected_oba(dtype):
    """Create fully connected OffsetBlockedArray."""
    # 4 nodes all connected to each other
    # Node 0: [1, 2, 3]
    # Node 1: [0, 2, 3]
    # Node 2: [0, 1, 3]
    # Node 3: [0, 1, 2]
    offsets = np.array([0, 3, 6, 9, 12], dtype=dtype)
    data = np.array([1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2], dtype=dtype)
    return tf.OffsetBlockedArray(offsets, data)


# ==============================================================================
# ndarray Tests - Basic Functionality
# ==============================================================================

@pytest.mark.parametrize("dtype", INDEX_DTYPES)
def test_label_connected_components_ndarray_single_component(dtype):
    """Test ndarray with single connected component."""
    connectivity = create_single_component_ndarray(dtype)

    num_components, labels = tf.label_connected_components(connectivity)

    # Verify return types
    assert isinstance(num_components, int), "num_components should be int"
    assert isinstance(labels, np.ndarray), "labels should be numpy array"
    assert labels.dtype == np.int32, f"labels should be int32, got {labels.dtype}"
    assert labels.shape == (len(connectivity),), \
        f"labels shape should be ({len(connectivity)},), got {labels.shape}"

    # Should have exactly 1 component
    assert num_components == 1, f"Expected 1 component, got {num_components}"

    # All nodes should have same label
    assert np.all(labels == labels[0]), "All nodes should have same label for single component"

    # Labels should be 0-indexed
    assert labels[0] == 0, "Component labels should start at 0"


@pytest.mark.parametrize("dtype", INDEX_DTYPES)
def test_label_connected_components_ndarray_two_components(dtype):
    """Test ndarray with exactly two components."""
    connectivity = create_two_components_ndarray(dtype)

    num_components, labels = tf.label_connected_components(connectivity)

    # Verify return types
    assert isinstance(num_components, int)
    assert isinstance(labels, np.ndarray)
    assert labels.dtype == np.int32
    assert labels.shape == (len(connectivity),)

    # Should have exactly 2 components
    assert num_components == 2, f"Expected 2 components, got {num_components}"

    # Verify component structure
    unique_labels = np.unique(labels)
    assert len(unique_labels) == 2, f"Expected 2 unique labels, got {len(unique_labels)}"

    # Labels should be 0 and 1
    assert np.array_equal(unique_labels, np.array([0, 1])), "Labels should be [0, 1]"

    # Nodes 0,1,2 should be in one component, nodes 3,4,5 in another
    assert labels[0] == labels[1] == labels[2], "Nodes 0,1,2 should have same label"
    assert labels[3] == labels[4] == labels[5], "Nodes 3,4,5 should have same label"
    assert labels[0] != labels[3], "Components should have different labels"


@pytest.mark.parametrize("dtype", INDEX_DTYPES)
def test_label_connected_components_ndarray_many_components(dtype):
    """Test ndarray where each node is its own component."""
    connectivity = create_many_components_ndarray(dtype)

    num_components, labels = tf.label_connected_components(connectivity)

    # Verify return types
    assert isinstance(num_components, int)
    assert isinstance(labels, np.ndarray)
    assert labels.dtype == np.int32

    # Should have 5 components (one per node)
    assert num_components == 5, f"Expected 5 components, got {num_components}"

    # Each node should have unique label
    unique_labels = np.unique(labels)
    assert len(unique_labels) == 5, f"Expected 5 unique labels, got {len(unique_labels)}"

    # Labels should be consecutive from 0
    assert np.array_equal(unique_labels, np.arange(5)), "Labels should be [0,1,2,3,4]"


@pytest.mark.parametrize("dtype", INDEX_DTYPES)
def test_label_connected_components_ndarray_fully_connected(dtype):
    """Test ndarray with fully connected graph."""
    connectivity = create_fully_connected_ndarray(dtype)

    num_components, labels = tf.label_connected_components(connectivity)

    # Verify return types
    assert isinstance(num_components, int)
    assert isinstance(labels, np.ndarray)
    assert labels.dtype == np.int32

    # Should have 1 component
    assert num_components == 1, f"Expected 1 component, got {num_components}"

    # All nodes should have same label
    assert np.all(labels == labels[0]), "All nodes should have same label"


# ==============================================================================
# OffsetBlockedArray Tests - Basic Functionality
# ==============================================================================

@pytest.mark.parametrize("dtype", INDEX_DTYPES)
def test_label_connected_components_oba_single_component(dtype):
    """Test OffsetBlockedArray with single connected component."""
    connectivity = create_single_component_oba(dtype)

    num_components, labels = tf.label_connected_components(connectivity)

    # Verify return types
    assert isinstance(num_components, int)
    assert isinstance(labels, np.ndarray)
    assert labels.dtype == np.int32
    assert labels.shape == (len(connectivity),)

    # Should have exactly 1 component
    assert num_components == 1

    # All nodes should have same label
    assert np.all(labels == labels[0])
    assert labels[0] == 0


@pytest.mark.parametrize("dtype", INDEX_DTYPES)
def test_label_connected_components_oba_two_components(dtype):
    """Test OffsetBlockedArray with exactly two components."""
    connectivity = create_two_components_oba(dtype)

    num_components, labels = tf.label_connected_components(connectivity)

    # Verify return types
    assert isinstance(num_components, int)
    assert isinstance(labels, np.ndarray)
    assert labels.dtype == np.int32

    # Should have exactly 2 components
    assert num_components == 2

    # Verify component structure
    unique_labels = np.unique(labels)
    assert len(unique_labels) == 2
    assert np.array_equal(unique_labels, np.array([0, 1]))

    # Nodes 0,1,2 in one component, 3,4,5 in another
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] == labels[4] == labels[5]
    assert labels[0] != labels[3]


@pytest.mark.parametrize("dtype", INDEX_DTYPES)
def test_label_connected_components_oba_many_components(dtype):
    """Test OffsetBlockedArray where each node is its own component."""
    connectivity = create_many_components_oba(dtype)

    num_components, labels = tf.label_connected_components(connectivity)

    # Verify return types
    assert isinstance(num_components, int)
    assert isinstance(labels, np.ndarray)
    assert labels.dtype == np.int32

    # Should have 5 components
    assert num_components == 5

    # Each node should have unique label
    unique_labels = np.unique(labels)
    assert len(unique_labels) == 5
    assert np.array_equal(unique_labels, np.arange(5))


@pytest.mark.parametrize("dtype", INDEX_DTYPES)
def test_label_connected_components_oba_fully_connected(dtype):
    """Test OffsetBlockedArray with fully connected graph."""
    connectivity = create_fully_connected_oba(dtype)

    num_components, labels = tf.label_connected_components(connectivity)

    # Verify return types
    assert isinstance(num_components, int)
    assert isinstance(labels, np.ndarray)
    assert labels.dtype == np.int32

    # Should have 1 component
    assert num_components == 1

    # All nodes should have same label
    assert np.all(labels == labels[0])


# ==============================================================================
# Consistency Tests - OffsetBlockedArray vs ndarray
# ==============================================================================

@pytest.mark.parametrize("dtype", INDEX_DTYPES)
def test_label_connected_components_oba_ndarray_consistency_single(dtype):
    """Test that OffsetBlockedArray and ndarray give same results for single component."""
    conn_oba = create_single_component_oba(dtype)
    conn_ndarray = create_single_component_ndarray(dtype)

    num_comp_oba, labels_oba = tf.label_connected_components(conn_oba)
    num_comp_ndarray, labels_ndarray = tf.label_connected_components(conn_ndarray)

    # Should give identical results
    assert num_comp_oba == num_comp_ndarray, \
        "OffsetBlockedArray and ndarray should give same num_components"
    assert np.array_equal(labels_oba, labels_ndarray), \
        "OffsetBlockedArray and ndarray should give same labels"


@pytest.mark.parametrize("dtype", INDEX_DTYPES)
def test_label_connected_components_oba_ndarray_consistency_two(dtype):
    """Test that OffsetBlockedArray and ndarray give same results for two components."""
    conn_oba = create_two_components_oba(dtype)
    conn_ndarray = create_two_components_ndarray(dtype)

    num_comp_oba, labels_oba = tf.label_connected_components(conn_oba)
    num_comp_ndarray, labels_ndarray = tf.label_connected_components(conn_ndarray)

    # Should give identical results
    assert num_comp_oba == num_comp_ndarray
    assert np.array_equal(labels_oba, labels_ndarray)


# ==============================================================================
# Optional Parameter Tests
# ==============================================================================

@pytest.mark.parametrize("dtype", INDEX_DTYPES)
def test_label_connected_components_with_expected_hint_small(dtype):
    """Test with expected_number_of_components hint (small value for parallel)."""
    connectivity = create_two_components_ndarray(dtype)

    # Use small hint (should use parallel algorithm)
    num_components, labels = tf.label_connected_components(
        connectivity, expected_number_of_components=2
    )

    # Should still get correct result
    assert num_components == 2
    unique_labels = np.unique(labels)
    assert len(unique_labels) == 2


@pytest.mark.parametrize("dtype", INDEX_DTYPES)
def test_label_connected_components_with_expected_hint_large(dtype):
    """Test with expected_number_of_components hint (large value for sequential)."""
    connectivity = create_many_components_ndarray(dtype)

    # Use large hint (should use sequential algorithm)
    num_components, labels = tf.label_connected_components(
        connectivity, expected_number_of_components=1000
    )

    # Should still get correct result
    assert num_components == 5
    unique_labels = np.unique(labels)
    assert len(unique_labels) == 5


@pytest.mark.parametrize("dtype", INDEX_DTYPES)
def test_label_connected_components_hint_does_not_affect_result(dtype):
    """Test that hint doesn't change result, only performance."""
    connectivity = create_two_components_ndarray(dtype)

    num_comp_none, labels_none = tf.label_connected_components(connectivity)
    num_comp_small, labels_small = tf.label_connected_components(
        connectivity, expected_number_of_components=2
    )
    num_comp_large, labels_large = tf.label_connected_components(
        connectivity, expected_number_of_components=1000
    )

    # All should give identical results
    assert num_comp_none == num_comp_small == num_comp_large
    assert np.array_equal(labels_none, labels_small)
    assert np.array_equal(labels_none, labels_large)


# ==============================================================================
# Edge Cases
# ==============================================================================

@pytest.mark.parametrize("dtype", INDEX_DTYPES)
def test_label_connected_components_single_node_ndarray(dtype):
    """Test ndarray with single node."""
    connectivity = np.array([[-1, -1]], dtype=dtype)

    num_components, labels = tf.label_connected_components(connectivity)

    assert num_components == 1
    assert len(labels) == 1
    assert labels[0] == 0


@pytest.mark.parametrize("dtype", INDEX_DTYPES)
def test_label_connected_components_single_node_oba(dtype):
    """Test OffsetBlockedArray with single node."""
    offsets = np.array([0, 0], dtype=dtype)
    data = np.array([], dtype=dtype)
    connectivity = tf.OffsetBlockedArray(offsets, data)

    num_components, labels = tf.label_connected_components(connectivity)

    assert num_components == 1
    assert len(labels) == 1
    assert labels[0] == 0


@pytest.mark.parametrize("dtype", INDEX_DTYPES)
def test_label_connected_components_empty_oba(dtype):
    """Test OffsetBlockedArray with no nodes."""
    offsets = np.array([0], dtype=dtype)
    data = np.array([], dtype=dtype)
    connectivity = tf.OffsetBlockedArray(offsets, data)

    num_components, labels = tf.label_connected_components(connectivity)

    assert num_components == 0
    assert len(labels) == 0


# ==============================================================================
# Error Validation Tests
# ==============================================================================

def test_label_connected_components_invalid_dtype_ndarray():
    """Test error for invalid dtype in ndarray."""
    connectivity = np.array([[0, 1], [1, 0]], dtype=np.float32)

    with pytest.raises(TypeError, match="connectivity dtype must be int32 or int64"):
        tf.label_connected_components(connectivity)


def test_label_connected_components_invalid_dtype_oba_offsets():
    """Test error for invalid dtype in OffsetBlockedArray offsets."""
    offsets = np.array([0, 1, 2], dtype=np.float32)
    data = np.array([1, 0], dtype=np.int32)

    with pytest.raises(TypeError, match="offsets must be int32 or int64"):
        connectivity = tf.OffsetBlockedArray(offsets, data)
        tf.label_connected_components(connectivity)


def test_label_connected_components_invalid_dtype_oba_data():
    """Test error for invalid dtype in OffsetBlockedArray data."""
    offsets = np.array([0, 1, 2], dtype=np.int32)
    data = np.array([1, 0], dtype=np.float32)

    with pytest.raises(TypeError, match="data must be int32 or int64"):
        connectivity = tf.OffsetBlockedArray(offsets, data)
        tf.label_connected_components(connectivity)


def test_label_connected_components_invalid_ndarray_shape():
    """Test error for invalid ndarray shape (not 2D)."""
    # 1D array
    connectivity = np.array([0, 1, 2], dtype=np.int32)

    with pytest.raises(ValueError, match="connectivity array must be 2D"):
        tf.label_connected_components(connectivity)


def test_label_connected_components_invalid_type():
    """Test error for unsupported input type."""
    invalid_input = "not valid"

    with pytest.raises(TypeError, match="Expected np.ndarray or OffsetBlockedArray"):
        tf.label_connected_components(invalid_input)


def test_label_connected_components_invalid_expected_type():
    """Test error for invalid expected_number_of_components type."""
    connectivity = np.array([[1, -1], [0, -1]], dtype=np.int32)

    with pytest.raises(TypeError, match="expected_number_of_components must be int or None"):
        tf.label_connected_components(connectivity, expected_number_of_components=2.5)


# ==============================================================================
# Main runner
# ==============================================================================

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
