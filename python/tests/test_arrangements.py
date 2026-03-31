"""
Tests for mesh_arrangements and polygon_arrangements.

Verifies that intersection curves, non-manifold edges, and arrangement
outputs are consistent across all intersection/arrangement APIs.

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

import sys
import numpy as np
import pytest
import trueform as tf


INDEX_DTYPES = [np.int32, np.int64]
REAL_DTYPES = [np.float32, np.float64]
MESH_TYPES = ['triangle', 'dynamic']


# ==============================================================================
# Helpers
# ==============================================================================

def prepare_mesh(mesh):
    mesh.build_tree()
    mesh.build_face_membership()
    mesh.build_manifold_edge_link()
    return mesh


def make_mesh(faces, points, mesh_type='triangle'):
    if isinstance(faces, tf.OffsetBlockedArray):
        return tf.Mesh(faces, points)
    if mesh_type == 'dynamic':
        return tf.Mesh(tf.as_offset_blocked(faces), points)
    return tf.Mesh(faces, points)


def make_sphere(radius, center, stacks=40, segments=40, dtype=np.float64,
                index_dtype=np.int32, mesh_type='triangle'):
    faces, points = tf.make_sphere_mesh(
        radius, stacks=stacks, segments=segments,
        dtype=dtype, index_dtype=index_dtype)
    points = points + np.array(center, dtype=dtype)
    mesh = tf.Mesh(faces, points)
    new_faces = tf.ensure_positive_orientation(mesh)
    return prepare_mesh(make_mesh(new_faces, points, mesh_type))


def make_three_spheres(dtype=np.float64, index_dtype=np.int32,
                       mesh_type='triangle'):
    d = 1.0
    centers = [
        [0.0, 0.0, 0.0],
        [d, 0.0, 0.0],
        [d / 2, d * np.sqrt(3) / 2, 0.0],
    ]
    return [make_sphere(1.0, c, dtype=dtype, index_dtype=index_dtype,
                        mesh_type=mesh_type)
            for c in centers]


def merge_meshes(meshes, mesh_type='triangle'):
    all_points = np.vstack([m.points for m in meshes])
    offset = 0
    all_faces_list = []
    for m in meshes:
        f = np.array(m.faces)
        all_faces_list.append(f + offset)
        offset += len(m.points)
    all_faces = np.vstack(all_faces_list).astype(meshes[0].faces.dtype)
    return prepare_mesh(make_mesh(all_faces, all_points, mesh_type))


def curve_stats(paths):
    edge_counts = []
    n_closed = 0
    n_open = 0
    endpoints = set()
    for path in paths:
        ids = np.array(path)
        edge_counts.append(len(ids) - 1)
        if len(ids) >= 2 and ids[0] == ids[-1]:
            n_closed += 1
        else:
            n_open += 1
            if len(ids) >= 2:
                endpoints.add(int(ids[0]))
                endpoints.add(int(ids[-1]))
    return len(paths), sorted(edge_counts), n_closed, n_open, len(endpoints)


def get_num_faces(faces):
    if isinstance(faces, tf.OffsetBlockedArray):
        return len(faces)
    return faces.shape[0]


# ==============================================================================
# mesh_arrangements
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("dtype", REAL_DTYPES)
@pytest.mark.parametrize("mesh_type", MESH_TYPES)
def test_mesh_arrangements_labels(index_dtype, dtype, mesh_type):
    spheres = make_three_spheres(dtype=dtype, index_dtype=index_dtype,
                                 mesh_type=mesh_type)
    (faces, points), tag_labels, face_labels = tf.mesh_arrangements(spheres)

    assert tag_labels.min() >= 0
    assert tag_labels.max() <= 2
    assert set(np.unique(tag_labels)) == {0, 1, 2}

    for tag in range(3):
        mask = tag_labels == tag
        orig = face_labels[mask]
        assert orig.min() >= 0
        assert orig.max() < get_num_faces(spheres[tag].faces)

    total_original = sum(get_num_faces(s.faces) for s in spheres)
    assert get_num_faces(faces) >= total_original


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("dtype", REAL_DTYPES)
@pytest.mark.parametrize("mesh_type", MESH_TYPES)
def test_mesh_arrangements_closed_topology(index_dtype, dtype, mesh_type):
    spheres = make_three_spheres(dtype=dtype, index_dtype=index_dtype,
                                 mesh_type=mesh_type)
    (faces, points), _, _ = tf.mesh_arrangements(spheres)

    arr_mesh = tf.Mesh(faces, points)
    arr_mesh.build_face_membership()
    assert len(tf.boundary_paths(arr_mesh)) == 0


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("dtype", REAL_DTYPES)
@pytest.mark.parametrize("mesh_type", MESH_TYPES)
def test_mesh_arrangements_curves_match_nonmanifold(index_dtype, dtype,
                                                     mesh_type):
    spheres = make_three_spheres(dtype=dtype, index_dtype=index_dtype,
                                 mesh_type=mesh_type)
    (faces, points), _, _, (paths, curve_pts) = \
        tf.mesh_arrangements(spheres, return_curves=True)

    assert len(paths) > 0
    assert len(curve_pts) > 0

    arr_mesh = tf.Mesh(faces, points)
    arr_mesh.build_face_membership()
    nm_edges = tf.non_manifold_edges(arr_mesh)
    nm_paths = tf.connect_edges_to_paths(nm_edges)

    c_n, c_ec, c_nc, c_no, c_nep = curve_stats(paths)
    nm_n, nm_ec, nm_nc, nm_no, nm_nep = curve_stats(nm_paths)

    assert c_n == nm_n
    assert c_ec == nm_ec
    assert c_nc == nm_nc
    assert c_no == nm_no
    assert c_nep == nm_nep

    # expected values for 3 overlapping spheres
    assert nm_n == 6, f"Expected 6 non-manifold paths, got {nm_n}"
    assert nm_nep == 2, f"Expected 2 non-manifold endpoints, got {nm_nep}"


# ==============================================================================
# polygon_arrangements
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("dtype", REAL_DTYPES)
@pytest.mark.parametrize("mesh_type", MESH_TYPES)
def test_polygon_arrangements_basic(index_dtype, dtype, mesh_type):
    spheres = make_three_spheres(dtype=dtype, index_dtype=index_dtype,
                                 mesh_type=mesh_type)
    merged = merge_meshes(spheres, mesh_type=mesh_type)

    (faces, points), face_labels = tf.polygon_arrangements(merged)
    assert get_num_faces(faces) >= get_num_faces(merged.faces)
    assert face_labels.min() >= 0
    assert face_labels.max() < get_num_faces(merged.faces)


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("dtype", REAL_DTYPES)
@pytest.mark.parametrize("mesh_type", MESH_TYPES)
def test_polygon_arrangements_curves_match_nonmanifold(index_dtype, dtype,
                                                        mesh_type):
    spheres = make_three_spheres(dtype=dtype, index_dtype=index_dtype,
                                 mesh_type=mesh_type)
    merged = merge_meshes(spheres, mesh_type=mesh_type)

    (faces, points), _, (paths, curve_pts) = \
        tf.polygon_arrangements(merged, return_curves=True)

    arr_mesh = tf.Mesh(faces, points)
    arr_mesh.build_face_membership()
    nm_edges = tf.non_manifold_edges(arr_mesh)
    nm_paths = tf.connect_edges_to_paths(nm_edges)

    c_n, c_ec, c_nc, c_no, c_nep = curve_stats(paths)
    nm_n, nm_ec, nm_nc, nm_no, nm_nep = curve_stats(nm_paths)

    assert c_n == nm_n
    assert c_ec == nm_ec
    assert c_nep == nm_nep

    assert nm_n == 6, f"Expected 6 non-manifold paths, got {nm_n}"
    assert nm_nep == 2, f"Expected 2 non-manifold endpoints, got {nm_nep}"


# ==============================================================================
# Cross-API consistency: all curve sources must agree
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("dtype", REAL_DTYPES)
@pytest.mark.parametrize("mesh_type", MESH_TYPES)
def test_all_sources_same_curve_count(index_dtype, dtype, mesh_type):
    spheres = make_three_spheres(dtype=dtype, index_dtype=index_dtype,
                                 mesh_type=mesh_type)
    merged = merge_meshes(spheres, mesh_type=mesh_type)

    ic_paths, _ = tf.intersection_curves(spheres, mode="primitives")
    si_paths, _ = tf.self_intersection_curves(merged, mode="primitives")
    (_, _), _, _, (ma_paths, _) = \
        tf.mesh_arrangements(spheres, return_curves=True)
    (_, _), _, (pa_paths, _) = \
        tf.polygon_arrangements(merged, return_curves=True)

    counts = [len(ic_paths), len(si_paths), len(ma_paths), len(pa_paths)]
    assert len(set(counts)) == 1, \
        f"Curve counts differ: ic={counts[0]}, si={counts[1]}, ma={counts[2]}, pa={counts[3]}"
    assert counts[0] == 6, f"Expected 6 curves, got {counts[0]}"


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("dtype", REAL_DTYPES)
@pytest.mark.parametrize("mesh_type", MESH_TYPES)
def test_all_sources_same_edge_counts(index_dtype, dtype, mesh_type):
    spheres = make_three_spheres(dtype=dtype, index_dtype=index_dtype,
                                 mesh_type=mesh_type)
    merged = merge_meshes(spheres, mesh_type=mesh_type)

    ic_paths, _ = tf.intersection_curves(spheres, mode="primitives")
    si_paths, _ = tf.self_intersection_curves(merged, mode="primitives")
    (_, _), _, _, (ma_paths, _) = \
        tf.mesh_arrangements(spheres, return_curves=True)
    (_, _), _, (pa_paths, _) = \
        tf.polygon_arrangements(merged, return_curves=True)

    all_ec = [
        curve_stats(p)[1] for p in [ic_paths, si_paths, ma_paths, pa_paths]
    ]
    assert all(ec == all_ec[0] for ec in all_ec), \
        f"Edge counts differ: {all_ec}"


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("dtype", REAL_DTYPES)
@pytest.mark.parametrize("mesh_type", MESH_TYPES)
def test_all_sources_same_endpoints(index_dtype, dtype, mesh_type):
    spheres = make_three_spheres(dtype=dtype, index_dtype=index_dtype,
                                 mesh_type=mesh_type)
    merged = merge_meshes(spheres, mesh_type=mesh_type)

    ic_paths, _ = tf.intersection_curves(spheres, mode="primitives")
    si_paths, _ = tf.self_intersection_curves(merged, mode="primitives")
    (_, _), _, _, (ma_paths, _) = \
        tf.mesh_arrangements(spheres, return_curves=True)
    (_, _), _, (pa_paths, _) = \
        tf.polygon_arrangements(merged, return_curves=True)

    all_stats = [
        curve_stats(p) for p in [ic_paths, si_paths, ma_paths, pa_paths]
    ]

    closed = [s[2] for s in all_stats]
    assert len(set(closed)) == 1, f"Closed counts differ: {closed}"

    opened = [s[3] for s in all_stats]
    assert len(set(opened)) == 1, f"Open counts differ: {opened}"

    endpoints = [s[4] for s in all_stats]
    assert len(set(endpoints)) == 1, f"Endpoint counts differ: {endpoints}"
    assert endpoints[0] == 2, f"Expected 2 endpoints, got {endpoints[0]}"


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("dtype", REAL_DTYPES)
@pytest.mark.parametrize("mesh_type", MESH_TYPES)
def test_all_sources_same_point_count(index_dtype, dtype, mesh_type):
    spheres = make_three_spheres(dtype=dtype, index_dtype=index_dtype,
                                 mesh_type=mesh_type)
    merged = merge_meshes(spheres, mesh_type=mesh_type)

    _, ic_pts = tf.intersection_curves(spheres, mode="primitives")
    _, si_pts = tf.self_intersection_curves(merged, mode="primitives")
    (_, _), _, _, (_, ma_pts) = \
        tf.mesh_arrangements(spheres, return_curves=True)
    (_, _), _, (_, pa_pts) = \
        tf.polygon_arrangements(merged, return_curves=True)

    counts = [len(ic_pts), len(si_pts), len(ma_pts), len(pa_pts)]
    assert len(set(counts)) == 1, \
        f"Point counts differ: ic={counts[0]}, si={counts[1]}, ma={counts[2]}, pa={counts[3]}"


# ==============================================================================

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
