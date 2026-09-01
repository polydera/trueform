"""
Tests for the transform arm of every arrangement entry.

Each operand carries a frame: the wrapper's transformation when it has
one, identity otherwise. The oracle for every transform-carrying entry is
the same computation with the transformation pre-applied to the vertex
array in numpy and no wrapper transformation set — the two arms must
agree.

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

import sys
import numpy as np
import pytest
import trueform as tf


REAL_DTYPES = [np.float32, np.float64]
INDEX_DTYPES = [np.int32, np.int64]

BOOLEAN_OPS = [tf.boolean_union, tf.boolean_intersection, tf.boolean_difference]

# The two arms quantize coordinates that differ by one rounding step of
# the real dtype, so positions agree to that scale and never bitwise.
POINT_ATOL = {np.float32: 1e-4, np.float64: 1e-10}
MEASURE_RTOL = {np.float32: 1e-4, np.float64: 1e-9}


# ==============================================================================
# Transformations
# ==============================================================================

def _rigid(axis, angle, translation, dtype):
    """A rotation about a non-axis-aligned axis, plus a translation."""
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    k = np.array([[0.0, -axis[2], axis[1]],
                  [axis[2], 0.0, -axis[0]],
                  [-axis[1], axis[0], 0.0]])
    r = np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)
    t = np.eye(4)
    t[:3, :3] = r
    t[:3, 3] = translation
    return t.astype(dtype)


def transform_a(dtype):
    return _rigid((0.3, -0.5, 0.81), 0.7, (0.37, -0.21, 0.53), dtype)


def transform_b(dtype):
    return _rigid((-0.7, 0.2, 0.65), -1.1, (-0.15, 0.44, -0.26), dtype)


def apply_transformation(points, transformation):
    """
    The oracle arm: the transformation baked into the vertex array.

    Accumulates in the points' own dtype, translation first, in the order
    the native affine apply uses, so the two arms differ by at most one
    rounding step per coordinate.
    """
    dtype = points.dtype
    out = np.empty_like(points)
    for i in range(3):
        acc = np.full(points.shape[0], transformation[i, 3], dtype=dtype)
        for j in range(3):
            acc = acc + points[:, j] * transformation[i, j]
        out[:, i] = acc
    return out


# ==============================================================================
# Meshes
# ==============================================================================

def prepare(mesh):
    mesh.build_tree()
    mesh.build_face_membership()
    mesh.build_manifold_edge_link()
    return mesh


def box(center, size=1.0, real_dtype=np.float32, index_dtype=np.int32):
    c = np.asarray(center, dtype=real_dtype)
    h = size / 2
    points = np.array(
        [[x, y, z] for x in (-h, h) for y in (-h, h) for z in (-h, h)],
        dtype=real_dtype,
    ) + c
    faces = np.array(
        [[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
         [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]],
        dtype=index_dtype,
    )
    return faces, points


def sphere(radius, center, real_dtype=np.float32, index_dtype=np.int32,
           stacks=12, segments=12):
    faces, points = tf.make_sphere_mesh(
        radius, stacks=stacks, segments=segments,
        dtype=real_dtype, index_dtype=index_dtype)
    points = (points + np.array(center, dtype=real_dtype)).astype(real_dtype)
    return faces, points


def spheres_soup(real_dtype=np.float32, index_dtype=np.int32):
    """Two overlapping spheres concatenated into one self-intersecting mesh."""
    f0, p0 = sphere(1.0, (0, 0, 0), real_dtype, index_dtype)
    p1 = (p0 + np.array([1.0, 0.0, 0.0], dtype=real_dtype)).astype(real_dtype)
    faces, points = tf.concatenated([(f0, p0), (f0, p1)])
    return (faces.astype(index_dtype, copy=False),
            points.astype(real_dtype, copy=False))


def tagged(faces, points, transformation):
    """The mesh whose wrapper carries the transformation."""
    mesh = tf.Mesh(faces, points)
    mesh.transformation = transformation
    return mesh


def baked(faces, points, transformation):
    """The oracle mesh: same geometry, transformation already in the array."""
    return tf.Mesh(faces, apply_transformation(points, transformation))


# ==============================================================================
# Comparison
# ==============================================================================

def assert_point_sets_match(got, want, real_dtype):
    """
    Point sets agree as sets, within the dtype's tolerance.

    Position, not slot: a one-rounding-step coordinate difference can
    permute the created-point table, and that permutation is not a
    disagreement about the arrangement.
    """
    a = np.asarray(got, dtype=np.float64)
    b = np.asarray(want, dtype=np.float64)
    assert a.shape == b.shape, f"point counts differ: {a.shape} vs {b.shape}"
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    atol = POINT_ATOL[real_dtype]
    assert d.min(axis=1).max() <= atol
    assert d.min(axis=0).max() <= atol


def surface_area(faces, points):
    f = np.asarray(faces)
    p = np.asarray(points, dtype=np.float64)
    e0 = p[f[:, 1]] - p[f[:, 0]]
    e1 = p[f[:, 2]] - p[f[:, 0]]
    return 0.5 * np.linalg.norm(np.cross(e0, e1), axis=1).sum()


def assert_meshes_match(got, want, real_dtype):
    """Same face count, same surface, same point set."""
    (faces_a, points_a) = got
    (faces_b, points_b) = want
    assert faces_a.shape == faces_b.shape
    assert faces_a.dtype == faces_b.dtype
    assert points_a.dtype == points_b.dtype
    assert surface_area(faces_a, points_a) == pytest.approx(
        surface_area(faces_b, points_b), rel=MEASURE_RTOL[real_dtype])
    assert_point_sets_match(points_a, points_b, real_dtype)


def assert_curves_match(got, want, real_dtype):
    paths_a, points_a = got
    paths_b, points_b = want
    assert len(paths_a) == len(paths_b)
    assert sorted(len(p) for p in paths_a) == sorted(len(p) for p in paths_b)
    assert_point_sets_match(points_a, points_b, real_dtype)


def assert_curves_identical(got, want):
    np.testing.assert_array_equal(np.asarray(got[0].offsets),
                                  np.asarray(want[0].offsets))
    np.testing.assert_array_equal(np.asarray(got[0].data),
                                  np.asarray(want[0].data))
    np.testing.assert_array_equal(got[1], want[1])


# ==============================================================================
# boolean: both operands transformed
# ==============================================================================

@pytest.mark.parametrize("op", BOOLEAN_OPS)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_boolean_both_operands_transformed(op, real_dtype):
    t = transform_a(real_dtype)
    fa, pa = box((0, 0, 0), real_dtype=real_dtype)
    fb, pb = box((0.4, 0.3, 0.2), real_dtype=real_dtype)

    (faces, points), tags, face_labels = op(
        tagged(fa, pa, t), tagged(fb, pb, t))
    (want_faces, want_points), want_tags, want_face_labels = op(
        baked(fa, pa, t), baked(fb, pb, t))

    np.testing.assert_array_equal(faces, want_faces)
    np.testing.assert_array_equal(tags, want_tags)
    np.testing.assert_array_equal(face_labels, want_face_labels)
    np.testing.assert_allclose(points, want_points,
                               atol=POINT_ATOL[real_dtype], rtol=0)
    assert_meshes_match((faces, points), (want_faces, want_points), real_dtype)
    assert tf.volume((faces, points)) == pytest.approx(
        tf.volume((want_faces, want_points)), rel=MEASURE_RTOL[real_dtype])


@pytest.mark.parametrize("op", BOOLEAN_OPS)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_boolean_one_operand_transformed(op, real_dtype):
    t = transform_a(real_dtype)
    fa, pa = box((0, 0, 0), real_dtype=real_dtype)
    fb, pb = box((0.4, 0.3, 0.2), real_dtype=real_dtype)

    (faces, points), tags, face_labels = op(
        tf.Mesh(fa, pa), tagged(fb, pb, t))
    (want_faces, want_points), want_tags, want_face_labels = op(
        tf.Mesh(fa, pa), baked(fb, pb, t))

    np.testing.assert_array_equal(faces, want_faces)
    np.testing.assert_array_equal(tags, want_tags)
    np.testing.assert_array_equal(face_labels, want_face_labels)
    assert_meshes_match((faces, points), (want_faces, want_points), real_dtype)


@pytest.mark.parametrize("op", BOOLEAN_OPS)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_boolean_distinct_transformations(op, real_dtype):
    ta = transform_a(real_dtype)
    tb = transform_b(real_dtype)
    fa, pa = box((0, 0, 0), real_dtype=real_dtype)
    fb, pb = box((0.4, 0.3, 0.2), real_dtype=real_dtype)

    (faces, points), tags, face_labels = op(
        tagged(fa, pa, ta), tagged(fb, pb, tb))
    (want_faces, want_points), want_tags, want_face_labels = op(
        baked(fa, pa, ta), baked(fb, pb, tb))

    np.testing.assert_array_equal(faces, want_faces)
    np.testing.assert_array_equal(tags, want_tags)
    np.testing.assert_array_equal(face_labels, want_face_labels)
    assert_meshes_match((faces, points), (want_faces, want_points), real_dtype)
    # the operands really moved: the union no longer bounds the two boxes
    # in their own coordinates
    assert points.min() < box((0, 0, 0), real_dtype=real_dtype)[1].min()


@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_boolean_difference_both_directions_transformed(real_dtype):
    ta = transform_a(real_dtype)
    tb = transform_b(real_dtype)
    fa, pa = box((0, 0, 0), real_dtype=real_dtype)
    fb, pb = box((0.4, 0.3, 0.2), size=0.8, real_dtype=real_dtype)

    a_minus_b, _, _ = tf.boolean_difference(
        tagged(fa, pa, ta), tagged(fb, pb, tb))
    want_a_minus_b, _, _ = tf.boolean_difference(
        baked(fa, pa, ta), baked(fb, pb, tb))
    b_minus_a, _, _ = tf.boolean_difference(
        tagged(fb, pb, tb), tagged(fa, pa, ta))
    want_b_minus_a, _, _ = tf.boolean_difference(
        baked(fb, pb, tb), baked(fa, pa, ta))

    np.testing.assert_array_equal(a_minus_b[0], want_a_minus_b[0])
    np.testing.assert_array_equal(b_minus_a[0], want_b_minus_a[0])
    assert_meshes_match(a_minus_b, want_a_minus_b, real_dtype)
    assert_meshes_match(b_minus_a, want_b_minus_a, real_dtype)
    # the two directions are genuinely different solids
    assert tf.volume(a_minus_b) != pytest.approx(tf.volume(b_minus_a), rel=1e-3)


@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_boolean_transformed_int64_index(real_dtype):
    t = transform_a(real_dtype)
    fa, pa = box((0, 0, 0), real_dtype=real_dtype, index_dtype=np.int64)
    fb, pb = box((0.4, 0.3, 0.2), real_dtype=real_dtype, index_dtype=np.int64)

    (faces, points), tags, face_labels = tf.boolean_union(
        tagged(fa, pa, t), tagged(fb, pb, t))
    (want_faces, want_points), want_tags, _ = tf.boolean_union(
        baked(fa, pa, t), baked(fb, pb, t))

    assert faces.dtype == np.int64
    np.testing.assert_array_equal(faces, want_faces)
    np.testing.assert_array_equal(tags, want_tags)
    assert_meshes_match((faces, points), (want_faces, want_points), real_dtype)


@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_boolean_transformed_return_curves(real_dtype):
    t = transform_a(real_dtype)
    fa, pa = box((0, 0, 0), real_dtype=real_dtype)
    fb, pb = box((0.4, 0.3, 0.2), real_dtype=real_dtype)

    (faces, points), _, _, curves = tf.boolean_union(
        tf.Mesh(fa, pa), tagged(fb, pb, t), return_curves=True)
    (want_faces, want_points), _, _, want_curves = tf.boolean_union(
        tf.Mesh(fa, pa), baked(fb, pb, t), return_curves=True)

    np.testing.assert_array_equal(faces, want_faces)
    assert_meshes_match((faces, points), (want_faces, want_points), real_dtype)
    assert len(curves[0]) > 0
    assert_curves_match(curves, want_curves, real_dtype)


@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_boolean_transformation_after_prebuilt_structures(real_dtype):
    """The tree is the mesh's own; setting the frame after it is built
    changes nothing about the answer."""
    t = transform_a(real_dtype)
    fa, pa = box((0, 0, 0), real_dtype=real_dtype)
    fb, pb = box((0.4, 0.3, 0.2), real_dtype=real_dtype)

    before_a = prepare(tf.Mesh(fa, pa))
    before_a.transformation = t
    before_b = prepare(tf.Mesh(fb, pb))
    before_b.transformation = t
    (faces, points), _, _ = tf.boolean_intersection(before_a, before_b)

    after_a = tagged(fa, pa, t)
    after_b = tagged(fb, pb, t)
    prepare(after_a)
    prepare(after_b)
    (after_faces, after_points), _, _ = tf.boolean_intersection(
        after_a, after_b)

    np.testing.assert_array_equal(faces, after_faces)
    np.testing.assert_array_equal(points, after_points)

    (want_faces, want_points), _, _ = tf.boolean_intersection(
        prepare(baked(fa, pa, t)), prepare(baked(fb, pb, t)))
    np.testing.assert_array_equal(faces, want_faces)
    assert_meshes_match((faces, points), (want_faces, want_points), real_dtype)


# ==============================================================================
# intersection_curves
# ==============================================================================

@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_intersection_curves_pair_transformed(real_dtype):
    t = transform_a(real_dtype)
    f0, p0 = sphere(1.0, (0, 0, 0), real_dtype)
    f1, p1 = sphere(0.9, (0.8, 0.3, 0.2), real_dtype)

    got = tf.intersection_curves(tf.Mesh(f0, p0), tagged(f1, p1, t))
    want = tf.intersection_curves(tf.Mesh(f0, p0), baked(f1, p1, t))

    assert len(got[0]) > 0
    assert_curves_match(got, want, real_dtype)
    # the seam genuinely moved: the untransformed operands do not produce it
    plain = tf.intersection_curves(tf.Mesh(f0, p0), tf.Mesh(f1, p1))
    assert plain[1].shape != got[1].shape or not np.allclose(plain[1], got[1])


@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_intersection_curves_list_transformed_member(real_dtype):
    t = transform_a(real_dtype)
    f0, p0 = sphere(1.0, (0, 0, 0), real_dtype)
    f1, p1 = sphere(0.9, (0.8, 0.3, 0.2), real_dtype)
    f2, p2 = sphere(0.7, (0.2, 0.9, -0.3), real_dtype)

    got = tf.intersection_curves(
        [tf.Mesh(f0, p0), tagged(f1, p1, t), tf.Mesh(f2, p2)])
    want = tf.intersection_curves(
        [tf.Mesh(f0, p0), baked(f1, p1, t), tf.Mesh(f2, p2)])

    assert len(got[0]) > 1
    assert_curves_match(got, want, real_dtype)


# ==============================================================================
# mesh_arrangements
# ==============================================================================

@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_mesh_arrangements_transformed_member(real_dtype):
    t = transform_a(real_dtype)
    f0, p0 = sphere(1.0, (0, 0, 0), real_dtype)
    f1, p1 = sphere(0.9, (0.8, 0.3, 0.2), real_dtype)
    f2, p2 = sphere(0.7, (0.2, 0.9, -0.3), real_dtype)

    (faces, points), tags, face_labels = tf.mesh_arrangements(
        [prepare(tf.Mesh(f0, p0)), prepare(tagged(f1, p1, t)),
         prepare(tf.Mesh(f2, p2))])
    (want_faces, want_points), want_tags, want_face_labels = \
        tf.mesh_arrangements(
            [prepare(tf.Mesh(f0, p0)), prepare(baked(f1, p1, t)),
             prepare(tf.Mesh(f2, p2))])

    np.testing.assert_array_equal(tags, want_tags)
    np.testing.assert_array_equal(face_labels, want_face_labels)
    assert_meshes_match((faces, points), (want_faces, want_points), real_dtype)


@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_mesh_arrangements_transformed_member_return_curves(real_dtype):
    t = transform_a(real_dtype)
    f0, p0 = sphere(1.0, (0, 0, 0), real_dtype)
    f1, p1 = sphere(0.9, (0.8, 0.3, 0.2), real_dtype)
    f2, p2 = sphere(0.7, (0.2, 0.9, -0.3), real_dtype)

    (faces, points), tags, face_labels, curves = tf.mesh_arrangements(
        [prepare(tf.Mesh(f0, p0)), prepare(tagged(f1, p1, t)),
         prepare(tf.Mesh(f2, p2))], return_curves=True)
    (want_faces, want_points), want_tags, want_face_labels, want_curves = \
        tf.mesh_arrangements(
            [prepare(tf.Mesh(f0, p0)), prepare(baked(f1, p1, t)),
             prepare(tf.Mesh(f2, p2))], return_curves=True)

    np.testing.assert_array_equal(tags, want_tags)
    np.testing.assert_array_equal(face_labels, want_face_labels)
    assert_meshes_match((faces, points), (want_faces, want_points), real_dtype)
    assert len(curves[0]) > 1
    assert_curves_match(curves, want_curves, real_dtype)


# ==============================================================================
# CsgGraph
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_csg_graph_mesh_transformed(real_dtype, index_dtype):
    ta = transform_a(real_dtype)
    tb = transform_b(real_dtype)
    fa, pa = box((0, 0, 0), real_dtype=real_dtype, index_dtype=index_dtype)
    fb, pb = box((0.4, 0.3, 0.2), real_dtype=real_dtype,
                 index_dtype=index_dtype)

    graph = tf.CsgGraph([tagged(fa, pa, ta), tagged(fb, pb, tb)])
    want_graph = tf.CsgGraph([baked(fa, pa, ta), baked(fb, pb, tb)])

    for expr in [tf.op(0) | tf.op(1), tf.op(0) & tf.op(1),
                 tf.op(0) - tf.op(1), tf.op(1) - tf.op(0)]:
        faces, points = graph.mesh(expr)
        want_faces, want_points = want_graph.mesh(expr)
        assert faces.dtype == index_dtype
        assert points.dtype == real_dtype
        np.testing.assert_array_equal(faces, want_faces)
        assert_meshes_match((faces, points), (want_faces, want_points),
                            real_dtype)

    assert_point_sets_match(graph.created_points, want_graph.created_points,
                            real_dtype)


@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_csg_graph_domains_transformed(real_dtype):
    ta = transform_a(real_dtype)
    tb = transform_b(real_dtype)
    fa, pa = box((0, 0, 0), real_dtype=real_dtype)
    fb, pb = box((0.4, 0.3, 0.2), real_dtype=real_dtype)

    graph = tf.CsgGraph([tagged(fa, pa, ta), tagged(fb, pb, tb)])
    want_graph = tf.CsgGraph([baked(fa, pa, ta), baked(fb, pb, tb)])

    cells, ids = graph.domains()
    want_cells, want_ids = want_graph.domains()

    assert len(cells) == 3
    np.testing.assert_array_equal(ids, want_ids)
    volumes = sorted(abs(tf.volume(cell)) for cell in cells)
    want_volumes = sorted(abs(tf.volume(cell)) for cell in want_cells)
    for v, w in zip(volumes, want_volumes):
        assert v == pytest.approx(w, rel=MEASURE_RTOL[real_dtype])
    for cell, want_cell in zip(cells, want_cells):
        assert_meshes_match(cell, want_cell, real_dtype)


@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_csg_graph_intersection_curves_transformed(real_dtype):
    ta = transform_a(real_dtype)
    tb = transform_b(real_dtype)
    fa, pa = box((0, 0, 0), real_dtype=real_dtype)
    fb, pb = box((0.4, 0.3, 0.2), real_dtype=real_dtype)

    graph = tf.CsgGraph([tagged(fa, pa, ta), tagged(fb, pb, tb)])
    want_graph = tf.CsgGraph([baked(fa, pa, ta), baked(fb, pb, tb)])

    got = graph.intersection_curves()
    want = want_graph.intersection_curves()

    assert len(got[0]) > 0
    assert_curves_match(got, want, real_dtype)


# ==============================================================================
# The single-form entries ignore the wrapper's transformation
#
# Ruled contract: transforms are only used when needed, and with a single
# mesh they are not needed. self_intersection_curves, polygon_arrangements
# and outer_shell run in the mesh's own coordinates.
# ==============================================================================

@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_self_intersection_curves_ignores_transformation(real_dtype):
    # Ruled: with one mesh a transform is not needed, so this entry ignores it.
    t = transform_a(real_dtype)
    faces, points = spheres_soup(real_dtype)

    plain = tf.self_intersection_curves(tf.Mesh(faces, points))
    with_transformation = tf.self_intersection_curves(tagged(faces, points, t))
    assert_curves_identical(with_transformation, plain)

    # the transformation is not trivial: applying it would have moved the
    # curves, so the assertion above is not a tautology
    moved = tf.self_intersection_curves(baked(faces, points, t))
    assert (moved[1].shape != plain[1].shape
            or not np.allclose(moved[1], plain[1]))


@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_polygon_arrangements_ignores_transformation(real_dtype):
    # Ruled: with one mesh a transform is not needed, so this entry ignores it.
    t = transform_a(real_dtype)
    faces, points = spheres_soup(real_dtype)

    (plain_faces, plain_points), plain_labels = tf.polygon_arrangements(
        tf.Mesh(faces, points))
    (got_faces, got_points), got_labels = tf.polygon_arrangements(
        tagged(faces, points, t))

    np.testing.assert_array_equal(got_faces, plain_faces)
    np.testing.assert_array_equal(got_points, plain_points)
    np.testing.assert_array_equal(got_labels, plain_labels)

    (moved_faces, moved_points), _ = tf.polygon_arrangements(
        baked(faces, points, t))
    assert (moved_points.shape != plain_points.shape
            or not np.allclose(moved_points, plain_points))


@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_polygon_arrangements_curves_ignore_transformation(real_dtype):
    # Ruled: with one mesh a transform is not needed, so this entry ignores it.
    t = transform_a(real_dtype)
    faces, points = spheres_soup(real_dtype)

    (plain_faces, plain_points), plain_labels, plain_curves = \
        tf.polygon_arrangements(tf.Mesh(faces, points), return_curves=True)
    (got_faces, got_points), got_labels, got_curves = \
        tf.polygon_arrangements(tagged(faces, points, t), return_curves=True)

    np.testing.assert_array_equal(got_faces, plain_faces)
    np.testing.assert_array_equal(got_points, plain_points)
    np.testing.assert_array_equal(got_labels, plain_labels)
    assert len(plain_curves[0]) > 0
    assert_curves_identical(got_curves, plain_curves)


@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_outer_shell_ignores_transformation(real_dtype):
    # Ruled: with one mesh a transform is not needed, so this entry ignores it.
    t = transform_a(real_dtype)
    faces, points = spheres_soup(real_dtype)

    plain = tf.outer_shell(tf.Mesh(faces, points))
    got = tf.outer_shell(tagged(faces, points, t))

    assert len(got.faces) == len(plain.faces)
    assert len(got.points) == len(plain.points)
    # the repair emitted the mesh's own coordinates, not the frame's
    np.testing.assert_array_equal(got.points.min(axis=0),
                                  plain.points.min(axis=0))
    np.testing.assert_array_equal(got.points.max(axis=0),
                                  plain.points.max(axis=0))
    assert tf.signed_volume(got) == pytest.approx(
        tf.signed_volume(plain), rel=MEASURE_RTOL[real_dtype])

    moved = tf.outer_shell(baked(faces, points, t))
    assert not np.allclose(moved.points.min(axis=0), plain.points.min(axis=0))


# ==============================================================================
# Main runner
# ==============================================================================

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
