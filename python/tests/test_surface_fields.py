"""
Test geodesic_distances and dirichlet_field

Copyright (c) 2026 Ziga Sajovic, XLAB
"""

import gc

import numpy as np
import pytest
import trueform as tf

INDEX_DTYPES = [np.int32, np.int64]
REAL_DTYPES = [np.float32, np.float64]


# ==============================================================================
# Helpers
# ==============================================================================

def plane(index_dtype, real_dtype, ticks=8):
    return tf.make_plane_mesh(4.0, 4.0, ticks, ticks, dtype=real_dtype,
                              index_dtype=index_dtype)


def find(points, x, y):
    hits = np.flatnonzero((points[:, 0] == x) & (points[:, 1] == y))
    assert len(hits) == 1
    return int(hits[0])


def as_dynamic(faces):
    offsets = np.arange(0, 3 * len(faces) + 1, 3, dtype=faces.dtype)
    return tf.OffsetBlockedArray(offsets, faces.reshape(-1).copy())


def boundary_held(points, linear, rings=1, ticks=8):
    """The points within `rings` grid steps of the plane's rim."""
    step = 4.0 / ticks
    edge = 2.0 - (rings - 1) * step - 1e-6
    held = np.flatnonzero((np.abs(points[:, 0]) >= edge) |
                          (np.abs(points[:, 1]) >= edge))
    p = points[held].astype(np.float64)
    return held, linear(p[:, 0], p[:, 1])


def jittered_cap(seed=7):
    """A jittered grid lifted into a cap, with stated positive weights."""
    rng = np.random.default_rng(seed)
    faces, points = tf.make_plane_mesh(1.0, 1.0, 9, 9, dtype=np.float64,
                                       index_dtype=np.int32)
    points = points.copy()
    interior = (np.abs(points[:, 0]) < 0.49) & (np.abs(points[:, 1]) < 0.49)
    points[interior, :2] += rng.uniform(-0.02, 0.02, (interior.sum(), 2))
    points[:, 2] = 0.3 * (1.0 - 2.0 * (points[:, 0] ** 2 + points[:, 1] ** 2))
    weights = rng.uniform(0.2, 2.0, 3 * len(faces))
    held = np.flatnonzero(~interior)
    values = np.where(points[held, 0] > 0, 1.0, 0.0) + points[held, 1]
    return faces, points, weights, held, values


def dense_field(faces, points, weights, fixed, values, order):
    """The same system solved densely: L from the stated weights, M lumped."""
    n = len(points)
    L = np.zeros((n, n))
    for f, face in enumerate(faces):
        for c in range(3):
            i, j = face[(c + 1) % 3], face[(c + 2) % 3]
            w = weights[3 * f + c]
            L[i, j] -= w
            L[j, i] -= w
            L[i, i] += w
            L[j, j] += w
    p = points.astype(np.float64)
    areas = 0.5 * np.linalg.norm(
        np.cross(p[faces[:, 1]] - p[faces[:, 0]], p[faces[:, 2]] - p[faces[:, 0]]),
        axis=1)
    mass = np.zeros(n)
    np.add.at(mass, faces.reshape(-1), np.repeat(areas / 3.0, 3))
    A = L if order == 1 else L @ np.diag(1.0 / mass) @ L
    free = np.setdiff1d(np.arange(n), fixed)
    x = np.zeros(n)
    x[fixed] = values
    x[free] = np.linalg.solve(A[np.ix_(free, free)], -A[np.ix_(free, fixed)] @ values)
    return x


# ==============================================================================
# geodesic_distances
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_geodesic_distances_on_a_plane_are_the_straight_line(index_dtype, real_dtype):
    faces, points = plane(index_dtype, real_dtype)
    seed = find(points, -2.0, -2.0)
    for mesh in (tf.Mesh(faces, points), tf.Mesh(as_dynamic(faces), points)):
        d = tf.geodesic_distances(mesh, [seed])
        assert d.dtype == real_dtype
        assert d.shape == (len(points),)
        line = np.hypot(points[:, 0] - points[seed, 0],
                        points[:, 1] - points[seed, 1]).astype(np.float64)
        np.testing.assert_array_less(np.abs(d - line), 1e-6 * (1.0 + line))
        assert d[seed] == 0.0


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
def test_geodesic_distances_on_a_sphere_answer_the_arc(index_dtype, real_dtype):
    radius = 2.0
    faces, points = tf.make_sphere_mesh(radius, 60, 80, dtype=real_dtype,
                                        index_dtype=index_dtype)
    seed = int(np.argmin(np.abs(points[:, 2])))
    d = tf.geodesic_distances(tf.Mesh(faces, points), [seed])
    p = points.astype(np.float64)
    arc = radius * np.arccos(np.clip(p @ p[seed] / radius ** 2, -1.0, 1.0))
    span = (arc >= 0.5) & (arc <= 2.5 * radius)
    assert np.max(np.abs(d[span] - arc[span]) / arc[span]) < 0.01


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_geodesic_distances_answer_the_nearest_seed(index_dtype):
    faces, points = plane(index_dtype, np.float64, ticks=16)
    seeds = np.array([find(points, -2.0, -2.0), find(points, 2.0, 2.0)])
    d = tf.geodesic_distances(tf.Mesh(faces, points), seeds)
    first, second = (np.linalg.norm(points - points[s], axis=1) for s in seeds)
    nearest = np.minimum(first, second)
    # Where the two fronts meet, a face sees one corner from each seed and
    # unfolds a source neither of them is: the answer stays near, not exact.
    apart = nearest < 0.85 * np.maximum(first, second)
    assert np.all(np.abs(d[apart] - nearest[apart]) <= 1e-9 * (1.0 + nearest[apart]))
    assert np.all(np.abs(d[~apart] - nearest[~apart]) <= 0.03 * nearest[~apart])
    assert apart.sum() > 200


def test_geodesic_distances_reach_unreached_and_no_seed():
    faces, points = plane(np.int32, np.float64, ticks=16)
    centre = find(points, 0.0, 0.0)
    d = tf.geodesic_distances((faces, points), [centre], radius=1.3)
    line = np.linalg.norm(points - points[centre], axis=1)
    within, past = line <= 1.25, line > 1.3
    assert np.all(np.abs(d[within] - line[within]) <= 1e-9 * (1.0 + line[within]))
    assert np.all(np.isinf(d[past]))
    assert within.sum() > 50 and past.sum() > 100
    assert np.all(np.isinf(tf.geodesic_distances((faces, points), [])))

    seed = find(points, -2.0, -2.0)
    shifted = points.copy()
    shifted[:, 0] += 10.0
    two = np.vstack([faces, faces + len(points)])
    d = tf.geodesic_distances((two, np.vstack([points, shifted])), [seed])
    assert np.all(np.isfinite(d[:len(points)]))
    assert np.all(np.isinf(d[len(points):]))


def test_geodesic_distances_refuse_at_the_boundary():
    faces, points = plane(np.int32, np.float32)
    mesh = tf.Mesh(faces, points)
    with pytest.raises(IndexError):
        tf.geodesic_distances(mesh, [len(points)])
    with pytest.raises(IndexError):
        tf.geodesic_distances(mesh, [-1])
    with pytest.raises(ValueError):
        tf.geodesic_distances(mesh, [[0]])
    with pytest.raises(TypeError):
        tf.geodesic_distances(mesh, [0.5])
    with pytest.raises(ValueError):
        tf.geodesic_distances(mesh, [0], radius=float("nan"))
    with pytest.raises(ValueError):
        tf.geodesic_distances(mesh, [0], radius=-1.0)
    flat = tf.Mesh(faces, np.ascontiguousarray(points[:, :2]))
    with pytest.raises(ValueError):
        tf.geodesic_distances(flat, [0])


def test_geodesic_distances_outlive_their_mesh():
    faces, points = plane(np.int32, np.float64)
    mesh = tf.Mesh(faces, points)
    d = tf.geodesic_distances(mesh, [0])
    expected = d.copy()
    del mesh, faces, points
    gc.collect()
    np.testing.assert_array_equal(d, expected)


# ==============================================================================
# dirichlet_field
# ==============================================================================

@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("real_dtype", REAL_DTYPES)
@pytest.mark.parametrize("order", ["harmonic", "biharmonic"])
def test_cotangent_dirichlet_field_on_a_plane_is_linear(index_dtype, real_dtype, order):
    faces, points = plane(index_dtype, real_dtype, ticks=12)

    def linear(x, y):
        return 2.0 * x - 3.0 * y + 0.5

    rings = 2 if order == "biharmonic" else 1
    held, values = boundary_held(points, linear, rings=rings, ticks=12)
    result = tf.dirichlet_field(tf.Mesh(faces, points), held, values, order)
    assert isinstance(result, tf.DirichletField)
    assert result.status == tf.DirichletFieldStatus.SOLVED
    assert result.field.dtype == real_dtype
    assert result.field.shape == (len(points),)
    p = points.astype(np.float64)
    # The field is stated in the mesh's dtype: a float32 one rounds to it.
    atol = 1e-6 if real_dtype == np.float64 else 1e-5
    np.testing.assert_allclose(result.field, linear(p[:, 0], p[:, 1]), atol=atol)
    assert len(points) - len(held) > 50


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("order", ["harmonic", "biharmonic"])
def test_stated_weights_answer_the_dense_solve(index_dtype, order):
    faces, points, weights, held, values = jittered_cap()
    faces = faces.astype(index_dtype)
    result = tf.dirichlet_field(tf.Mesh(faces, points), held, values, order,
                                corner_weights=weights)
    assert result.status == tf.DirichletFieldStatus.SOLVED
    dense = dense_field(faces, points, weights, held, values,
                        2 if order == "biharmonic" else 1)
    np.testing.assert_allclose(result.field, dense, atol=1e-10)
    assert np.max(np.abs(dense - dense.mean())) > 0.3


def test_stated_weights_are_read_not_replaced_by_the_cotangent():
    faces, points, weights, held, values = jittered_cap()
    mesh = tf.Mesh(faces, points)
    weighted = tf.dirichlet_field(mesh, held, values, "harmonic",
                                  corner_weights=weights)
    cotangent = tf.dirichlet_field(mesh, held, values, "harmonic")
    assert weighted.status == tf.DirichletFieldStatus.SOLVED
    assert cotangent.status == tf.DirichletFieldStatus.SOLVED
    assert np.max(np.abs(weighted.field - cotangent.field)) > 0.05


@pytest.mark.parametrize("order", ["harmonic", "biharmonic"])
def test_an_unanchored_run_refuses_and_holds_its_fixed_rows(order):
    faces, points, weights, held, values = jittered_cap()
    base = len(points)
    points = np.vstack([points, [[4.0, 0.0, 0.0], [5.0, 0.0, 0.0], [4.0, 1.0, 0.0]]])
    faces = np.vstack([faces, [[base, base + 1, base + 2]]]).astype(np.int32)
    weights = np.concatenate([weights, [0.1, 0.3, 0.7]])
    result = tf.dirichlet_field((faces, points), held, values, order,
                                corner_weights=weights)
    assert result.status == tf.DirichletFieldStatus.REFUSED_UNANCHORED
    assert result.field.shape == (len(points),)
    np.testing.assert_array_equal(result.field[held], values)
    anchored = tf.dirichlet_field((faces, points), np.append(held, base),
                                  np.append(values, 0.25), order,
                                  corner_weights=weights)
    assert anchored.status == tf.DirichletFieldStatus.SOLVED


def test_indefinite_weights_refuse():
    faces, points, weights, held, values = jittered_cap()
    result = tf.dirichlet_field((faces, points), held, values, "harmonic",
                                corner_weights=-weights)
    assert result.status == tf.DirichletFieldStatus.REFUSED_INDEFINITE
    np.testing.assert_array_equal(result.field[held], values)


def test_dirichlet_field_refuses_at_the_boundary():
    faces, points = plane(np.int32, np.float64)
    mesh = tf.Mesh(faces, points)
    held, values = boundary_held(points, lambda x, y: x)
    with pytest.raises(ValueError):
        tf.dirichlet_field(mesh, [held[0], held[0]], [0.0, 1.0], "harmonic")
    with pytest.raises(ValueError):
        tf.dirichlet_field(mesh, held, values[:-1], "harmonic")
    with pytest.raises(ValueError):
        tf.dirichlet_field(mesh, held, np.full(len(held), np.inf), "harmonic")
    with pytest.raises(ValueError):
        tf.dirichlet_field(mesh, held, values, "triharmonic")
    with pytest.raises(IndexError):
        tf.dirichlet_field(mesh, [len(points)], [0.0], "harmonic")
    with pytest.raises(ValueError):
        tf.dirichlet_field(mesh, held, values, "harmonic",
                           corner_weights=np.ones(3))
    with pytest.raises(ValueError):
        tf.dirichlet_field(tf.Mesh(as_dynamic(faces), points), held, values,
                           "harmonic")


def test_dirichlet_field_names_the_operator_it_cannot_state():
    faces, points = plane(np.int32, np.float64)
    held, values = boundary_held(points, lambda x, y: x)
    unstated = tf.dirichlet_field((faces, points), held, values, "harmonic",
                                  corner_weights=np.full(3 * len(faces), np.nan))
    assert unstated.status == tf.DirichletFieldStatus.REFUSED_OPERATOR
    np.testing.assert_array_equal(unstated.field[held], values)
    lonely = np.vstack([points, [[9.0, 9.0, 9.0]]])
    cornerless = tf.dirichlet_field((faces, lonely), held, values, "harmonic")
    assert cornerless.status == tf.DirichletFieldStatus.REFUSED_OPERATOR


def test_dirichlet_field_outlives_its_mesh():
    faces, points = plane(np.int64, np.float32)
    mesh = tf.Mesh(faces, points)
    held, values = boundary_held(points, lambda x, y: x + y)
    result = tf.dirichlet_field(mesh, held, values, "harmonic")
    expected = result.field.copy()
    del mesh, faces, points
    gc.collect()
    np.testing.assert_array_equal(result.field, expected)
