"""
Test volume (scalar grid) operations: Volume, isosurface, SDF generators,
volume boolean, and slice contours.

Copyright (c) 2025 Žiga Sajovic, XLAB
"""

import gc
import sys
import pytest
import numpy as np
import trueform as tf


def sphere_field(dims, spacing, origin, center, radius, dtype):
    """The sphere SDF computed in NumPy, indexed field[x, y, z]."""
    x, y, z = np.meshgrid(
        origin[0] + spacing[0] * np.arange(dims[0]),
        origin[1] + spacing[1] * np.arange(dims[1]),
        origin[2] + spacing[2] * np.arange(dims[2]),
        indexing="ij",
    )
    field = np.sqrt(
        (x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2
    ) - radius
    return field.astype(dtype)


def box_field(dims, spacing, origin, center, half_extents, dtype):
    """The exact box SDF computed in NumPy, indexed field[x, y, z]."""
    x, y, z = np.meshgrid(
        origin[0] + spacing[0] * np.arange(dims[0]),
        origin[1] + spacing[1] * np.arange(dims[1]),
        origin[2] + spacing[2] * np.arange(dims[2]),
        indexing="ij",
    )
    qx = np.abs(x - center[0]) - half_extents[0]
    qy = np.abs(y - center[1]) - half_extents[1]
    qz = np.abs(z - center[2]) - half_extents[2]
    outside = np.sqrt(
        np.maximum(qx, 0) ** 2 + np.maximum(qy, 0) ** 2 + np.maximum(qz, 0) ** 2
    )
    inside = np.minimum(np.maximum(qx, np.maximum(qy, qz)), 0)
    return (outside + inside).astype(dtype)


def closed_manifold(faces, points):
    mesh = tf.Mesh(faces, points)
    return tf.is_closed(mesh) and tf.is_manifold(mesh)


INTEGER_SAMPLE_DTYPES = [np.int16, np.uint16, np.uint8]

# Byte order is a fact only where there is more than one byte to order.
MULTIBYTE_SAMPLE_DTYPES = [np.int16, np.uint16]


def ct_sphere_counts(dims, spacing, center, radius, dtype, scale=100.0):
    """A sphere's signed distance rasterized into an integer field, as CT is.

    The samples are counts in `dtype`; the surface sits where the count crosses
    `bias`, which is 0 for a signed field and the type's midpoint for an
    unsigned one — an unsigned field cannot hold a negative inside.
    """
    x, y, z = np.meshgrid(
        spacing[0] * np.arange(dims[0]),
        spacing[1] * np.arange(dims[1]),
        spacing[2] * np.arange(dims[2]),
        indexing="ij",
    )
    signed = np.sqrt(
        (x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2
    ) - radius
    info = np.iinfo(dtype)
    bias = 0.0 if info.min < 0 else float(info.max // 2)
    counts = np.clip(np.rint(bias + scale * signed), info.min, info.max)
    return np.asfortranarray(counts.astype(dtype)), bias


def radial_extent(points, center):
    return np.linalg.norm(points - np.asarray(center), axis=1)


def mean_outward_sign(faces, points, center):
    """+1 when the triangles wind away from `center`, -1 when they wind into it."""
    tri = points[faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    midpoints = tri.mean(axis=1)
    return np.sign(
        np.sum(np.einsum("ij,ij->i", normals, midpoints - np.asarray(center)))
    )


# ============================================================================
# Construction, dtype, layout
# ============================================================================


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_volume_construction(dtype):
    """Volume preserves dims, spacing, origin, dtype, and sample values"""
    samples = sphere_field((8, 10, 12), (0.5, 0.5, 0.5), (0, 0, 0),
                           (2.0, 2.5, 3.0), 1.5, dtype)
    vol = tf.Volume(samples, spacing=(0.5, 0.5, 0.5), origin=(0.0, 0.0, 0.0))

    assert vol.dims == (8, 10, 12)
    assert vol.spacing == (0.5, 0.5, 0.5)
    assert vol.origin == (0.0, 0.0, 0.0)
    assert vol.voxel_count == 8 * 10 * 12
    assert vol.dtype == np.dtype(dtype)
    assert vol.samples.shape == (8, 10, 12)
    assert np.array_equal(vol.samples, samples)


def test_volume_defaults_and_setters():
    """Default spacing/origin and their setters"""
    vol = tf.Volume(np.zeros((4, 4, 4), dtype=np.float32))
    assert vol.spacing == (1.0, 1.0, 1.0)
    assert vol.origin == (0.0, 0.0, 0.0)

    vol.spacing = (0.25, 0.5, 1.0)
    vol.origin = (-1.0, 2.0, 3.0)
    assert vol.spacing == (0.25, 0.5, 1.0)
    assert vol.origin == (-1.0, 2.0, 3.0)


def test_volume_non_contiguous_samples():
    """A non-contiguous (sliced) samples array is normalized at the boundary"""
    base = sphere_field((16, 16, 16), (1, 1, 1), (0, 0, 0),
                        (8, 8, 8), 5.0, np.float32)
    strided = base[::2, ::2, ::2]
    assert not strided.flags["C_CONTIGUOUS"]
    assert not strided.flags["F_CONTIGUOUS"]

    vol = tf.Volume(strided)
    assert vol.dims == (8, 8, 8)
    assert np.array_equal(vol.samples, strided)


@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.float16])
def test_volume_unaccepted_dtype_converts_to_float32(dtype):
    """A dtype that is not an accepted SAMPLE type converts to float32"""
    samples = np.arange(27, dtype=dtype).reshape(3, 3, 3)
    vol = tf.Volume(samples)
    assert vol.dtype == np.float32
    assert np.array_equal(vol.samples, samples.astype(np.float32))


def test_volume_invalid_inputs():
    """Rank, emptiness, and grid-parameter validation"""
    with pytest.raises(TypeError):
        tf.Volume([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError):
        tf.Volume(np.zeros((4, 4), dtype=np.float32))
    with pytest.raises(ValueError):
        tf.Volume(np.zeros((0, 4, 4), dtype=np.float32))
    with pytest.raises(ValueError):
        tf.Volume(np.zeros((4, 4, 4), dtype=np.float32), spacing=(1.0, 0.0, 1.0))
    with pytest.raises(ValueError):
        tf.Volume(np.zeros((4, 4, 4), dtype=np.float32), origin=(1.0, 2.0))


def test_volume_samples_view_mutates_field():
    """The samples property is a view of the native field, not a copy"""
    vol = tf.Volume(np.ones((4, 4, 4), dtype=np.float32))
    vol.samples[1, 2, 3] = -7.0
    assert vol.samples[1, 2, 3] == -7.0


# ============================================================================
# Ownership / lifetime
# ============================================================================


def test_volume_borrows_samples_zero_copy():
    """An F-ordered float source is borrowed with no copy: one shared memory"""
    samples = np.asfortranarray(
        sphere_field((16, 16, 16), (0.5, 0.5, 0.5), (0, 0, 0),
                     (4, 4, 4), 2.5, np.float32)
    )
    vol = tf.Volume(samples, spacing=(0.5, 0.5, 0.5))

    assert np.shares_memory(samples, vol.samples)
    assert np.array_equal(vol.samples, samples)

    # A write through the source array is a write to the field itself.
    original = samples.copy()
    samples[:] = 1.0
    faces, points = tf.isosurface(vol)
    assert len(faces) == 0
    samples[:] = original
    faces, points = tf.isosurface(vol)
    assert len(faces) > 0


def test_volume_survives_source_deletion():
    """The volume stays valid after its source array is deleted"""
    samples = sphere_field((16, 16, 16), (0.5, 0.5, 0.5), (0, 0, 0),
                           (4, 4, 4), 2.5, np.float32)
    expected = samples.copy()
    vol = tf.Volume(samples, spacing=(0.5, 0.5, 0.5))
    del samples
    gc.collect()

    assert np.array_equal(vol.samples, expected)
    faces, points = tf.isosurface(vol)
    assert len(faces) > 0


def test_samples_view_keeps_native_field_alive():
    """The samples view retains its native owner after the Volume is deleted"""
    vol = tf.Volume(
        sphere_field((8, 8, 8), (1, 1, 1), (0, 0, 0), (4, 4, 4), 2.0,
                     np.float32)
    )
    view = vol.samples
    expected = view.copy()
    del vol
    gc.collect()

    assert np.array_equal(view, expected)


# ============================================================================
# sphere_sdf + isosurface
# ============================================================================


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_sphere_sdf_matches_numpy(dtype):
    """The native sphere SDF equals the analytic field"""
    vol = tf.sphere_sdf((12, 12, 12), (0.5, 0.5, 0.5), (0.0, 0.0, 0.0),
                        (3.0, 3.0, 3.0), 2.0, dtype=dtype)
    expected = sphere_field((12, 12, 12), (0.5, 0.5, 0.5), (0, 0, 0),
                            (3, 3, 3), 2.0, dtype)
    assert vol.dtype == np.dtype(dtype)
    assert np.allclose(vol.samples, expected, atol=1e-5)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_isosurface_sphere_closed_manifold(dtype):
    """The sphere isosurface is a closed, manifold, finite mesh on the sphere"""
    radius = 5.0
    center = (8.0, 8.0, 8.0)
    vol = tf.sphere_sdf((33, 33, 33), (0.5, 0.5, 0.5), (0.0, 0.0, 0.0),
                        center, radius, dtype=dtype)
    faces, points = tf.isosurface(vol)

    assert faces.dtype == np.int32
    assert points.dtype == np.dtype(dtype)
    assert len(faces) > 0
    assert np.all(np.isfinite(points))
    assert closed_manifold(faces, points)

    # Every vertex lies on the zero level set, up to interpolation error.
    dist = np.linalg.norm(points - np.array(center), axis=1)
    assert np.max(np.abs(dist - radius)) < 0.5

    # Outward winding: the enclosed signed volume approximates the sphere's.
    sv = tf.signed_volume((faces, points))
    assert sv > 0
    assert abs(sv - 4.0 / 3.0 * np.pi * radius**3) < 0.1 * 4.0 / 3.0 * np.pi * radius**3


def test_isosurface_offset_surface():
    """A nonzero isovalue extracts the offset sphere"""
    radius = 4.0
    center = (8.0, 8.0, 8.0)
    vol = tf.sphere_sdf((33, 33, 33), (0.5, 0.5, 0.5), (0.0, 0.0, 0.0),
                        center, radius, dtype=np.float32)
    faces, points = tf.isosurface(vol, iso=1.0)

    assert len(faces) > 0
    dist = np.linalg.norm(points - np.array(center), axis=1)
    assert np.max(np.abs(dist - (radius + 1.0))) < 0.5


def test_isosurface_not_crossed_is_empty():
    """An isovalue outside the field's range produces an empty mesh"""
    vol = tf.Volume(np.ones((8, 8, 8), dtype=np.float32))
    faces, points = tf.isosurface(vol)
    assert len(faces) == 0
    assert len(points) == 0


def test_isosurface_invalid_options():
    """Method and stabilizer validation"""
    vol = tf.Volume(np.ones((4, 4, 4), dtype=np.float32))
    with pytest.raises(ValueError):
        tf.isosurface(vol, method="marching_cubes")
    with pytest.raises(ValueError):
        tf.isosurface(vol, method="dual_contouring", stabilizer=-1.0)
    with pytest.raises(TypeError):
        tf.isosurface(np.ones((4, 4, 4), dtype=np.float32))


# ============================================================================
# Dual contouring on a box SDF
# ============================================================================


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_dual_contouring_recovers_box_corners(dtype):
    """Dual contouring lands vertices on the box corner; flying edges cannot"""
    dims = (24, 24, 24)
    spacing = (0.25, 0.25, 0.25)
    center = (2.875, 2.875, 2.875)
    half = (1.8, 1.8, 1.8)
    samples = box_field(dims, spacing, (0, 0, 0), center, half, dtype)
    vol = tf.Volume(samples, spacing=spacing)

    fe_faces, fe_points = tf.isosurface(vol)
    dc_faces, dc_points = tf.isosurface(vol, method="dual_contouring")

    assert closed_manifold(fe_faces, fe_points)
    assert closed_manifold(dc_faces, dc_points)

    corner = np.array(center) + np.array(half)
    fe_corner_dist = np.min(np.linalg.norm(fe_points - corner, axis=1))
    dc_corner_dist = np.min(np.linalg.norm(dc_points - corner, axis=1))

    # The corner sits off the grid, so a grid-edge vertex stays away from it
    # while a fitted vertex reaches it: sharper geometry, stated relatively.
    assert dc_corner_dist < fe_corner_dist
    assert dc_corner_dist < spacing[0]


# ============================================================================
# mesh_sdf round-trip
# ============================================================================


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
def test_mesh_sdf_round_trip(dtype, index_dtype):
    """mesh -> SDF -> isosurface recovers the original surface within a voxel"""
    radius = 5.0
    faces, points = tf.make_sphere_mesh(radius, dtype=dtype,
                                        index_dtype=index_dtype)
    spacing = 0.4
    margin = 3 * spacing
    vol = tf.mesh_sdf(
        (faces, points),
        dims=(32, 32, 32),
        spacing=(spacing, spacing, spacing),
        origin=(-radius - margin,) * 3,
    )
    assert vol.dtype == np.dtype(dtype)

    out_faces, out_points = tf.isosurface(vol)
    assert closed_manifold(out_faces, out_points)

    # The sphere mesh's vertices lie exactly at the radius, so every extracted
    # vertex must lie within a voxel of the analytic surface.
    dist = np.linalg.norm(out_points, axis=1)
    assert np.max(np.abs(dist - radius)) < spacing

    sv_in = tf.signed_volume((faces, points))
    sv_out = tf.signed_volume((out_faces, out_points))
    assert abs(sv_out - sv_in) < 0.1 * abs(sv_in)


def test_mesh_sdf_sign_convention():
    """Negative inside, positive outside"""
    faces, points = tf.make_sphere_mesh(3.0)
    vol = tf.mesh_sdf(tf.Mesh(faces, points), (17, 17, 17),
                      (0.5, 0.5, 0.5), (-4.0, -4.0, -4.0))
    assert vol.samples[8, 8, 8] < 0  # grid center = sphere center
    assert vol.samples[0, 0, 0] > 0  # grid corner, outside


def test_mesh_sdf_samples_the_frame_the_mesh_states():
    """A transformed mesh's SDF puts the zero level set where the mesh stands"""
    faces, points = tf.make_sphere_mesh(1.0)
    mesh = tf.Mesh(faces, points)
    # x spans [-2, 6] (41 samples), y and z span [-2, 2]: both the local
    # center (0,0,0) and the translated center (4,0,0) are covered.
    dims = (41, 21, 21)
    spacing = (0.2, 0.2, 0.2)
    origin = (-2.0, -2.0, -2.0)

    local = tf.mesh_sdf(mesh, dims, spacing, origin)

    t = np.eye(4, dtype=np.float32)
    t[0, 3] = 4.0
    mesh.transformation = t
    world = tf.mesh_sdf(mesh, dims, spacing, origin)

    # Sample (10,10,10) is world (0,0,0); sample (30,10,10) is world (4,0,0).
    assert local.samples[10, 10, 10] < 0
    assert local.samples[30, 10, 10] > 0
    assert world.samples[30, 10, 10] < 0
    assert world.samples[10, 10, 10] > 0

    # The transformed field is the local field translated by the frame.
    assert np.isclose(world.samples[30, 10, 10], local.samples[10, 10, 10],
                      atol=1e-4)
    assert np.isclose(world.samples[35, 12, 11], local.samples[15, 12, 11],
                      atol=1e-4)


def test_mesh_sdf_dynamic_mesh():
    """A dynamic (offset-blocked) mesh samples through the same entry"""
    faces, points = tf.make_box_mesh(4.0, 4.0, 4.0)
    dyn_faces = tf.as_offset_blocked(faces)
    vol = tf.mesh_sdf(tf.Mesh(dyn_faces, points), (17, 17, 17),
                      (0.5, 0.5, 0.5), (-4.0, -4.0, -4.0))
    assert vol.samples[8, 8, 8] < 0
    assert vol.samples[0, 0, 0] > 0


def test_mesh_sdf_invalid_inputs():
    """Dimensionality and argument validation"""
    faces, points = tf.make_sphere_mesh(1.0)
    with pytest.raises(ValueError):
        tf.mesh_sdf((faces, points), (8, 8), (1, 1, 1), (0, 0, 0))
    with pytest.raises(ValueError):
        tf.mesh_sdf((faces, points), (8, 8, 0), (1, 1, 1), (0, 0, 0))
    with pytest.raises(TypeError):
        tf.mesh_sdf("not a mesh", (8, 8, 8), (1, 1, 1), (0, 0, 0))


# ============================================================================
# volume_boolean
# ============================================================================


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_volume_boolean_union_matching_grids(dtype):
    """Union of two overlapping spheres on one grid is min(a, b) exactly"""
    grid = dict(dims=(33, 33, 33), spacing=(0.5, 0.5, 0.5),
                origin=(0.0, 0.0, 0.0))
    a = tf.sphere_sdf(center=(6.0, 8.0, 8.0), radius=4.0, dtype=dtype, **grid)
    b = tf.sphere_sdf(center=(10.0, 8.0, 8.0), radius=4.0, dtype=dtype, **grid)

    merged = tf.volume_boolean(a, b, "union")
    assert merged.dims == a.dims
    assert np.array_equal(merged.samples, np.minimum(a.samples, b.samples))

    faces, points = tf.isosurface(merged)
    assert closed_manifold(faces, points)

    sphere = 4.0 / 3.0 * np.pi * 4.0**3
    sv = tf.signed_volume((faces, points))
    assert sphere < sv < 2 * sphere


def test_volume_boolean_intersection_and_difference():
    """Intersection and difference relate to the operands as solids"""
    grid = dict(dims=(33, 33, 33), spacing=(0.5, 0.5, 0.5),
                origin=(0.0, 0.0, 0.0))
    a = tf.sphere_sdf(center=(6.0, 8.0, 8.0), radius=4.0, **grid)
    b = tf.sphere_sdf(center=(10.0, 8.0, 8.0), radius=4.0, **grid)

    inter_faces, inter_points = tf.isosurface(
        tf.volume_boolean(a, b, "intersection"))
    diff_faces, diff_points = tf.isosurface(
        tf.volume_boolean(a, b, "difference"))

    assert closed_manifold(inter_faces, inter_points)
    assert closed_manifold(diff_faces, diff_points)

    sphere = 4.0 / 3.0 * np.pi * 4.0**3
    sv_inter = tf.signed_volume((inter_faces, inter_points))
    sv_diff = tf.signed_volume((diff_faces, diff_points))
    assert 0 < sv_inter < sphere
    assert 0 < sv_diff < sphere
    # A = (A \ B) + (A n B), up to sampling error.
    assert abs(sv_diff + sv_inter - sphere) < 0.1 * sphere


def test_volume_boolean_resampled_grids():
    """Fields on different grids resample onto a common grid and stay closed"""
    a = tf.sphere_sdf((33, 33, 33), (0.5, 0.5, 0.5), (0.0, 0.0, 0.0),
                      (8.0, 8.0, 8.0), 4.0)
    b = tf.sphere_sdf((41, 41, 41), (0.4, 0.4, 0.4), (2.0, 2.0, 2.0),
                      (10.0, 10.0, 10.0), 4.0)
    merged = tf.volume_boolean(a, b, "union")
    faces, points = tf.isosurface(merged)
    assert closed_manifold(faces, points)


def test_volume_boolean_invalid_inputs():
    """Operation and dtype validation"""
    a = tf.Volume(np.ones((4, 4, 4), dtype=np.float32))
    b = tf.Volume(np.ones((4, 4, 4), dtype=np.float64))
    with pytest.raises(TypeError):
        tf.volume_boolean(a, b, "union")
    with pytest.raises(ValueError):
        tf.volume_boolean(a, tf.Volume(np.ones((4, 4, 4), dtype=np.float32)),
                          "xor")


# ============================================================================
# volume_slice_contours
# ============================================================================


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_slice_contours_sphere(dtype):
    """A center slice of a sphere SDF contours to the great circle"""
    radius = 5.0
    center = (8.0, 8.0, 8.0)
    vol = tf.sphere_sdf((33, 33, 33), (0.5, 0.5, 0.5), (0.0, 0.0, 0.0),
                        center, radius, dtype=dtype)

    paths, points = tf.volume_slice_contours(
        vol,
        plane_origin=(0.0, 0.0, 8.0),
        u=(1.0, 0.0, 0.0),
        v=(0.0, 1.0, 0.0),
        dims2=(65, 65),
        spacing2=(0.25, 0.25),
        isovalues=0.0,
    )

    # The great circle is one closed polyline over welded points.
    assert len(paths) == 1
    assert points.dtype == np.dtype(dtype)
    ring = paths[0]
    assert len(ring) > 3
    assert ring[0] == ring[-1]
    # All contour points sit in the slice plane, on the great circle.
    assert np.allclose(points[:, 2], 8.0, atol=1e-5)
    dist = np.linalg.norm(points[:, :2] - np.array(center[:2]), axis=1)
    assert np.max(np.abs(dist - radius)) < 0.5


def test_slice_contours_multiple_isovalues():
    """Several isovalues contour the same slice; each adds segments"""
    vol = tf.sphere_sdf((33, 33, 33), (0.5, 0.5, 0.5), (0.0, 0.0, 0.0),
                        (8.0, 8.0, 8.0), 4.0)
    paths_one, _ = tf.volume_slice_contours(
        vol, (0.0, 0.0, 8.0), (1, 0, 0), (0, 1, 0), (65, 65), (0.25, 0.25),
        0.0)
    paths_two, _ = tf.volume_slice_contours(
        vol, (0.0, 0.0, 8.0), (1, 0, 0), (0, 1, 0), (65, 65), (0.25, 0.25),
        [0.0, 1.0])
    # one closed ring per isovalue
    assert len(paths_one) == 1
    assert len(paths_two) == 2


def test_slice_contours_missed_isovalue_is_empty():
    """An isovalue the slice never crosses produces no contours"""
    vol = tf.Volume(np.ones((8, 8, 8), dtype=np.float32))
    paths, points = tf.volume_slice_contours(
        vol, (0.0, 0.0, 4.0), (1, 0, 0), (0, 1, 0), (16, 16), (0.5, 0.5),
        -1.0)
    assert len(paths) == 0
    assert len(points) == 0


# ============================================================================
# The type a call emits in
# ============================================================================


@pytest.mark.parametrize("storage", [np.float32, np.float64])
@pytest.mark.parametrize("request_dtype", [np.float32, np.float64])
def test_isosurface_emits_the_requested_dtype(storage, request_dtype):
    """The coordinate type is the call's request, not the storage's"""
    vol = tf.sphere_sdf((33, 33, 33), (0.5,) * 3, (0.0,) * 3,
                        (8.0, 8.0, 8.0), 5.0, dtype=storage)
    faces, points = tf.isosurface(vol, dtype=request_dtype)
    assert points.dtype == np.dtype(request_dtype)
    # the storage width does not move a crossing off its edge
    own_faces, _ = tf.isosurface(vol)
    assert faces.shape == own_faces.shape


def test_isosurface_default_dtype_is_the_volumes():
    """An unstated request is the volume's own sample dtype"""
    for dtype in (np.float32, np.float64):
        vol = tf.sphere_sdf((17, 17, 17), (1.0,) * 3, (0.0,) * 3,
                            (8.0, 8.0, 8.0), 5.0, dtype=dtype)
        _, points = tf.isosurface(vol)
        assert points.dtype == np.dtype(dtype)


def test_sphere_sdf_unstated_dtype_is_the_documented_default():
    """`dtype=None` is unstated, and unstated is the entry's own default

    A generator has no input to inherit from, so its unstated request is its
    documented default. `np.dtype(None)` is float64, so an entry that resolves
    None by handing it to NumPy answers something else entirely.
    """
    unstated = tf.sphere_sdf((9, 9, 9), (1.0,) * 3, (0.0,) * 3,
                             (4.0,) * 3, 2.0, dtype=None)
    default = tf.sphere_sdf((9, 9, 9), (1.0,) * 3, (0.0,) * 3,
                            (4.0,) * 3, 2.0)

    assert unstated.dtype == np.dtype(np.float32)
    assert unstated.dtype == default.dtype
    assert np.array_equal(unstated.samples, default.samples)


def test_volume_entries_reject_an_unsupported_dtype():
    """Only float32 and float64 are coordinate types"""
    vol = tf.sphere_sdf((9, 9, 9), (1.0,) * 3, (0.0,) * 3, (4.0,) * 3, 2.0)
    mesh_faces, mesh_points = tf.make_sphere_mesh(1.0)
    with pytest.raises(TypeError):
        tf.isosurface(vol, dtype=np.int32)
    with pytest.raises(TypeError):
        tf.volume_boolean(vol, vol, "union", dtype=np.int32)
    with pytest.raises(TypeError):
        tf.volume_slice_contours(vol, (0.0, 0.0, 4.0), (1, 0, 0), (0, 1, 0),
                                 (8, 8), (1.0, 1.0), 0.0, dtype=np.int32)
    with pytest.raises(TypeError):
        tf.mesh_sdf((mesh_faces, mesh_points), (8, 8, 8), (0.5,) * 3,
                    (-2.0,) * 3, dtype=np.int32)


def test_produced_fields_carry_the_requested_dtype():
    """mesh_sdf and volume_boolean store the field in the requested type"""
    faces, points = tf.make_sphere_mesh(1.0)
    field = tf.mesh_sdf((faces, points), (12, 12, 12), (0.4,) * 3,
                        (-2.2,) * 3, dtype=np.float64)
    assert field.dtype == np.dtype(np.float64)

    a = tf.sphere_sdf((17, 17, 17), (0.5,) * 3, (0.0,) * 3, (4.0,) * 3, 2.0)
    combined = tf.volume_boolean(a, a, "union", dtype=np.float64)
    assert combined.dtype == np.dtype(np.float64)


def test_slice_contours_isovalue_is_not_narrowed_to_the_sample_type():
    """An isovalue reaches the call in the deciding type, not the storage's

    On a float32 field asked to decide in float64, an isovalue that float32
    cannot hold exactly must still arrive exact: contouring the double and
    contouring its own float32 rounding are different level sets.
    """
    spacing = (0.25, 0.25, 0.25)
    vol = tf.Volume(
        sphere_field((33, 33, 33), spacing, (0, 0, 0), (4.0, 4.0, 4.0), 2.0,
                     np.float32),
        spacing=spacing,
    )

    isovalue = 0.1
    rounded = float(np.float32(isovalue))
    assert isovalue != rounded

    def contour_at(value):
        return tf.volume_slice_contours(
            vol, (0.0, 0.0, 4.0), (1, 0, 0), (0, 1, 0), (40, 40), (0.2, 0.2),
            value, dtype=np.float64)[1]

    exact, narrowed = contour_at(isovalue), contour_at(rounded)
    assert len(exact) > 0
    assert exact.shape == narrowed.shape
    assert not np.array_equal(exact, narrowed)
    # The difference is the isovalue's own float32 error, not a coarse shift.
    assert np.abs(exact - narrowed).max() < 1e-6


def test_slice_contours_carry_the_requested_dtype():
    """The contour points are emitted in the requested type"""
    vol = tf.sphere_sdf((33, 33, 33), (0.5,) * 3, (0.0,) * 3,
                        (8.0, 8.0, 8.0), 5.0)
    _, points = tf.volume_slice_contours(
        vol, (0.0, 0.0, 8.0), (1, 0, 0), (0, 1, 0), (65, 65), (0.25, 0.25),
        0.0, dtype=np.float64)
    assert points.dtype == np.dtype(np.float64)


# ============================================================================
# Integer sample rows: what a sample is vs where it stands
# ============================================================================


@pytest.mark.parametrize("dtype", INTEGER_SAMPLE_DTYPES)
def test_integer_volume_construction_round_trip(dtype):
    """An integer field keeps its dtype and its fractional grid"""
    counts, _ = ct_sphere_counts((8, 10, 12), (0.7, 0.7, 1.5),
                                 (2.8, 3.5, 9.0), 2.0, dtype)
    vol = tf.Volume(counts, spacing=(0.7, 0.7, 1.5), origin=(-1.0, 2.0, 0.5))

    assert vol.dtype == np.dtype(dtype)
    assert vol.dims == (8, 10, 12)
    assert vol.voxel_count == 8 * 10 * 12
    assert np.array_equal(vol.samples, counts)
    assert vol.samples.dtype == np.dtype(dtype)
    # The grid is floating even though the samples are not: this is the whole
    # point — 0.7 mm spacing is not expressible in the sample type.
    assert vol.coordinate_dtype == np.dtype(np.float32)
    assert vol.spacing == pytest.approx((0.7, 0.7, 1.5), abs=1e-6)
    assert vol.origin == pytest.approx((-1.0, 2.0, 0.5), abs=1e-6)


@pytest.mark.parametrize("dtype", INTEGER_SAMPLE_DTYPES)
def test_integer_volume_is_borrowed_not_widened(dtype):
    """The samples are borrowed in place: no copy, no widening"""
    counts, _ = ct_sphere_counts((16, 16, 16), (0.5, 0.5, 0.5),
                                 (4.0, 4.0, 4.0), 2.5, dtype)
    vol = tf.Volume(counts, spacing=(0.5, 0.5, 0.5))

    assert np.shares_memory(counts, vol.samples)
    assert vol.samples.nbytes == counts.nbytes
    assert vol.samples.itemsize == np.dtype(dtype).itemsize

    # One field under two names: a write through either is the same write.
    vol.samples[2, 3, 4] = 7
    assert counts[2, 3, 4] == 7
    counts[4, 3, 2] = 9
    assert vol.samples[4, 3, 2] == 9


@pytest.mark.parametrize("dtype", INTEGER_SAMPLE_DTYPES)
def test_integer_volume_survives_source_deletion(dtype):
    """The retained ndarray keeps the field alive after the name is dropped"""
    counts, bias = ct_sphere_counts((20, 20, 20), (0.5, 0.5, 0.5),
                                    (5.0, 5.0, 5.0), 3.0, dtype)
    vol = tf.Volume(counts, spacing=(0.5, 0.5, 0.5))
    del counts
    gc.collect()

    faces, points = tf.isosurface(vol, bias)
    assert len(faces) > 0
    assert closed_manifold(faces, points)


def test_isosurface_of_an_int16_ct_field():
    """An int16 field on an anisotropic millimetre grid extracts in millimetres"""
    spacing = (0.7, 0.7, 1.5)
    center = (11.2, 11.2, 24.0)
    radius = 6.0
    counts, bias = ct_sphere_counts((32, 32, 32), spacing, center, radius,
                                    np.int16)
    vol = tf.Volume(counts, spacing=spacing)

    faces, points = tf.isosurface(vol, bias)
    assert len(faces) > 0
    assert closed_manifold(faces, points)

    # The interpolation ran in the deciding float type, so the vertices land
    # off the integer lattice and recover the sphere in world units.
    r = radial_extent(points, center)
    assert r.min() > radius - 0.1
    assert r.max() < radius + 0.1
    assert np.any(np.abs(points - np.rint(points)) > 1e-4)


@pytest.mark.parametrize("dtype", INTEGER_SAMPLE_DTYPES)
@pytest.mark.parametrize("request_dtype", [None, np.float32, np.float64])
def test_integer_field_emits_a_real_valued_surface(dtype, request_dtype):
    """An integer-sampled field emits float geometry; float32 when unstated"""
    counts, bias = ct_sphere_counts((24, 24, 24), (0.5, 0.5, 0.5),
                                    (6.0, 6.0, 6.0), 3.5, dtype)
    vol = tf.Volume(counts, spacing=(0.5, 0.5, 0.5))

    faces, points = tf.isosurface(vol, bias, dtype=request_dtype)
    expected = np.float32 if request_dtype is None else request_dtype
    assert points.dtype == np.dtype(expected)
    assert len(faces) > 0


def test_isosurface_of_a_uint8_mask_surfaces_the_background_side():
    """A mask and its negation are the same surface, wound opposite ways

    `sample < iso` is inside, so thresholding a 0/255 mask at its midpoint
    treats the BACKGROUND as inside and winds into the foreground. Negating
    the mask into a signed field restores outward winding. Both fields put
    every crossing at t = 0.5 of its edge, so the identity is exact: the same
    points, and each face the other's reversed winding.
    """
    dims, center, radius = (24, 24, 24), (11.5, 11.5, 11.5), 7.0
    x, y, z = np.meshgrid(*(np.arange(d) for d in dims), indexing="ij")
    inside = np.sqrt(
        (x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2
    ) < radius

    mask = np.asfortranarray(np.where(inside, 255, 0).astype(np.uint8))
    mask_faces, mask_points = tf.isosurface(tf.Volume(mask), 127.5)

    signed = np.asfortranarray((127.5 - mask.astype(np.float32)))
    signed_faces, signed_points = tf.isosurface(tf.Volume(signed), 0.0)

    assert len(mask_faces) > 0
    assert closed_manifold(mask_faces, mask_points)
    assert np.array_equal(mask_points, signed_points)
    assert np.array_equal(
        mask_faces, np.roll(signed_faces[:, ::-1], 1, axis=1)
    )
    assert mean_outward_sign(mask_faces, mask_points, center) == -1.0
    assert mean_outward_sign(signed_faces, signed_points, center) == +1.0


@pytest.mark.parametrize("dtype", INTEGER_SAMPLE_DTYPES)
def test_dual_contouring_of_an_integer_field(dtype):
    """The other extractor reads an integer field through the same one cast"""
    spacing = (0.5, 0.5, 0.5)
    center = (5.0, 5.0, 5.0)
    counts, bias = ct_sphere_counts((24, 24, 24), spacing, center, 3.0, dtype)
    vol = tf.Volume(counts, spacing=spacing)

    faces, points = tf.isosurface(vol, bias, method="dual_contouring")
    assert len(faces) > 0
    assert points.dtype == np.dtype(np.float32)
    assert closed_manifold(faces, points)

    r = radial_extent(points, center)
    assert r.min() > 3.0 - 0.5
    assert r.max() < 3.0 + 0.5


@pytest.mark.parametrize("dtype", MULTIBYTE_SAMPLE_DTYPES)
def test_byte_swapped_samples_keep_their_row(dtype):
    """A big-endian field — NIfTI's own layout — stays an integer field"""
    native, bias = ct_sphere_counts((16, 16, 16), (0.5, 0.5, 0.5),
                                    (4.0, 4.0, 4.0), 2.5, dtype)
    swapped = np.asfortranarray(
        native.astype(native.dtype.newbyteorder(">"))
    )
    assert swapped.dtype.byteorder == ">"

    vol = tf.Volume(swapped, spacing=(0.5, 0.5, 0.5))
    assert vol.dtype == np.dtype(dtype)
    assert vol.coordinate_dtype == np.dtype(np.float32)
    assert np.array_equal(vol.samples, native)

    # The byteswap is a copy, so this one is not borrowed — but it is the
    # sample width the caller gave, not a float32 widening of it.
    assert vol.samples.itemsize == np.dtype(dtype).itemsize

    faces, _ = tf.isosurface(vol, bias)
    native_faces, _ = tf.isosurface(
        tf.Volume(native, spacing=(0.5, 0.5, 0.5)), bias)
    assert np.array_equal(faces, native_faces)


def test_volume_wrapper_refuses_a_mismatched_sample_dtype():
    """The native rows are noconvert: no row silently borrows a conversion"""
    from trueform._trueform.volume import VolumeWrapperInt16

    flat = np.zeros(64, dtype=np.float32)
    with pytest.raises(TypeError):
        VolumeWrapperInt16(flat, (4, 4, 4), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))


def test_slice_contours_of_an_integer_field_take_a_fractional_isovalue():
    """The isovalue is a field value in the deciding type, not a sample"""
    spacing = (0.7, 0.7, 1.5)
    center = (11.2, 11.2, 24.0)
    counts, bias = ct_sphere_counts((32, 32, 32), spacing, center, 6.0,
                                    np.int16)
    vol = tf.Volume(counts, spacing=spacing)

    def contour_at(isovalue):
        return tf.volume_slice_contours(
            vol, (0.0, 0.0, center[2]), (1, 0, 0), (0, 1, 0), (48, 48),
            (0.5, 0.5), isovalue)

    paths, fractional = contour_at(bias + 250.5)
    assert len(paths) > 0
    assert len(fractional) > 0
    assert fractional.dtype == np.dtype(np.float32)

    # The pair is a fractional isovalue and its own truncation: were the
    # isovalue narrowed to the int16 sample type at the boundary, both would
    # arrive as 250 and these contours would be the same points.
    _, truncated = contour_at(bias + 250.0)
    assert len(truncated) > 0
    assert not np.array_equal(fractional, truncated)


def test_volume_boolean_refuses_unsigned_samples():
    """An unsigned field is not an SDF, and the refusal names the accepted set"""
    mask = np.zeros((8, 8, 8), dtype=np.uint8, order="F")
    a, b = tf.Volume(mask), tf.Volume(mask.copy(order="F"))

    with pytest.raises(TypeError) as excinfo:
        tf.volume_boolean(a, b, "union")
    message = str(excinfo.value)
    assert "uint8" in message
    for accepted in ("float32", "float64", "int16"):
        assert accepted in message

    wide = np.zeros((8, 8, 8), dtype=np.uint16, order="F")
    with pytest.raises(TypeError):
        tf.volume_boolean(tf.Volume(wide), tf.Volume(wide.copy(order="F")))


def test_volume_boolean_of_int16_fields_decides_in_float():
    """Signed integer samples combine, and the combined field is real-valued"""
    spacing = (0.5, 0.5, 0.5)
    left, bias = ct_sphere_counts((20, 20, 20), spacing, (4.0, 5.0, 5.0), 2.5,
                                  np.int16)
    right, _ = ct_sphere_counts((20, 20, 20), spacing, (6.0, 5.0, 5.0), 2.5,
                                np.int16)
    a = tf.Volume(left, spacing=spacing)
    b = tf.Volume(right, spacing=spacing)

    merged = tf.volume_boolean(a, b, "union")
    assert merged.dtype == np.dtype(np.float32)
    assert merged.dims == a.dims
    assert merged.spacing == pytest.approx(spacing, abs=1e-6)
    assert np.allclose(merged.samples, np.minimum(left, right))

    faces, points = tf.isosurface(merged, bias)
    assert closed_manifold(faces, points)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))


def test_resampled_volume_identity_reproduces_the_field():
    field = tf.sphere_sdf((16, 16, 16), (0.5,) * 3, (0.0,) * 3,
                          (4.0, 4.0, 4.0), 3.0)
    same = tf.resampled_volume(field, field.dims, field.spacing, field.origin)
    assert same.dims == field.dims
    np.testing.assert_array_equal(np.asarray(same.samples),
                                  np.asarray(field.samples))


def test_resampled_volume_tracks_the_analytic_field():
    center = np.array([4.0, 4.0, 4.0])
    field = tf.sphere_sdf((32, 32, 32), (0.25,) * 3, (0.0,) * 3,
                          tuple(center), 3.0)
    coarse = tf.resampled_volume(field, (16, 16, 16), (0.5,) * 3,
                                 (0.0, 0.0, 0.0))
    xs = np.arange(16) * 0.5
    grid = np.stack(np.meshgrid(xs, xs, xs, indexing="ij"), axis=-1)
    analytic = np.linalg.norm(grid - center, axis=-1) - 3.0
    got = np.asarray(coarse.samples)
    assert np.abs(got - analytic).max() < 0.08


def test_resampled_volume_emits_the_requested_dtype():
    field = tf.sphere_sdf((8, 8, 8), (1.0,) * 3, (0.0,) * 3,
                          (4.0, 4.0, 4.0), 2.0)
    fine = 1.0 + 2.0 ** -30  # representable in float64, not in float32
    wide = tf.resampled_volume(field, (8, 8, 8), (fine,) * 3, (0.0,) * 3,
                               dtype=np.float64)
    assert wide.dtype == np.float64
    assert wide.spacing == (fine,) * 3


def test_volume_transformation_mirrors_mesh():
    field = tf.sphere_sdf((8, 8, 8), (1.0,) * 3, (0.0,) * 3,
                          (4.0, 4.0, 4.0), 2.0)
    assert field.transformation is None
    pose = np.eye(4, dtype=np.float32)
    pose[0, 3] = 5.0
    field.transformation = pose
    got = np.asarray(field.transformation)
    np.testing.assert_array_equal(got, pose)
    field.transformation = None
    assert field.transformation is None
    with pytest.raises(ValueError):
        field.transformation = np.eye(3, dtype=np.float32)
    with pytest.raises(TypeError):
        field.transformation = np.eye(4, dtype=np.float64)


def test_posed_isosurface_emits_world_points():
    field = tf.sphere_sdf((12, 12, 12), (0.5,) * 3, (-2.75,) * 3,
                          (0.0, 0.0, 0.0), 2.0)
    _, local_points = tf.isosurface(field)
    pose = np.eye(4, dtype=np.float32)
    pose[0, 3] = 4.0
    field.transformation = pose
    _, posed_points = tf.isosurface(field)
    np.testing.assert_allclose(posed_points[:, 0], local_points[:, 0] + 4.0,
                               atol=1e-5)
    np.testing.assert_allclose(posed_points[:, 1:], local_points[:, 1:],
                               atol=1e-5)


def test_matched_pose_boolean_keeps_the_pose():
    a = tf.sphere_sdf((10, 10, 10), (0.5,) * 3, (0.0,) * 3,
                      (2.25, 2.25, 2.25), 1.5)
    b = tf.sphere_sdf((10, 10, 10), (0.5,) * 3, (0.0,) * 3,
                      (2.75, 2.25, 2.25), 1.5)
    flat = tf.volume_boolean(a, b, "union")
    pose = np.eye(4, dtype=np.float32)
    pose[1, 3] = -3.0
    a.transformation = pose
    b.transformation = pose
    posed = tf.volume_boolean(a, b, "union")
    np.testing.assert_array_equal(np.asarray(posed.samples),
                                  np.asarray(flat.samples))
    np.testing.assert_allclose(np.asarray(posed.transformation), pose,
                               atol=0.0)


def test_differently_posed_boolean_is_world_aligned():
    a = tf.sphere_sdf((10, 10, 10), (0.5,) * 3, (0.0,) * 3,
                      (2.25, 2.25, 2.25), 1.5)
    b = tf.sphere_sdf((10, 10, 10), (0.5,) * 3, (0.0,) * 3,
                      (2.25, 2.25, 2.25), 1.5)
    pose = np.eye(4, dtype=np.float32)
    pose[0, 3] = 1.0
    a.transformation = pose
    merged = tf.volume_boolean(a, b, "union")
    assert merged.transformation is None


def test_posed_resample_reads_through_the_pose():
    field = tf.sphere_sdf((12, 12, 12), (0.5,) * 3, (-2.75,) * 3,
                          (0.0, 0.0, 0.0), 2.0)
    pose = np.eye(4, dtype=np.float32)
    pose[2, 3] = 10.0
    field.transformation = pose
    world = tf.resampled_volume(field, (12, 12, 12), (0.5,) * 3,
                                (-2.75, -2.75, 7.25))
    # Node (5, 5, 5) sits at world (-0.25, -0.25, 9.75); the inverse pose
    # lands it at local (-0.25, -0.25, -0.25).
    expected = np.linalg.norm([-0.25, -0.25, -0.25]) - 2.0
    assert abs(float(world.samples[5, 5, 5]) - expected) < 0.08


def test_identity_pose_reads_back_as_none():
    field = tf.sphere_sdf((8, 8, 8), (1.0,) * 3, (0.0,) * 3,
                          (4.0, 4.0, 4.0), 2.0)
    field.transformation = np.eye(4, dtype=np.float32)
    assert field.transformation is None


def test_a_pose_that_only_rounds_to_the_identity_is_kept():
    # The native combine compares poses by exact matrix equality, so a
    # part-per-million scale is a pose and must survive the setter.
    field = tf.sphere_sdf((8, 8, 8), (1.0,) * 3, (0.0,) * 3,
                          (4.0, 4.0, 4.0), 2.0)
    scale = np.eye(4, dtype=np.float32)
    scale[0, 0] = np.float32(1.0 + 1e-6)
    scale[1, 1] = np.float32(1.0 + 1e-6)
    scale[2, 2] = np.float32(1.0 + 1e-6)
    field.transformation = scale
    assert field.transformation is not None
    np.testing.assert_array_equal(np.asarray(field.transformation), scale)


def test_posed_slice_contours_emit_world_points():
    field = tf.sphere_sdf((16, 16, 16), (0.5,) * 3, (-3.75,) * 3,
                          (0.0, 0.0, 0.0), 2.0)
    plane = (-3.75, -3.75, 0.0)
    _, local_points = tf.volume_slice_contours(
        field, plane, (1, 0, 0), (0, 1, 0), (32, 32), (0.25, 0.25), 0.0)
    pose = np.eye(4, dtype=np.float32)
    pose[1, 3] = 7.0
    field.transformation = pose
    _, posed_points = tf.volume_slice_contours(
        field, plane, (1, 0, 0), (0, 1, 0), (32, 32), (0.25, 0.25), 0.0)
    assert len(local_points) > 0
    assert len(posed_points) == len(local_points)
    np.testing.assert_allclose(posed_points[:, 1], local_points[:, 1] + 7.0,
                               atol=1e-5)
    np.testing.assert_allclose(posed_points[:, [0, 2]],
                               local_points[:, [0, 2]], atol=1e-5)


def test_mixed_posed_boolean_combines_on_the_world_grid():
    # One operand tagged, one not — the shape the binding produces when only
    # one Volume carries a transformation.
    a = tf.sphere_sdf((16, 16, 16), (0.5,) * 3, (-3.75,) * 3,
                      (0.0, 0.0, 0.0), 2.0)
    b = tf.sphere_sdf((16, 16, 16), (0.5,) * 3, (-3.75,) * 3,
                      (1.0, 0.0, 0.0), 1.5)
    pose = np.eye(4, dtype=np.float32)
    pose[0, 3] = 2.0
    a.transformation = pose

    merged = tf.volume_boolean(a, b, "union")
    assert merged.transformation is None

    samples = np.asarray(merged.samples)
    dims = merged.dims
    spacing = np.asarray(merged.spacing)
    origin = np.asarray(merged.origin)
    asserted = 0
    for i in range(0, dims[0], 3):
        for j in range(0, dims[1], 3):
            for k in range(0, dims[2], 3):
                p = origin + np.array([i, j, k]) * spacing
                la = p - np.array([2.0, 0.0, 0.0])
                # Inside both domains, away from either cone apex.
                if np.any(np.abs(la) > 3.0) or np.any(np.abs(p) > 3.0):
                    continue
                da = np.linalg.norm(la) - 2.0
                db = np.linalg.norm(p - np.array([1.0, 0.0, 0.0])) - 1.5
                if np.linalg.norm(la) < 1.0 or \
                        np.linalg.norm(p - np.array([1.0, 0.0, 0.0])) < 1.0:
                    continue
                asserted += 1
                assert abs(float(samples[i, j, k]) - min(da, db)) < 0.1
    assert asserted > 0
