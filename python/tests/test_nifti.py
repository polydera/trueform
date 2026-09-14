"""
NIfTI IO tests

Copyright (c) 2025 Žiga Sajovic, XLAB
Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
Commercial licensing available via info@polydera.com.
https://github.com/polydera/trueform
"""

import numpy as np
import pytest

import trueform as tf


def _sphere_samples(dtype, scale=1.0):
    x, y, z = np.meshgrid(np.arange(12.0), np.arange(12.0), np.arange(12.0),
                          indexing="ij")
    sdf = np.sqrt((x - 5.5) ** 2 + (y - 5.5) ** 2 + (z - 5.5) ** 2) - 4.0
    return (sdf * scale).astype(dtype)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int16,
                                   np.uint16, np.uint8])
@pytest.mark.parametrize("suffix", [".nii", ".nii.gz"])
def test_nifti_native_round_trip(tmp_path, dtype, suffix):
    samples = _sphere_samples(dtype, scale=10.0 if dtype != np.uint8 else 1.0)
    vol = tf.Volume(samples, spacing=(0.7, 0.7, 1.5))
    path = tmp_path / f"field{suffix}"
    tf.write_nifti(vol, path)

    header = tf.read_nifti_header(path)
    assert header.dtype == np.dtype(dtype)
    assert header.dims == (12, 12, 12)
    assert not header.posed

    back = tf.read_nifti(path)
    assert back.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(np.asarray(back.samples), samples)
    np.testing.assert_allclose(back.spacing, (0.7, 0.7, 1.5), atol=1e-6)
    assert back.transformation is None


def test_nifti_posed_round_trip(tmp_path):
    vol = tf.Volume(_sphere_samples(np.float32))
    # A pure translation is absorbed into the origin; a rotation cannot be,
    # so the file states the pose.
    c, s = np.cos(np.pi / 6), np.sin(np.pi / 6)
    pose = np.array(
        [[c, -s, 0.0, 4.5],
         [s, c, 0.0, -2.25],
         [0.0, 0.0, 1.0, 0.0],
         [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    vol.transformation = pose
    path = tmp_path / "posed.nii.gz"
    tf.write_nifti(vol, path)

    header = tf.read_nifti_header(path)
    assert header.posed
    assert not header.reflecting

    back = tf.read_nifti(path)
    assert back.transformation is not None
    np.testing.assert_allclose(np.asarray(back.transformation), pose,
                               atol=1e-5)


def test_nifti_reflecting_pose_is_named(tmp_path):
    vol = tf.Volume(_sphere_samples(np.float32))
    mirror = np.eye(4, dtype=np.float32)
    mirror[0, 0] = -1.0
    vol.transformation = mirror
    path = tmp_path / "mirror.nii"
    tf.write_nifti(vol, path)
    header = tf.read_nifti_header(path)
    assert header.posed
    assert header.reflecting


def test_nifti_dtype_converts_on_read(tmp_path):
    samples = _sphere_samples(np.int16, scale=10.0)
    path = tmp_path / "ct.nii"
    tf.write_nifti(tf.Volume(samples), path)
    as_float = tf.read_nifti(path, dtype=np.float32)
    assert as_float.dtype == np.dtype(np.float32)
    np.testing.assert_allclose(np.asarray(as_float.samples),
                               samples.astype(np.float32), atol=0.0)


def test_nifti_refusal_names_its_fact(tmp_path):
    path = tmp_path / "garbage.nii"
    path.write_bytes(b"not a nifti file at all, far too short a header")
    with pytest.raises(ValueError):
        tf.read_nifti(str(path))
    missing = tmp_path / "absent.nii"
    with pytest.raises(ValueError):
        tf.read_nifti_header(str(missing))


def test_nifti_scan_to_patient_space_mesh(tmp_path):
    vol = tf.Volume(_sphere_samples(np.int16, scale=10.0))
    pose = np.eye(4, dtype=np.float32)
    pose[0, 3] = 100.0
    vol.transformation = pose
    path = tmp_path / "scan.nii.gz"
    assert tf.write_nifti(vol, path)

    scan = tf.read_nifti(path)
    # A pure translation is one the axis-aligned grid can hold, so the file
    # states no pose and the origin carries it instead.
    assert scan.transformation is None
    np.testing.assert_allclose(scan.origin, (100.0, 0.0, 0.0), atol=1e-5)
    faces, points = tf.isosurface(scan, iso=0)
    assert len(faces) > 0
    center = points.mean(axis=0)
    assert abs(center[0] - 105.5) < 0.5
    assert abs(center[1] - 5.5) < 0.5


def _world_point(volume, index):
    """Where a sample stands in world space: the pose applied to its grid point."""
    local = np.asarray(volume.origin) + np.asarray(index, dtype=float) * \
        np.asarray(volume.spacing)
    pose = volume.transformation
    if pose is None:
        return local
    pose = np.asarray(pose, dtype=float)
    return pose[:3, :3] @ local + pose[:3, 3]


def test_nifti_folds_the_origin_into_the_pose(tmp_path):
    vol = tf.Volume(_sphere_samples(np.float32), spacing=(0.5, 0.5, 0.5),
                    origin=(2.0, 3.0, -4.0))
    c, s = np.cos(np.pi / 5), np.sin(np.pi / 5)
    vol.transformation = np.array(
        [[c, -s, 0.0, 1.5],
         [s, c, 0.0, 0.0],
         [0.0, 0.0, 1.0, -2.0],
         [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    path = tmp_path / "posed_origin.nii"
    assert tf.write_nifti(vol, path)

    back = tf.read_nifti(path)
    # The file states one placement and the read splits it its own way: the
    # origin folds into the pose, so the composed world position is what
    # round-trips, not the split.
    assert back.transformation is not None
    np.testing.assert_allclose(back.origin, (0.0, 0.0, 0.0), atol=1e-5)
    for index in [(0, 0, 0), (11, 0, 0), (5, 7, 3), (11, 11, 11)]:
        np.testing.assert_allclose(_world_point(back, index),
                                   _world_point(vol, index), atol=1e-4)


def test_nifti_refuses_a_dtype_it_cannot_read(tmp_path):
    path = tmp_path / "field.nii"
    assert tf.write_nifti(tf.Volume(_sphere_samples(np.float32)), path)
    with pytest.raises(ValueError) as refusal:
        tf.read_nifti(path, dtype=np.int32)
    assert "float32" in str(refusal.value)
    assert "int16" in str(refusal.value)
