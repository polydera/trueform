"""
Working with arrangements: domains, selection, and signed cuts.

Builds two overlapping cubes (closed) and a bisecting plane (open),
computes the arrangement, partitions it into watertight bounded
domains, and uses the plane's stored winding to split the resulting
volumes into "above" and "below" sets.

Mirrors examples/src/arrangements.cpp on the C++ side.

Run:
    python arrangements.py

Requires: trueform, numpy
"""

import numpy as np
import trueform as tf


# -----------------------------------------------------------------------------
# 1. Geometry: two overlapping cubes (closed) and a bisecting plane (open).
#    Cube 0 sits at x=-0.5, cube 1 at x=+0.5, plane lies on z=0 with stored
#    normal +Z (this is the winding `tf.make_plane_mesh` produces).
# -----------------------------------------------------------------------------
cube_faces, cube_points = tf.make_box_mesh(2.0, 2.0, 2.0)
plane_faces, plane_points = tf.make_plane_mesh(4.0, 4.0)

cube0 = tf.Mesh(cube_faces.copy(), cube_points + np.array([-0.5, 0, 0], dtype=cube_points.dtype))
cube1 = tf.Mesh(cube_faces.copy(), cube_points + np.array([+0.5, 0, 0], dtype=cube_points.dtype))
knife = tf.Mesh(plane_faces, plane_points)   # tag 2


# -----------------------------------------------------------------------------
# 2. Arrangement: split every face at every intersection and merge into a
#    single triangle mesh. tag_labels[f] records which input operand face f
#    came from (0, 1, or 2 here).
# -----------------------------------------------------------------------------
(arr_faces, arr_points), tag_labels, face_labels = tf.mesh_arrangements(
    [cube0, cube1, knife])

print("=== Arrangement ===")
print(f"Faces:  {len(arr_faces)}")
print(f"Points: {len(arr_points)}")


# -----------------------------------------------------------------------------
# 3. Clean coincident vertices. The per-face tag_labels array goes stale
#    when cleaning drops duplicate faces, so we reindex it through the
#    face index map (kept_faces gives the surviving old ids in new order).
# -----------------------------------------------------------------------------
(arr_faces, arr_points), (face_f, kept_faces), _ = tf.cleaned(
    (arr_faces, arr_points), 1e-6, return_index_map=True)
tag_labels = tag_labels[kept_faces]
arr = tf.Mesh(arr_faces, arr_points)

print("\n=== Cleaned ===")
print(f"Faces:  {len(arr_faces)}")
print(f"Points: {len(arr_points)}")


# -----------------------------------------------------------------------------
# 4. Domain labels. ignore_open_fragments parks the plane's open outer ring
#    at the sentinel; exclude_outer_shell folds the unbounded universe into
#    the same sentinel. What remains is the bounded interior domains.
# -----------------------------------------------------------------------------
labels_2d, n_domains, _ = tf.domain_labels(
    arr, ignore_open_fragments=True, exclude_outer_shell=True)
print("\n=== Domain labels ===")
print(f"Bounded domains: {n_domains}")


# -----------------------------------------------------------------------------
# 5. Split into per-domain watertight outward-oriented submeshes.
#    comp_labels[i] is the domain id of volumes[i].
# -----------------------------------------------------------------------------
volumes, comp_labels = tf.split_into_domains(arr, (labels_2d, n_domains))
print(f"Volumes extracted: {len(volumes)}")


# -----------------------------------------------------------------------------
# 6. Split the volumes by signed side of the knife.
#
#    labels[f, 0] = domain containing face f with reversed winding (the side
#                   f's stored normal points INTO).
#    labels[f, 1] = domain containing face f with forward winding.
#
#    The knife has tag 2 with stored normal +Z, so slot 0 yields the "above"
#    domain and slot 1 yields the "below" domain. Across all interior knife
#    faces we collect the unique domain ids — there can be multiple per side
#    when the knife cuts through several closed regions (here: cube0-only,
#    intersection, cube1-only).
# -----------------------------------------------------------------------------
knife_mask = tag_labels == 2
above = labels_2d[knife_mask, 0]
below = labels_2d[knife_mask, 1]
inside = (above < n_domains) & (below < n_domains)
above_ids = np.unique(above[inside])
below_ids = np.unique(below[inside])

print("\n=== Signed side of the knife ===")
print(f"Above (+normal): {len(above_ids)} volumes")
print(f"Below (-normal): {len(below_ids)} volumes")


# -----------------------------------------------------------------------------
# 7. Write each side's volumes to disk for visualisation.
# -----------------------------------------------------------------------------
domain_to_idx = np.full(n_domains, -1, dtype=np.int64)
domain_to_idx[comp_labels] = np.arange(len(comp_labels))


def write_side(ids, prefix):
    for k, d in enumerate(ids):
        vf, vp = volumes[int(domain_to_idx[d])]
        m = tf.Mesh(vf, vp)
        fname = f"{prefix}_{k}.stl"
        tf.write_stl(m, fname)
        print(f"  wrote {fname} "
              f"(faces={len(vf)}, "
              f"closed={tf.is_closed(m)}, "
              f"manifold={tf.is_manifold(m)})")


write_side(above_ids, "above")
write_side(below_ids, "below")
