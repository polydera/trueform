"""
Sidedness relations: select one side of a cut.

Two perpendicular planes intersect along a single line. The arrangement
cuts each plane in half along that line; four manifold-edge-connected
components result, one per quadrant. ``tf.sidedness_relations`` reports
which other operand each component touches at the cut and the
sidedness against that operand's oriented surface — letting us extract
"the half of mesh_a on the +normal side of mesh_b" in a few vectorized
numpy lines.

Run:
    python sidedness_relations.py

Requires: trueform, numpy
"""

import numpy as np
import trueform as tf


# -----------------------------------------------------------------------------
# 1. Two perpendicular planes (XY and XZ), 2x2 each.
# -----------------------------------------------------------------------------
faces_a, points_a = tf.make_plane_mesh(2.0, 2.0)

# Rotate the second plane 90 deg around X so it lies in the XZ plane.
# Its outward normal ends up pointing in -Y.
R = np.array([
    [1, 0,  0],
    [0, 0, -1],
    [0, 1,  0],
], dtype=points_a.dtype)
points_b = points_a @ R.T
faces_b = faces_a.copy()

mesh_a = tf.Mesh(faces_a, points_a)
mesh_b = tf.Mesh(faces_b, points_b)


# -----------------------------------------------------------------------------
# 2. Arrangement + sidedness relations in one call.
# -----------------------------------------------------------------------------
(faces, points), tag_labels, _ = tf.mesh_arrangements([mesh_a, mesh_b])
arr = tf.Mesh(faces, points)

(tags, sides), (n_components, comp_labels) = tf.sidedness_relations(
    arr, tag_labels)

print(f"arrangement: {len(faces)} faces, {len(points)} points")
print(f"components:  {n_components}")


# -----------------------------------------------------------------------------
# 3. Per-component sidedness vs the OTHER operand.
# -----------------------------------------------------------------------------
_SIDE_NAMES = {
    0: "+normal",   # tf::sidedness::on_positive_side
    1: "-normal",   # tf::sidedness::on_negative_side
    2: "boundary",  # tf::sidedness::on_boundary
    3: "none",      # tf::sidedness::none (selection-only sentinel)
}

# Per-component source tag (each component is single-source after MEL,
# so the scatter — last write wins — is well-defined).
comp_src = np.empty(n_components, dtype=tag_labels.dtype)
comp_src[comp_labels] = tag_labels

print("\nsidedness:")
for cid in range(n_components):
    rendered = ", ".join(
        f"vs tag {int(t)} -> {_SIDE_NAMES[int(s)]}"
        for t, s in zip(tags[cid], sides[cid])
    ) or "no contacts"
    print(f"  component {cid} (from tag {int(comp_src[cid])}): {rendered}")


# -----------------------------------------------------------------------------
# 4. Vectorized selection: mesh_a components on +normal side of mesh_b.
#
# Pattern:
#   1. Element-wise predicate over the flat (tags.data, sides.data).
#   2. Prefix-sum + difference at offsets = per-component count.
#   3. Boolean mask, then logical-AND with "source is mesh_a".
#
# This shape generalises to any (operand, sidedness) rule on an
# OffsetBlockedArray — drop in any boolean op, drop in any reduction.
# -----------------------------------------------------------------------------
MESH_B_TAG = 1
POSITIVE_SIDE = 0

matches = (tags.data == MESH_B_TAG) & (sides.data == POSITIVE_SIDE)
csum = np.concatenate([[0], np.cumsum(matches)])
match_count = csum[tags.offsets[1:]] - csum[tags.offsets[:-1]]

keep = (match_count > 0) & (comp_src == 0)   # mesh_a (tag 0) only

keep_components = np.where(keep)[0]
print(f"\n+normal side of mesh_b, restricted to mesh_a: {keep_components}")


# -----------------------------------------------------------------------------
# 5. Lift per-component mask → per-face mask → sub-mesh.
# -----------------------------------------------------------------------------
face_mask = np.isin(comp_labels, keep_components)
kept_faces, kept_points = tf.reindex_by_mask((faces, points), face_mask)

c = kept_points.mean(axis=0)
print(
    f"kept sub-mesh: faces={len(kept_faces)}, points={len(kept_points)}, "
    f"centroid=({c[0]:+.3f}, {c[1]:+.3f}, {c[2]:+.3f})"
)
