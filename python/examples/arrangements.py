"""
Booleans and domains from one CsgGraph build.

One scene -- a sphere straddling a knife plane declared a sheet, plus
two floaters that touch nothing -- queried three ways: the sides as
boolean meshes, the volumes individually via expression-selected
domains, and the same selection by hand from the inclusion matrix.

Mirrors examples/src/arrangements.cpp on the C++ side.

Run:
    python arrangements.py

Requires: trueform, numpy
"""

import numpy as np
import trueform as tf


def vol(faces, points):
    p = np.asarray(points, np.float64)
    f = np.asarray(faces)
    return float(np.einsum("ij,ij->", np.cross(p[f[:, 0]], p[f[:, 1]]),
                           p[f[:, 2]])) / 6.0


# -----------------------------------------------------------------------------
# 1. Geometry: a sphere straddling z=0, one floater above, one below,
#    and a 4x4 knife plane on z=0 with stored normal +Z.
# -----------------------------------------------------------------------------
def sphere(center, radius):
    sf, sp = tf.make_sphere_mesh(radius, 32, 32)
    return tf.Mesh(sf, np.asarray(sp) + np.asarray(center, sp.dtype))


plane_faces, plane_points = tf.make_plane_mesh(4.0, 4.0)
meshes = [sphere((0, 0, 0), 1.0),      # op 0: straddles the knife
          sphere((0, 0, 2), 0.5),      # op 1: floats above, touches nothing
          sphere((0, 0, -2), 0.5),     # op 2: floats below, touches nothing
          tf.Mesh(plane_faces, plane_points)]  # op 3: the knife


# -----------------------------------------------------------------------------
# 2. One build. Declaring the plane a sheet makes op(3) an oriented
#    separator: its operand bit means "behind the sheet's normal" (-Z
#    here), so the knife cuts volumes without enclosing one.
# -----------------------------------------------------------------------------
graph = tf.CsgGraph(meshes, sheets=[3])
solids = tf.op(0) | tf.op(1) | tf.op(2)


# -----------------------------------------------------------------------------
# 3. The boolean path: one mesh per side. Each comes back closed --
#    the straddler's half capped by the knife, the floater whole --
#    as disjoint pieces of a single mesh.
# -----------------------------------------------------------------------------
above_mesh = graph.mesh(solids - tf.op(3))
below_mesh = graph.mesh(solids & tf.op(3))
print("=== Boolean meshes ===")
for name, (mf, mp) in [("solids - knife", above_mesh),
                       ("solids & knife", below_mesh)]:
    m = tf.Mesh(mf, mp)
    print(f"  {name}: vol={vol(mf, mp):.4f} closed={tf.is_closed(m)}")


# -----------------------------------------------------------------------------
# 4. Domains by expression: the same volumes, individually.
# -----------------------------------------------------------------------------
above_cells, above_ids = graph.domains(solids - tf.op(3))
below_cells, below_ids = graph.domains(solids & tf.op(3))
print("\n=== Domains by expression ===")
print(f"  above: {len(above_cells)} cells, "
      f"vols {[round(vol(f, p), 4) for f, p in above_cells]}")
print(f"  below: {len(below_cells)} cells, "
      f"vols {[round(vol(f, p), 4) for f, p in below_cells]}")
# the pieces of the boolean mesh, one by one
assert abs(sum(vol(f, p) for f, p in above_cells) - vol(*above_mesh)) < 1e-9


# -----------------------------------------------------------------------------
# 5. Domains by hand: extract everything once, select by mask. The
#    knife's column is 3; behind its +Z normal = below.
# -----------------------------------------------------------------------------
cells, ids, imap = graph.domains(return_index_map=True)
below = imap.inclusion[:, 3]
above = ~below
print("\n=== Domains by hand ===")
print(f"  {len(cells)} cells; above {int(above.sum())}, below {int(below.sum())}")

# masks select the same cells as the expressions (ids are stable)
ids = np.asarray(ids)
assert sorted(ids[above].tolist()) == sorted(np.asarray(above_ids).tolist())
assert sorted(ids[below].tolist()) == sorted(np.asarray(below_ids).tolist())


# -----------------------------------------------------------------------------
# 6. Write and verify. Each cell is closed and manifold by construction.
# -----------------------------------------------------------------------------
def write_side(mask, prefix):
    k = 0
    for (faces, points), keep in zip(cells, mask):
        if not keep:
            continue
        m = tf.Mesh(faces, points)
        fname = f"{prefix}_{k}.stl"
        tf.write_stl(m, fname)
        print(f"  wrote {fname} (faces={len(faces)}, "
              f"closed={tf.is_closed(m)}, manifold={tf.is_manifold(m)})")
        k += 1


write_side(above, "above")
write_side(below, "below")
