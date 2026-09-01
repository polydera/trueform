"""
Boolean operations benchmark with MeshLib (Python)

Measures time to compute boolean union on two overlapping
copies of the same mesh (same setup as C++ benchmarks).
MeshLib uses mr.boolean with int32 exact predicates and SoS.

Copyright (c) 2025 Ziga Sajovic, XLAB
"""

import math
import os
import sys
import time

from meshlib import mrmeshpy as mr

BENCHMARK_MESHES = [
    "data/dragon-50k.stl",
    "data/dragon-125k.stl",
    "data/dragon-250k.stl",
    "data/dragon-500k.stl",
    "data/dragon-750k.stl",
    "data/dragon-1M.stl",
]

N_SAMPLES = 10


def make_rotation(angle_deg, axis_idx, center):
    angle = math.radians(angle_deg)
    if axis_idx == 0:
        axis = mr.Vector3f(1, 0, 0)
    elif axis_idx == 1:
        axis = mr.Vector3f(0, 1, 0)
    else:
        axis = mr.Vector3f(0, 0, 1)
    rot = mr.AffineXf3f.linear(mr.Matrix3f.rotation(axis, angle))
    to_origin = mr.AffineXf3f.translation(-center)
    from_origin = mr.AffineXf3f.translation(center)
    return from_origin * rot * to_origin


def run_benchmark(mesh_path, n_samples):
    mesh = mr.loadMesh(mesh_path)
    n_faces = mesh.topology.numValidFaces()

    bbox = mesh.computeBoundingBox()
    center = (bbox.min + bbox.max) * 0.5
    diag = bbox.max - bbox.min
    inv = [1.0 / diag.x, 1.0 / diag.y, 1.0 / diag.z]
    rot_axis = inv.index(max(inv))

    total_ms = 0.0
    for i in range(n_samples):
        angle = 360.0 * (i + 0.5) / n_samples
        xf = make_rotation(angle, rot_axis, center)

        mesh2 = mr.Mesh(mesh)
        mesh2.transform(xf)

        t0 = time.perf_counter()
        result = mr.boolean(mesh, mesh2, mr.BooleanOperation.Union)
        t1 = time.perf_counter()
        total_ms += (t1 - t0) * 1000.0

    mean_ms = total_ms / n_samples
    return n_faces, mean_ms


def main():
    print("polygons0,polygons1,time_ms")
    for mesh_rel in BENCHMARK_MESHES:
        mesh_path = os.path.join(os.getcwd(), mesh_rel)
        if not os.path.exists(mesh_path):
            print(f"# skipping {mesh_rel} (not found)", file=sys.stderr)
            continue
        print(f"Running: {mesh_rel}...", file=sys.stderr)
        n_faces, mean_ms = run_benchmark(mesh_path, N_SAMPLES)
        print(f"{n_faces},{n_faces},{mean_ms:.2f}")
        print(f"  {n_faces} polygons: {mean_ms:.2f} ms", file=sys.stderr)
        sys.stdout.flush()


if __name__ == "__main__":
    main()
