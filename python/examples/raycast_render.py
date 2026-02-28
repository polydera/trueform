"""
Ray-traced depth rendering using batch ray casting

Loads a mesh, generates a grid of rays (one per pixel), batch ray_casts
them against the mesh, and renders Lambertian shading from face normals.

Usage:
    python raycast_render.py [mesh.stl] [--resolution 512]

Default mesh: dragon-250k.stl
"""

import os
import time
import argparse
import numpy as np
import trueform as tf

# TrueForm color scheme (matches VTK examples)
BG_COLOR = np.array([27, 43, 52], dtype=np.float32) / 255     # dark blue-gray
TEAL = np.array([0.0, 0.659, 0.604])                          # base teal
TEAL_BRIGHT = np.array([0.0, 0.835, 0.745])                   # highlight teal


def main():
    parser = argparse.ArgumentParser(description="Ray-traced mesh rendering")
    parser.add_argument("mesh", nargs="?", default=None,
                        help="Path to .stl mesh")
    parser.add_argument("--resolution", type=int,
                        default=512, help="Image resolution")
    args = parser.parse_args()

    data_dir = os.path.join(os.path.dirname(
        __file__), "../../benchmarks/data/")
    mesh_path = args.mesh or os.path.join(data_dir, "dragon-125k.stl")
    res = args.resolution

    # Load mesh
    print(f"Loading {mesh_path}")
    faces, points = tf.read_stl(mesh_path)
    mesh = tf.Mesh(faces, points)
    print(f"  {mesh.number_of_faces} faces, {mesh.number_of_points} vertices")
    mesh.build_tree()
    mesh.normals

    # Bounding box
    aabb_min = points.min(axis=0)
    aabb_max = points.max(axis=0)
    size = aabb_max - aabb_min
    pad = size.max() * 0.1

    # Orthographic camera looking down -Z
    x = np.linspace(aabb_min[0] - pad, aabb_max[0] +
                    pad, res).astype(points.dtype)
    y = np.linspace(aabb_max[1] + pad, aabb_min[1] -
                    pad, res).astype(points.dtype)
    gx, gy = np.meshgrid(x, y)
    n_rays = res * res

    origins = np.zeros((n_rays, 3), dtype=points.dtype)
    origins[:, 0] = gx.ravel()
    origins[:, 1] = gy.ravel()
    origins[:, 2] = aabb_max[2] + pad

    directions = np.zeros((n_rays, 3), dtype=points.dtype)
    directions[:, 2] = -1.0

    rays = tf.Ray(origin=origins, direction=directions)

    # Batch ray cast
    print(f"Casting {n_rays:,} rays ({res}x{res})...")
    t0 = time.perf_counter()
    ids, ts = tf.ray_cast(rays, mesh)
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"  {elapsed:.1f} ms")

    # Reshape
    ids = ids.reshape(res, res)
    ts = ts.reshape(res, res)
    hit = ~np.isnan(ts)
    print(f"  {hit.sum():,} hits ({100 * hit.sum() / n_rays:.1f}%)")

    # Two-light Lambertian shading
    normals = mesh.normals
    hit_ids = ids[hit]
    hit_normals = normals[hit_ids].copy()

    # Flip back-facing normals
    hit_normals[hit_normals[:, 2] > 0] *= -1

    key_dir = np.array([0.4, 0.6, -0.7], dtype=np.float32)
    key_dir /= np.linalg.norm(key_dir)
    fill_dir = np.array([-0.5, -0.3, -0.6], dtype=np.float32)
    fill_dir /= np.linalg.norm(fill_dir)

    key = np.clip(hit_normals @ key_dir, 0.0, 1.0)
    fill = np.clip(hit_normals @ fill_dir, 0.0, 1.0)
    diffuse = 0.08 + 0.72 * key + 0.20 * fill  # ambient + key + fill

    # Rim lighting — bright edge glow where surface is near-perpendicular to view
    view_dir = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    facing = np.abs(hit_normals @ view_dir)
    rim = (1.0 - facing) ** 3
    intensity = diffuse + 0.35 * rim

    # Teal-shaded image on dark background
    image = np.empty((res, res, 3), dtype=np.float32)
    image[:] = BG_COLOR

    # Shade from dark teal to bright teal by intensity
    color = TEAL + np.clip(intensity, 0, 1)[:, np.newaxis] * (TEAL_BRIGHT - TEAL)
    image[hit] = color * intensity[:, np.newaxis]

    image = np.clip(image, 0, 1)

    # Display
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), facecolor=BG_COLOR)
    ax.imshow(image)
    ax.set_axis_off()
    ax.set_title(
        f"{os.path.basename(mesh_path)} \u2014 {res}\u00d7{res}, {elapsed:.0f} ms",
        color="white", fontsize=13, pad=12,
    )
    fig.patch.set_facecolor(BG_COLOR)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
