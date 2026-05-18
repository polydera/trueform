"""
Volumetric domain extraction from a set of input OBJ files — VTK
split-screen viewer.

Usage:

  # Load every .obj in a directory
  python domains_from_files.py /path/to/inputs/

  # Load specific files
  python domains_from_files.py foo.obj bar.obj baz.obj

Pipeline:
  1. Load each .obj.
  2. mesh_arrangements over the list.
  3. cleaned.
  4. domain_labels (ignore_open_fragments).
  5. split_into_domains.

Viewer:
  Left  — Full arrangement, ghosted except faces bounding the current
          domain. Non-manifold edges drawn as yellow tubes.
  Right — Current domain submesh, opaque, outward-oriented.

Controls
  Left / Right    Cycle current domain
  Mouse drag      Rotate camera
"""

import os
import sys
import time

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk

import trueform as tf

from util import (
    BaseInteractor,
    create_text_actor,
    numpy_to_polydata,
)


DT = np.float64
IDT = np.int32


# ----------------------------------------------------------------------------
# CLI input → list of (faces, points) meshes
# ----------------------------------------------------------------------------
def _collect_obj_paths(args):
    """Accept either a single directory or a list of file paths."""
    if len(args) == 1 and os.path.isdir(args[0]):
        d = args[0]
        return sorted(
            os.path.join(d, f) for f in os.listdir(d)
            if f.lower().endswith(".obj")
        )
    return list(args)


def _load_meshes(paths):
    meshes = []
    t0 = time.perf_counter()
    for p in paths:
        ts = time.perf_counter()
        faces, points = tf.read_obj(p, ngon=3, dtype=DT, index_dtype=IDT)
        meshes.append(tf.Mesh(faces, points))
        print(f"  loaded {os.path.basename(p)}: "
              f"{len(points)} pts, {len(faces)} tris  "
              f"({(time.perf_counter() - ts) * 1000:.1f} ms)")
    print(f"  total read_obj: {(time.perf_counter() - t0) * 1000:.1f} ms")
    return meshes


# ----------------------------------------------------------------------------
# Pipeline: mesh_arrangements → cleaned → domain_labels → split_into_domains
# ----------------------------------------------------------------------------
def build_arrangement_and_domains(meshes):
    print(f"running mesh_arrangements on {len(meshes)} meshes...")
    t0 = time.perf_counter()
    (af, ap), _tags, _faces = tf.mesh_arrangements(meshes)
    print(f"  arrangement: {len(ap)} pts, {len(af)} tris  "
          f"({(time.perf_counter() - t0) * 1000:.1f} ms)")

    print("running cleaned...")
    t0 = time.perf_counter()
    af, ap = tf.cleaned((af, ap), tolerance=1e-6)
    arrangement = tf.Mesh(af, ap)
    print(f"  cleaned: {len(ap)} pts, {len(af)} tris  "
          f"({(time.perf_counter() - t0) * 1000:.1f} ms)")

    print("running domain_labels...")
    t0 = time.perf_counter()
    dl = tf.domain_labels(arrangement, ignore_open_fragments=True)
    print(f"  {dl[1]} domains  ({(time.perf_counter() - t0) * 1000:.1f} ms)")

    print("running split_into_domains...")
    t0 = time.perf_counter()
    components, comp_labels = tf.split_into_domains(arrangement, dl)
    print(f"  {len(components)} components  "
          f"({(time.perf_counter() - t0) * 1000:.1f} ms)")

    # Display order: outer shell (most negative signed_volume) prepended,
    # everyone else sorted by max-Z descending (top→bottom).
    if components:
        volumes = np.array([tf.signed_volume(c) for c in components])
        max_z = np.array([points[:, 2].max() for (_, points) in components])
        outer_idx = int(np.argmin(volumes))
        other_idx = np.array([i for i in range(len(components))
                              if i != outer_idx])
        z_order = np.argsort(-max_z[other_idx])
        order = np.concatenate(([outer_idx], other_idx[z_order])).astype(int)
        components = [components[i] for i in order]
        comp_labels = comp_labels[order]

    return arrangement, dl, components, comp_labels


# ----------------------------------------------------------------------------
# VTK helpers (palette + actors + highlight)
# ----------------------------------------------------------------------------
PALETTE = np.array([
    [0.85, 0.20, 0.20],
    [0.20, 0.40, 0.85],
    [0.20, 0.75, 0.20],
    [0.95, 0.75, 0.10],
    [0.75, 0.30, 0.85],
    [0.20, 0.85, 0.85],
    [0.95, 0.55, 0.10],
    [0.55, 0.55, 0.60],
], dtype=np.float64)
_PALETTE_U8 = (PALETTE * 255).astype(np.uint8)
_SENTINEL_RGB = np.array([77, 77, 82], dtype=np.uint8)


def _color(i):
    return PALETTE[i % len(PALETTE)]


def aabb_diagonal(points):
    mn = points.min(axis=0)
    mx = points.max(axis=0)
    return float(np.linalg.norm(mx - mn))


def make_tubes_actor(paths, points, color=(1.0, 0.85, 0.0)):
    radius = max(aabb_diagonal(points) * 0.001, 1e-6)
    tube_faces, tube_points = tf.make_tube_mesh(
        (paths, points), radius=radius, radial_segments=12,
    )
    poly = numpy_to_polydata(tube_points.astype(np.float64), tube_faces)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(poly)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(*color)
    actor.GetProperty().SetOpacity(0.6)
    actor.GetProperty().LightingOff()
    return actor


def make_arrangement_actor(faces, points):
    poly = numpy_to_polydata(points.astype(np.float64), faces)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(poly)
    mapper.SetScalarModeToUseCellData()
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().EdgeVisibilityOn()
    actor.GetProperty().SetEdgeColor(0.0, 0.0, 0.0)
    actor.GetProperty().SetLineWidth(0.5)
    return actor, poly


def apply_highlight(poly, labels_2d, current_domain, n_domains):
    d0 = labels_2d[:, 0].astype(np.int64)
    d1 = labels_2d[:, 1].astype(np.int64)
    in_dom = (d0 == current_domain) | (d1 == current_domain)
    base = np.where(d0 < n_domains, d0, d1)
    base_safe = np.where(base < n_domains, base, 0)
    base_color = _PALETTE_U8[base_safe % len(_PALETTE_U8)]
    no_valid = (d0 >= n_domains) & (d1 >= n_domains)
    base_color[no_valid] = _SENTINEL_RGB
    cur_color = _PALETTE_U8[current_domain % len(_PALETTE_U8)]
    rgba = np.empty((d0.size, 4), dtype=np.uint8)
    rgba[..., :3] = np.where(in_dom[:, None], cur_color, base_color)
    rgba[..., 3] = np.where(in_dom, 255, 30)
    vtk_colors = numpy_to_vtk(rgba, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
    poly.GetCellData().SetScalars(vtk_colors)
    poly.GetCellData().Modified()


def make_domain_actor(faces, points, color):
    poly = numpy_to_polydata(points.astype(np.float64), faces)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(poly)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(*color)
    actor.GetProperty().EdgeVisibilityOn()
    actor.GetProperty().SetEdgeColor(0.0, 0.0, 0.0)
    actor.GetProperty().SetLineWidth(1.0)
    return actor


# ----------------------------------------------------------------------------
# Interactor — Left/Right cycle current domain
# ----------------------------------------------------------------------------
class DomainsInteractor(BaseInteractor):
    def __init__(self, left_ren, right_ren, text_actor, meshes, scene_name):
        super().__init__()
        self.left_ren = left_ren
        self.right_ren = right_ren
        self.text_actor = text_actor
        self.scene_name = scene_name
        self.domain_idx = 0
        self.left_actors = []
        self.right_actors = []

        arrangement, dl, comps, ids = build_arrangement_and_domains(meshes)
        labels_2d, n_domains, _ = dl
        self.labels_2d = labels_2d
        self.components = comps
        self.comp_labels = ids
        self.n_domains = int(n_domains)
        self.arrangement = arrangement

        # ---- Left viewport: arrangement + NM-edge tubes ----
        actor, poly = make_arrangement_actor(
            arrangement.faces, arrangement.points,
        )
        self.left_poly = poly
        self.left_actors.append(actor)
        nm_edges = tf.non_manifold_edges(arrangement)
        if len(nm_edges) > 0:
            paths = tf.connect_edges_to_paths(nm_edges)
            self.left_actors.append(
                make_tubes_actor(paths, arrangement.points)
            )
        for a in self.left_actors:
            self.left_ren.AddActor(a)

        self.refresh_current()
        self.left_ren.ResetCamera()
        self.right_ren.ResetCamera()

        self.AddObserver("KeyPressEvent", self.on_key_press)

    def refresh_current(self):
        for a in self.right_actors:
            self.right_ren.RemoveActor(a)
        self.right_actors = []
        if self.components:
            faces, points = self.components[self.domain_idx]
            label = int(self.comp_labels[self.domain_idx])
            actor = make_domain_actor(faces, points, _color(label))
            self.right_actors.append(actor)
            self.right_ren.AddActor(actor)
            self.right_ren.ResetCamera()
        if self.left_poly is not None and self.components:
            label = int(self.comp_labels[self.domain_idx])
            apply_highlight(self.left_poly, self.labels_2d, label,
                            self.n_domains)
        self.update_text()

    def update_text(self):
        if not self.components:
            self.text_actor.SetInput(f"{self.scene_name}  —  no domains")
            return
        label = int(self.comp_labels[self.domain_idx])
        self.text_actor.SetInput(
            f"{self.scene_name}  —  domain {label}  "
            f"({self.domain_idx + 1}/{len(self.components)} "
            f"of n_domains={self.n_domains})    "
            f"[Left/Right: cycle domain]"
        )

    def on_key_press(self, obj, event):
        key = self.GetInteractor().GetKeySym()
        if key == "Right" and self.components:
            self.domain_idx = (self.domain_idx + 1) % len(self.components)
            self.refresh_current()
            self.GetInteractor().Render()
            return
        if key == "Left" and self.components:
            self.domain_idx = (self.domain_idx - 1) % len(self.components)
            self.refresh_current()
            self.GetInteractor().Render()
            return
        vtk.vtkInteractorStyleTrackballCamera.OnKeyPress(self)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main(argv):
    if len(argv) < 1:
        print(__doc__)
        return 1
    paths = _collect_obj_paths(argv)
    if not paths:
        print("no .obj files found")
        return 1
    print(f"loading {len(paths)} files...")
    meshes = _load_meshes(paths)
    scene_name = (os.path.basename(os.path.normpath(argv[0]))
                  if len(argv) == 1 and os.path.isdir(argv[0])
                  else f"{len(paths)} files")

    left_ren = vtk.vtkRenderer()
    left_ren.SetViewport(0.0, 0.0, 0.5, 1.0)
    left_ren.SetBackground(0.15, 0.10, 0.10)

    right_ren = vtk.vtkRenderer()
    right_ren.SetViewport(0.5, 0.0, 1.0, 1.0)
    right_ren.SetBackground(0.08, 0.12, 0.08)

    text_ren = vtk.vtkRenderer()
    text_ren.SetViewport(0.0, 0.93, 1.0, 1.0)
    text_ren.SetBackground(0.05, 0.05, 0.05)
    text_ren.InteractiveOff()
    text = create_text_actor(
        "loading…", font_size=22, position=(0.02, 0.30),
        justification="left",
    )
    text_ren.AddViewProp(text)

    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(left_ren)
    render_window.AddRenderer(right_ren)
    render_window.AddRenderer(text_ren)
    render_window.SetSize(1600, 900)

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)

    style = DomainsInteractor(left_ren, right_ren, text, meshes, scene_name)
    interactor.SetInteractorStyle(style)

    render_window.Render()
    interactor.Start()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]) or 0)
