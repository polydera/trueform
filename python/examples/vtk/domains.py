"""
Volumetric domain extraction with VTK — split-screen viewer

Mirrors tf-test/test_domains.cpp.  Two viewports:

  Left  — Full arrangement (semi-transparent), with non-manifold edge paths
          overlaid as yellow tubes (vtkTubeFilter).  Faces are colour-coded
          by the domain on side 0.
  Right — The currently selected domain submesh, opaque, outward-oriented.

Two preset scenes:

  Scene 0 — Sphere + bisecting plane + small sphere at z=-0.5
            (analog of test_domains.cpp case 7).
  Scene 1 — Sphere + 2 horizontal planes (z=±0.3) + 2 vertical planes
            (analog of case 4), with the outer shell excluded.

Both scenes run in double precision.

Controls
  Left / Right   Cycle current domain
  N              Cycle to the next scene
  Mouse drag     Rotate camera

Run
  python domains.py
"""

import sys

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
# Mesh construction helpers
# ----------------------------------------------------------------------------
def _sphere(radius):
    return tf.make_sphere_mesh(radius, 32, 32, dtype=DT, index_dtype=IDT)


def _plane(side):
    return tf.make_plane_mesh(side, side, dtype=DT, index_dtype=IDT)


def _translate(points, dx=0.0, dy=0.0, dz=0.0):
    out = points.copy()
    out[:, 0] += dx
    out[:, 1] += dy
    out[:, 2] += dz
    return out


def _swap_axes(points, perm):
    out = np.empty_like(points)
    for new_i, old_i in enumerate(perm):
        out[:, new_i] = points[:, old_i]
    return out


# ----------------------------------------------------------------------------
# Scenes  — each returns (name, list[tf.Mesh], domain_labels_kwargs)
# ----------------------------------------------------------------------------
def scene_sphere_plane_inner_below():
    """Case 7: big sphere + plane at z=0 + small sphere at z=-0.5"""
    bf, bp = _sphere(1.0)
    pf, pp = _plane(3.0)
    sf, sp = _sphere(0.2)
    sp = _translate(sp, dz=-0.5)
    meshes = [tf.Mesh(bf, bp), tf.Mesh(pf, pp), tf.Mesh(sf, sp)]
    return ("sphere + plane + small sphere (z=-0.5)",
            meshes,
            {"ignore_open_fragments": True})


def scene_sphere_four_planes():
    """Case 4: sphere + 4 planes (2 horizontal + 2 vertical)"""
    sf, sp = _sphere(1.0)
    p_top_f, p_top_p = _plane(3.0)
    p_top_p = _translate(p_top_p, dz=0.3)
    p_bot_f, p_bot_p = _plane(3.0)
    p_bot_p = _translate(p_bot_p, dz=-0.3)
    p_xz_f, p_xz_p = _plane(3.0)
    p_xz_p = _swap_axes(p_xz_p, (0, 2, 1))   # XY → XZ (+y normal)
    p_yz_f, p_yz_p = _plane(3.0)
    p_yz_p = _swap_axes(p_yz_p, (2, 1, 0))   # XY → YZ (+x normal)
    meshes = [
        tf.Mesh(sf, sp),
        tf.Mesh(p_top_f, p_top_p),
        tf.Mesh(p_bot_f, p_bot_p),
        tf.Mesh(p_xz_f, p_xz_p),
        tf.Mesh(p_yz_f, p_yz_p),
    ]
    return ("sphere + 4 planes",
            meshes,
            {"ignore_open_fragments": True})


SCENES = [
    scene_sphere_plane_inner_below,
    scene_sphere_four_planes,
]


# ----------------------------------------------------------------------------
# Pipeline:  arrangements → cleaned → domain_labels → split_into_domains
# ----------------------------------------------------------------------------
def build_arrangement_and_domains(meshes, flag_kwargs):
    (af, ap), _tags, _faces = tf.mesh_arrangements(meshes)
    af, ap = tf.cleaned((af, ap), tolerance=1e-6)
    arrangement = tf.Mesh(af, ap)
    dl = tf.domain_labels(arrangement, **flag_kwargs)
    components, comp_labels = tf.split_into_domains(arrangement, dl)
    return arrangement, dl, components, comp_labels


# ----------------------------------------------------------------------------
# VTK helpers
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


def _color(i):
    return PALETTE[i % len(PALETTE)]


def aabb_diagonal(points):
    mn = points.min(axis=0)
    mx = points.max(axis=0)
    return float(np.linalg.norm(mx - mn))


def make_tubes_actor(paths, points, color=(1.0, 0.85, 0.0)):
    """tf.make_tube_mesh around the paths.  Radius = 0.1% of AABB diagonal."""
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
    actor.GetProperty().LightingOff()
    return actor


_PALETTE_U8 = (PALETTE * 255).astype(np.uint8)
_SENTINEL_RGB = np.array([77, 77, 82], dtype=np.uint8)


def make_arrangement_actor(faces, points):
    """Full arrangement actor; per-face RGBA cell-data is installed later
    by apply_highlight() based on the currently selected domain."""
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
    """Vectorized per-face RGBA cell-data: faces bounding `current_domain`
    on either side go to full alpha in the domain color; the rest are
    ghosted (alpha = 30) in the color of their first valid side, or in a
    neutral sentinel grey if both sides are sentinel-bound."""
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
# Interactor
# ----------------------------------------------------------------------------
class DomainsInteractor(BaseInteractor):
    """Left/Right cycle the current domain.  Up/Down cycle scenes."""

    def __init__(self, left_ren, right_ren, text_actor):
        super().__init__()
        self.left_ren = left_ren
        self.right_ren = right_ren
        self.text_actor = text_actor

        self.scene_idx = 0
        self.domain_idx = 0
        self.left_actors = []
        self.right_actors = []

        # Scene state populated by load_current_scene().
        self.components = []
        self.comp_labels = np.zeros(0, dtype=np.int32)
        self.n_domains = 0
        self.labels_2d = None
        self.left_poly = None

        self.load_current_scene()

        self.AddObserver("KeyPressEvent", self.on_key_press)

    def clear(self):
        for a in self.left_actors:
            self.left_ren.RemoveActor(a)
        for a in self.right_actors:
            self.right_ren.RemoveActor(a)
        self.left_actors = []
        self.right_actors = []

    def load_current_scene(self):
        self.clear()
        name, meshes, flag_kwargs = SCENES[self.scene_idx]()
        arrangement, dl, comps, ids = build_arrangement_and_domains(
            meshes, flag_kwargs,
        )
        labels_2d, n_domains, _ = dl
        self.labels_2d = labels_2d
        self.components = comps
        self.comp_labels = ids
        self.n_domains = int(n_domains)
        self.domain_idx = 0
        self.scene_name = name
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

        # ---- Right viewport: current domain submesh ----
        self.refresh_current()

        self.left_ren.ResetCamera()
        self.right_ren.ResetCamera()
        if self.GetInteractor() is not None:
            self.GetInteractor().Render()

    def refresh_current(self):
        # Right viewport
        for a in self.right_actors:
            self.right_ren.RemoveActor(a)
        self.right_actors = []
        if self.components:
            faces, points = self.components[self.domain_idx]
            label = int(self.comp_labels[self.domain_idx])
            actor = make_domain_actor(faces, points, _color(label))
            self.right_actors.append(actor)
            self.right_ren.AddActor(actor)
        # Left viewport — re-highlight against the current domain
        if self.left_poly is not None and self.labels_2d is not None \
                and self.components:
            label = int(self.comp_labels[self.domain_idx])
            apply_highlight(self.left_poly, self.labels_2d, label,
                            self.n_domains)
        self.update_text()

    def update_text(self):
        if len(self.components) == 0:
            self.text_actor.SetInput(f"[{self.scene_idx + 1}/{len(SCENES)}] "
                                     f"{self.scene_name}  —  no domains")
            return
        label = int(self.comp_labels[self.domain_idx])
        self.text_actor.SetInput(
            f"[{self.scene_idx + 1}/{len(SCENES)}] {self.scene_name}  —  "
            f"domain {label}  ({self.domain_idx + 1}/{len(self.components)} "
            f"of n_domains={self.n_domains})    "
            f"[Left/Right: cycle domain   Up/Down: cycle scene]"
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
        if key in ("Up", "Down"):
            delta = 1 if key == "Up" else -1
            self.scene_idx = (self.scene_idx + delta) % len(SCENES)
            self.load_current_scene()
            return
        vtk.vtkInteractorStyleTrackballCamera.OnKeyPress(self)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    left_ren = vtk.vtkRenderer()
    left_ren.SetViewport(0.0, 0.0, 0.5, 1.0)
    left_ren.SetBackground(0.15, 0.10, 0.10)

    right_ren = vtk.vtkRenderer()
    right_ren.SetViewport(0.5, 0.0, 1.0, 1.0)
    right_ren.SetBackground(0.08, 0.12, 0.08)

    # Overlay renderer just for the status text strip across the top.
    text_ren = vtk.vtkRenderer()
    text_ren.SetViewport(0.0, 0.93, 1.0, 1.0)
    text_ren.SetBackground(0.05, 0.05, 0.05)
    text_ren.InteractiveOff()
    text = create_text_actor(
        "loading…",
        font_size=22,
        position=(0.02, 0.30),
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

    style = DomainsInteractor(left_ren, right_ren, text)
    interactor.SetInteractorStyle(style)

    render_window.Render()
    interactor.Start()


if __name__ == "__main__":
    sys.exit(main() or 0)
