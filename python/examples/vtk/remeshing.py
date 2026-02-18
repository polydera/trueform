"""
Remesh pipeline example with VTK

Left: original mesh. Right top: decimated. Right bottom: isotropic remeshed.
Edge wireframe on right panels scales with window height.
"""

import vtk
import numpy as np
import trueform as tf
import time
import argparse
import os

from util import (
    numpy_to_polydata,
    create_text_actor,
)


EDGE_COLOR = (0.15, 0.15, 0.18)
BASE_HEIGHT = 700


def make_mesh_actor(faces, points, color=(0.85, 0.85, 0.88), edges=False):
    """Create a VTK actor from numpy face/point arrays"""
    polydata = numpy_to_polydata(points, faces)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    prop = actor.GetProperty()
    prop.SetColor(*color)
    prop.SetAmbient(0.2)
    prop.SetDiffuse(0.8)
    if edges:
        prop.EdgeVisibilityOn()
        prop.SetEdgeColor(*EDGE_COLOR)
        prop.SetLineWidth(1.0)
    return actor


def update_edge_width(actors, window):
    """Scale edge line width relative to window height"""
    h = window.GetSize()[1]
    w = max(0.5, 0.7 * h / BASE_HEIGHT)
    for a in actors:
        a.GetProperty().SetLineWidth(w)


def main():
    parser = argparse.ArgumentParser(
        description="Remesh pipeline: decimate then isotropic remesh",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "mesh", nargs="?", default=None,
        help="Path to mesh file (STL). Default: benchmarks/data/dragon-500k.stl",
    )
    parser.add_argument(
        "-t", "--target", type=float, default=0.05,
        help="Decimation target proportion (0.0-1.0). Default: 0.05",
    )
    args = parser.parse_args()

    # Default mesh path
    benchmarks_data = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..",
                     "..", "..", "benchmarks", "data")
    )
    mesh_path = args.mesh or os.path.join(benchmarks_data, "dragon-500k.stl")
    target = args.target

    # Load mesh
    print(f"Loading: {mesh_path}")
    faces, points = tf.read_stl(mesh_path)
    n_original = len(faces)
    print(f"  {n_original} faces, {len(points)} points")

    # Decimate
    print(f"Decimating to {target * 100:.1f}% ...")
    mesh = tf.Mesh(faces, points)
    t0 = time.perf_counter()
    dec_faces, dec_points = tf.decimated(mesh, target)
    t_dec = time.perf_counter() - t0
    n_dec = len(dec_faces)
    print(f"  {n_dec} faces, {len(dec_points)} points  ({t_dec * 1000:.1f} ms)")

    # Isotropic remesh to mean edge length of decimated mesh
    mel = tf.mean_edge_length((dec_faces, dec_points))
    print(f"Isotropic remesh (target length {mel:.4f}) ...")
    dec_mesh = tf.Mesh(dec_faces, dec_points)
    t0 = time.perf_counter()
    rem_faces, rem_points = tf.isotropic_remeshed(
        dec_mesh, mel, use_quadric=True)
    t_rem = time.perf_counter() - t0
    n_rem = len(rem_faces)
    print(f"  {n_rem} faces, {len(rem_points)} points  ({t_rem * 1000:.1f} ms)")

    # --- VTK setup: left half = original, right top = decimated, right bottom = remeshed ---

    ren_orig = vtk.vtkRenderer()
    ren_dec = vtk.vtkRenderer()
    ren_rem = vtk.vtkRenderer()
    ren_text = vtk.vtkRenderer()

    text_h = 0.10
    half = 0.5
    mid_y = text_h + (1.0 - text_h) / 2.0

    ren_orig.SetViewport(0.0, text_h, half, 1.0)
    ren_dec.SetViewport(half, mid_y, 1.0, 1.0)
    ren_rem.SetViewport(half, text_h, 1.0, mid_y)
    ren_text.SetViewport(0.0, 0.0, 1.0, text_h)
    ren_text.InteractiveOff()

    bg = (27.0 / 255.0, 43.0 / 255.0, 52.0 / 255.0)
    for r in (ren_orig, ren_dec, ren_rem):
        r.SetBackground(*bg)
    ren_text.SetBackground(0.090, 0.143, 0.173)

    # Actors
    actor_orig = make_mesh_actor(faces, points)
    actor_dec = make_mesh_actor(dec_faces, dec_points, edges=True)
    actor_rem = make_mesh_actor(rem_faces, rem_points, edges=True)
    ren_orig.AddActor(actor_orig)
    ren_dec.AddActor(actor_dec)
    ren_rem.AddActor(actor_rem)

    # Labels
    ren_orig.AddViewProp(create_text_actor(
        f"Original: {n_original:,} faces",
        font_size=18, position=(0.03, 0.03),
    ))
    ren_dec.AddViewProp(create_text_actor(
        f"Decimated: {n_dec:,} faces ({target * 100:.0f}%)",
        font_size=14, position=(0.03, 0.05),
    ))
    ren_rem.AddViewProp(create_text_actor(
        f"Remeshed: {n_rem:,} faces",
        font_size=14, position=(0.03, 0.05),
    ))

    # Text strip
    def fmt(t):
        return f"{t * 1000:.1f} ms" if t < 1.0 else f"{t:.2f} s"

    ren_text.AddViewProp(create_text_actor(
        f"Decimate {fmt(t_dec)}  |  Remesh {fmt(t_rem)}  |  "
        f"{n_original:,} \u2192 {n_dec:,} \u2192 {n_rem:,} faces",
        font_size=34, position=(0.02, 0.50),
    ))

    # Window
    window = vtk.vtkRenderWindow()
    window.AddRenderer(ren_orig)
    window.AddRenderer(ren_dec)
    window.AddRenderer(ren_rem)
    window.AddRenderer(ren_text)
    window.SetSize(1600, 800)

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(window)
    interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

    # Adaptive edge width on resize
    edge_actors = [actor_dec, actor_rem]
    update_edge_width(edge_actors, window)
    window.AddObserver("ModifiedEvent",
                       lambda obj, ev: update_edge_width(edge_actors, obj))

    # Share camera across all three
    ren_dec.SetActiveCamera(ren_orig.GetActiveCamera())
    ren_rem.SetActiveCamera(ren_orig.GetActiveCamera())
    ren_orig.ResetCamera()

    window.Render()
    interactor.Start()


if __name__ == "__main__":
    main()
