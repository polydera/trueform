"""
Error-budget simplification with VTK

Runs tf.simplified on one input mesh at a sweep of error_rel values, with
everything else held fixed (min_quality, feature_angle, iterations,
optimize_iterations), in a 2x3 grid of panels that share a single camera:

  input  |  error_rel = e0  |  e1
  e2     |  e3              |  e4

As error_rel grows the mesh coarsens hard, but the triangulation stays clean:
the min_quality floor and the flip+relax cleanup keep triangles well shaped
instead of leaving slivers. Close the window to exit.

  python simplify.py                      # default benchmarks/data/dragon-500k.stl
  python simplify.py path/to/mesh.stl
  python simplify.py path/to/mesh.obj     # .stl and .obj are both read
"""

import argparse
import os
import time

import numpy as np
import vtk
import trueform as tf

from util import numpy_to_polydata, create_text_actor


# Only error_rel varies across panels; everything else is fixed so the sweep
# isolates the error budget. Five values plus the input fill a 2x3 grid.
ERROR_RELS = [0.0005, 0.001, 0.002, 0.004, 0.008]
MIN_QUALITY = 0.3
FEATURE_ANGLE = 60.0       # organic mesh: let the error budget drive it
ITERATIONS = 3             # iterated error-budget passes
OPTIMIZE_ITERATIONS = 3    # flip + relax cleanup per pass

MESH_COLOR = (0.78, 0.80, 0.85)
EDGE_COLOR = (0.10, 0.12, 0.14)
BG = (27.0 / 255.0, 43.0 / 255.0, 52.0 / 255.0)


def load_mesh(path):
    """Read a mesh, choosing the reader from the file extension."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".stl":
        return tf.read_stl(path)
    if ext == ".obj":
        return tf.read_obj(path, ngon=3)
    raise ValueError(f"unsupported mesh extension '{ext}' (use .stl or .obj)")


def make_actor(faces, points):
    """VTK actor from numpy face/point arrays, with the wireframe shown."""
    poly = numpy_to_polydata(np.asarray(points, np.float64), faces)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(poly)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    prop = actor.GetProperty()
    prop.SetColor(*MESH_COLOR)
    prop.SetAmbient(0.2)
    prop.SetDiffuse(0.85)
    prop.EdgeVisibilityOn()
    prop.SetEdgeColor(*EDGE_COLOR)
    prop.SetLineWidth(1.0)
    return actor


def show(panels):
    """Lay the panels out in a 2x3 grid, all sharing the first panel's camera."""
    cols, rows = 3, 2
    window = vtk.vtkRenderWindow()
    window.SetWindowName("simplify: fixed quality, increasing error_rel")
    rens = []
    for i, (label, faces, points) in enumerate(panels):
        ren = vtk.vtkRenderer()
        ren.AddActor(make_actor(faces, points))
        ren.AddViewProp(create_text_actor(
            label, font_size=14, position=(0.04, 0.05)))
        ren.SetBackground(*BG)
        col, row = i % cols, i // cols
        y1 = 1.0 - row / rows  # top row first
        ren.SetViewport(col / cols, y1 - 1.0 / rows, (col + 1) / cols, y1)
        window.AddRenderer(ren)
        rens.append(ren)
    window.SetSize(1500, 1000)

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(window)
    interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

    rens[0].ResetCamera()
    for r in rens[1:]:
        r.SetActiveCamera(rens[0].GetActiveCamera())
    window.Render()
    interactor.Start()


def main():
    default_mesh = os.path.normpath(os.path.join(
        os.path.dirname(__file__), "..", "..", "..",
        "benchmarks", "data", "dragon-500k.stl"))

    parser = argparse.ArgumentParser(
        description="Error-budget simplification sweep (tf.simplified)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "mesh", nargs="?", default=default_mesh,
        help="Path to an input mesh (.stl or .obj). "
             "Default: benchmarks/data/dragon-500k.stl",
    )
    args = parser.parse_args()
    path = os.path.expanduser(args.mesh)
    if not os.path.exists(path):
        parser.error(f"mesh not found: {path}")

    faces, points = load_mesh(path)
    print(f"loaded {os.path.basename(path)}: "
          f"{faces.shape[0]:,} faces, {points.shape[0]:,} points")

    panels = [(f"input\n{faces.shape[0]:,} faces", faces, points)]
    for er in ERROR_RELS:
        t0 = time.perf_counter()
        sf, sp = tf.simplified(
            (faces, points),
            error_rel=er,
            min_quality=MIN_QUALITY,
            feature_angle=FEATURE_ANGLE,
            iterations=ITERATIONS,
            optimize_iterations=OPTIMIZE_ITERATIONS,
        )
        dt = time.perf_counter() - t0
        pct = 100.0 * sf.shape[0] / faces.shape[0]
        print(f"  error_rel={er:<7} -> {sf.shape[0]:>8,} faces "
              f"({pct:5.1f}%)  {dt * 1000:7.1f} ms")
        panels.append(
            (f"error_rel = {er}\n{sf.shape[0]:,} faces ({pct:.1f}%)", sf, sp))

    show(panels)


if __name__ == "__main__":
    main()
