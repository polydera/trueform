"""
Cellular fracture: one arrangement of a closed mesh + a grid of axis-aligned
cutting planes, split into its volumetric box cells, coloured and optionally
exploded. Showcases volumetric-domain extraction (not a boolean) from a single
build: tf.CsgGraph builds the arrangement once, and the interior cells are
read directly as the domains inside operand 0 — graph.domains(tf.op(0)).

  python csg_fracture.py MESH [cuts] [explode] [--all] [--png]

  MESH        path to a closed surface mesh (stl / obj)
  cuts        number of axis-aligned cuts per axis              [default 12]
  explode     outward cell displacement, fraction of extent     [default 0.0]
  --all       render every domain, skipping the interior filter
  --png       render offscreen to /tmp/fracture.png

Interactive controls: drag orbit, scroll zoom, 'c' print camera, 'q' quit.
"""
import os
import numpy as np
import vtk
import trueform as tf
from util import numpy_to_polydata, create_parser

DT, IDT = np.float64, np.int32

# Pastel qualitative palette derived from the paper's accents (tfteal/tfrose
# brand + cOne..cFive Tableau set), softened toward the light-teal/light-rose
# domain figures so this plate reads as part of the same family.
PALETTE = [
    "8FE3D6",  # soft teal   (tfteal / aFace)
    "EFA9C4",  # soft rose   (tfrose / bFace)
    "8FB4D8",  # soft blue   (cOne)
    "F6BE84",  # soft orange (cTwo)
    "8FC882",  # soft green  (cThree)
    "B9A0DC",  # soft purple (cFour)
    "EF9A98",  # soft red    (cFive)
    "6FC3B4",  # deep teal   (tfd2face)
    "F2CE86",  # soft gold
    "A9C4E6",  # soft sky
    "A6DDBE",  # soft mint
    "D6A9CE",  # soft mauve
]
EDGE_RGB = (0.18, 0.20, 0.24)   # near-ink cell outlines (tfink family)

# Camera override for reproducing a hand-picked view. Press 'c' in the
# interactive window to print a ready-to-paste block, drop it here, then a
# --png render replicates that exact view. None -> default ears-up framing.
#   CAM = dict(pos=(...), foc=(...), up=(...), angle=..., size=(W, H))
CAM = None


def cell_color(i):
    hx = PALETTE[(i * 5) % len(PALETTE)]   # coprime step spreads neighbours
    return tuple(int(hx[k:k + 2], 16) / 255 for k in (0, 2, 4))


def load_closed(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".stl":
        faces, pts = tf.read_stl(path, index_dtype=IDT)
    elif ext == ".obj":
        faces, pts = tf.read_obj(path, ngon=3, dtype=DT, index_dtype=IDT)
    else:
        raise SystemExit(f"unsupported mesh extension '{ext}' (use .stl or .obj)")
    faces, pts = tf.cleaned((faces, pts.astype(DT)), tolerance=1e-6)
    if not tf.is_closed(tf.Mesh(faces, pts)):
        raise SystemExit(f"{os.path.basename(path)} is not watertight; "
                         "'inside the solid' is undefined on an open surface")
    # normalise to unit max-extent, centred at the origin
    c = (pts.min(0) + pts.max(0)) / 2
    ext = (pts.max(0) - pts.min(0)).max()
    return faces, (pts - c) / ext


def rot_z_to(n):
    z = np.array([0, 0, 1.0])
    if np.allclose(n, z): return np.eye(3)
    if np.allclose(n, -z): return np.diag([1.0, -1.0, -1.0])
    ax = np.cross(z, n); ax /= np.linalg.norm(ax)
    co = float(np.dot(z, n)); si = np.sqrt(max(0.0, 1 - co * co))
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + si * K + (1 - co) * (K @ K)


def make_plane(center, normal, side=4.0):
    pf, pp = tf.make_plane_mesh(side, side, dtype=DT, index_dtype=IDT)
    pp = np.asarray(pp, DT)
    pp = (pp - pp.mean(0)) @ rot_z_to(normal).T + center
    return tf.Mesh(np.asarray(pf, IDT), pp)


def make_planes(bp, n):
    # n axis-aligned cuts per axis at evenly spaced interior offsets -> box cells
    lo, hi = bp.min(0), bp.max(0)
    mid = (lo + hi) / 2
    planes = []
    for axis in range(3):
        nrm = np.zeros(3); nrm[axis] = 1.0
        for i in range(n):
            t = (i + 1) / (n + 1)
            ctr = mid.copy(); ctr[axis] = lo[axis] + t * (hi[axis] - lo[axis])
            planes.append(make_plane(ctr, nrm))
    return planes


def set_camera(ren, win, cam):
    if CAM:
        cam.SetPosition(*CAM["pos"]); cam.SetFocalPoint(*CAM["foc"])
        cam.SetViewUp(*CAM["up"]); cam.SetViewAngle(CAM["angle"])
        win.SetSize(*CAM["size"])
    else:
        # explicit direction so z is genuinely up (avoids the parallel view-up
        # reset); look from the front (-y), to the side (+x) and above.
        win.SetSize(1400, 1400)
        cam.SetViewUp(0, 0, 1)
        cam.SetFocalPoint(0, 0, 0)
        cam.SetPosition(0.55, -1.0, 0.32)
        ren.ResetCamera()
        cam.Zoom(1.35)
    ren.ResetCameraClippingRange()


def main():
    parser = create_parser(__doc__, mesh_args=1)
    parser.add_argument("cuts", nargs="?", type=int, default=12,
                        help="number of axis-aligned cuts per axis")
    parser.add_argument("explode", nargs="?", type=float, default=0.0,
                        help="outward cell displacement, fraction of extent")
    parser.add_argument("--all", action="store_true", dest="all_domains",
                        help="render every domain, skip the interior filter")
    parser.add_argument("--png", action="store_true",
                        help="render offscreen to /tmp/fracture.png")
    args = parser.parse_args()

    sf, sp = load_closed(args.mesh)
    name = os.path.basename(args.mesh)
    print(f"{name}: {len(sf)} tris, bbox {sp.min(0)} .. {sp.max(0)}")

    meshes = [tf.Mesh(sf, sp)] + make_planes(sp, args.cuts)

    # One build; every query after this reuses it.
    graph = tf.CsgGraph(meshes)
    if args.all_domains:
        domains, _ids = graph.domains()
    else:
        # a cell is interior iff it lies inside operand 0 (the solid)
        domains, _ids = graph.domains(tf.op(0))
    print(f"cells: {len(domains)}" + ("  (ALL)" if args.all_domains else
                                      "  (interior of solid)"))

    scen = sp.mean(0)
    cells = []
    for cf, cp in domains:
        cp = np.asarray(cp, DT); cf = np.asarray(cf)
        if len(cp) == 0 or len(cf) < 1: continue
        cells.append((cf, cp + args.explode * (cp.mean(0) - scen)))
    print(f"rendered domains: {len(cells)}")

    ren = vtk.vtkRenderer(); ren.SetBackground(1, 1, 1)
    if not args.png:
        ren.UseFXAAOn()   # FXAA halos transparent edges; rely on MSAA for PNG
    for i, (cf, cp) in enumerate(cells):
        poly = numpy_to_polydata(cp.astype(np.float64), cf)
        m = vtk.vtkPolyDataMapper(); m.SetInputData(poly)
        a = vtk.vtkActor(); a.SetMapper(m)
        p = a.GetProperty()
        p.SetColor(*cell_color(i))
        p.SetEdgeVisibility(0)            # no triangulation
        p.SetInterpolationToPhong()
        p.SetAmbient(0.32); p.SetDiffuse(0.68)
        p.SetSpecular(0.18); p.SetSpecularPower(28)
        ren.AddActor(a)
        # crisp feature edges only (cut seams + silhouette), not the mesh
        fe = vtk.vtkFeatureEdges(); fe.SetInputData(poly)
        fe.BoundaryEdgesOn(); fe.FeatureEdgesOn(); fe.SetFeatureAngle(25)
        fe.ManifoldEdgesOff(); fe.NonManifoldEdgesOff(); fe.Update()
        em = vtk.vtkPolyDataMapper(); em.SetInputConnection(fe.GetOutputPort())
        em.ScalarVisibilityOff()
        ea = vtk.vtkActor(); ea.SetMapper(em)
        ea.GetProperty().SetColor(*EDGE_RGB)
        ea.GetProperty().SetLineWidth(1.1)
        ea.GetProperty().LightingOff()
        ren.AddActor(ea)

    # studio 3-point lighting for shape-revealing shading
    lk = vtk.vtkLightKit()
    lk.SetKeyLightIntensity(0.9)
    lk.SetKeyToFillRatio(2.2); lk.SetKeyToHeadRatio(3.5); lk.SetKeyToBackRatio(3.0)
    lk.AddLightsToRenderer(ren)

    win = vtk.vtkRenderWindow()
    win.SetMultiSamples(8)
    win.AddRenderer(ren)
    win.SetWindowName(f"csg_fracture: {name} / {len(cells)} cells / {args.cuts} cuts")
    cam = ren.GetActiveCamera()
    set_camera(ren, win, cam)

    if args.png:
        win.SetAlphaBitPlanes(1)          # transparent background (RGBA)
        win.SetOffScreenRendering(1)
        win.Render()
        w2i = vtk.vtkWindowToImageFilter(); w2i.SetInput(win)
        w2i.SetInputBufferTypeToRGBA(); w2i.ReadFrontBufferOff(); w2i.Update()
        wr = vtk.vtkPNGWriter(); wr.SetFileName("/tmp/fracture.png")
        wr.SetInputConnection(w2i.GetOutputPort()); wr.Write()
        print("wrote /tmp/fracture.png (RGBA)")
    else:
        iren = vtk.vtkRenderWindowInteractor()
        iren.SetRenderWindow(win)
        iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

        def dump_camera(obj, _evt):
            if obj.GetKeySym() not in ("c", "C"):
                return
            rd = lambda v: tuple(round(x, 4) for x in v)
            print("\nCAM = dict(")
            print(f"    pos={rd(cam.GetPosition())},")
            print(f"    foc={rd(cam.GetFocalPoint())},")
            print(f"    up={rd(cam.GetViewUp())},")
            print(f"    angle={round(cam.GetViewAngle(), 3)}, "
                  f"size={tuple(win.GetSize())})\n")
        iren.AddObserver("KeyPressEvent", dump_camera)

        win.Render()
        print("interactive: drag orbit, scroll zoom, 'c' print camera, 'q' quit")
        iren.Start()


if __name__ == "__main__":
    main()
