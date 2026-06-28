/*
 * Copyright (c) 2026 XLAB
 * All rights reserved.
 *
 * This file is part of trueform (trueform.polydera.com)
 *
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0.
 * Commercial licensing available via info@polydera.com.
 *
 * Author: Žiga Sajovic
 */
// Cellular fracture: one CSG arrangement of a closed mesh + a grid of
// axis-aligned cutting planes, split into its volumetric box cells, coloured
// and optionally exploded. Showcases expression-filtered volumetric-domain
// extraction from a single build.
//
//   csg_fracture MESH [cuts] [explode]
//
//   MESH        path to a closed surface mesh (.stl / .obj)
//   cuts        number of axis-aligned cuts per axis              [default 6]
//   explode     outward cell displacement, fraction of extent     [default 0.0]
//
// The Python sibling (python/examples/vtk/csg_fracture.py) walks the
// arrangement graph by hand to keep only the cells inside the solid. The CSG
// path expresses exactly that as a boolean: a cell is interior iff it lies
// inside operand 0 (the mesh), so `make_csg_domains(graph, tf::csg::op(0))`
// returns the interior cells directly. The default domain_config
// (exclude_outer_shell | ignore_open_fragments) drops the unbounded universe
// and the open cutting-plane fins, leaving the bounded box cells.

#include <trueform/csg.hpp>
#include <trueform/trueform.hpp>
#include <trueform/vtk.hpp>

#include <vtkFeatureEdges.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkLightKit.h>
#include <vtkNew.h>
#include <vtkOpenGLActor.h>
#include <vtkOpenGLPolyDataMapper.h>
#include <vtkOpenGLRenderer.h>
#include <vtkProperty.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>

#include <array>
#include <cstdio>
#include <string>
#include <vector>

using Index = int;
using Real = float;
using Mesh = tf::polygons_buffer<Index, Real, 3, 3>;

namespace {

// Pastel qualitative palette (the paper's tfteal/tfrose family), mirrored
// from the Python plate so the two render as siblings.
constexpr std::array<std::array<unsigned char, 3>, 12> PALETTE = {{
    {0x8F, 0xE3, 0xD6}, {0xEF, 0xA9, 0xC4}, {0x8F, 0xB4, 0xD8},
    {0xF6, 0xBE, 0x84}, {0x8F, 0xC8, 0x82}, {0xB9, 0xA0, 0xDC},
    {0xEF, 0x9A, 0x98}, {0x6F, 0xC3, 0xB4}, {0xF2, 0xCE, 0x86},
    {0xA9, 0xC4, 0xE6}, {0xA6, 0xDD, 0xBE}, {0xD6, 0xA9, 0xCE},
}};
constexpr std::array<double, 3> EDGE_RGB = {0.18, 0.20, 0.24};

// Coprime step spreads neighbouring cells across the palette.
auto cell_color(std::size_t i) -> std::array<double, 3> {
  auto &c = PALETTE[(i * 5) % PALETTE.size()];
  return {c[0] / 255.0, c[1] / 255.0, c[2] / 255.0};
}

// Read a closed surface (.stl / .obj), triangulate + weld, and normalise to
// unit max-extent centred at the origin.
auto load_normalized(const std::string &path) -> Mesh {
  Mesh m;
  auto dot = path.rfind('.');
  auto ext = dot == std::string::npos ? std::string{} : path.substr(dot + 1);
  if (ext == "stl" || ext == "STL") {
    m = tf::cleaned(tf::read_stl<Index>(path).polygons());
  } else if (ext == "obj" || ext == "OBJ") {
    auto raw = tf::read_obj<Index, Real>(path);
    m = tf::cleaned(tf::triangulated(raw.polygons()).polygons());
  } else {
    throw std::runtime_error("unsupported mesh format (use .stl / .obj): " +
                             path);
  }

  auto bb = tf::aabb_from(m.polygons());
  std::array<Real, 3> c{(bb.min[0] + bb.max[0]) * Real(0.5),
                        (bb.min[1] + bb.max[1]) * Real(0.5),
                        (bb.min[2] + bb.max[2]) * Real(0.5)};
  Real ext_max = Real(0);
  for (int a = 0; a < 3; ++a)
    ext_max = std::max(ext_max, bb.max[a] - bb.min[a]);
  const Real inv = ext_max > Real(0) ? Real(1) / ext_max : Real(1);

  auto &d = m.points_buffer().data_buffer();
  for (std::size_t i = 0; i + 3 <= d.size(); i += 3)
    for (int a = 0; a < 3; ++a)
      d[i + a] = (d[i + a] - c[a]) * inv;
  return m;
}

// A square cutting plane perpendicular to `axis` at coordinate `off`. The
// XY plane mesh is remapped so its two spanning coords land on the two
// non-`axis` axes and the third sits at `off`.
auto make_axis_plane(int axis, Real off, Real side) -> Mesh {
  Mesh m = tf::triangulated(tf::make_plane_mesh<Index>(side, side).polygons());
  const int b1 = (axis + 1) % 3, b2 = (axis + 2) % 3;
  auto &d = m.points_buffer().data_buffer();
  for (std::size_t i = 0; i + 3 <= d.size(); i += 3) {
    const Real x = d[i], y = d[i + 1]; // plane spans (x, y); z is 0
    d[i + axis] = off;
    d[i + b1] = x;
    d[i + b2] = y;
  }
  return m;
}

// `n` evenly spaced interior cuts per axis across the centred unit mesh.
auto make_planes(int n) -> std::vector<Mesh> {
  std::vector<Mesh> planes;
  planes.reserve(static_cast<std::size_t>(3 * n));
  const Real side = Real(2); // spans the unit-extent mesh with margin
  for (int axis = 0; axis < 3; ++axis)
    for (int i = 0; i < n; ++i) {
      const Real t = Real(i + 1) / Real(n + 1); // interior, exclusive
      planes.push_back(make_axis_plane(axis, Real(-0.5) + t, side));
    }
  return planes;
}

// Displace a cell outward from the origin by `factor` of its centroid.
auto explode_in_place(Mesh &cell, Real factor) -> void {
  auto bb = tf::aabb_from(cell.polygons());
  std::array<Real, 3> off{(bb.min[0] + bb.max[0]) * Real(0.5) * factor,
                          (bb.min[1] + bb.max[1]) * Real(0.5) * factor,
                          (bb.min[2] + bb.max[2]) * Real(0.5) * factor};
  auto &d = cell.points_buffer().data_buffer();
  for (std::size_t i = 0; i + 3 <= d.size(); i += 3)
    for (int a = 0; a < 3; ++a)
      d[i + a] += off[a];
}

auto add_cell_actors(vtkOpenGLRenderer *ren, const Mesh &cell, std::size_t i)
    -> void {
  auto poly = tf::vtk::make_vtk_polydata(cell);

  vtkNew<vtkOpenGLPolyDataMapper> mapper;
  mapper->SetInputData(poly);
  vtkNew<vtkOpenGLActor> actor;
  actor->SetMapper(mapper);
  auto rgb = cell_color(i);
  auto *p = actor->GetProperty();
  p->SetColor(rgb[0], rgb[1], rgb[2]);
  p->SetInterpolationToPhong();
  p->SetAmbient(0.32);
  p->SetDiffuse(0.68);
  p->SetSpecular(0.18);
  p->SetSpecularPower(28);
  ren->AddActor(actor);

  // Crisp feature edges only (cut seams + silhouette), not every triangle.
  vtkNew<vtkFeatureEdges> fe;
  fe->SetInputData(poly);
  fe->BoundaryEdgesOn();
  fe->FeatureEdgesOn();
  fe->ManifoldEdgesOff();
  fe->NonManifoldEdgesOff();
  fe->SetFeatureAngle(25);
  fe->Update();
  vtkNew<vtkOpenGLPolyDataMapper> emapper;
  emapper->SetInputConnection(fe->GetOutputPort());
  emapper->ScalarVisibilityOff();
  vtkNew<vtkOpenGLActor> eactor;
  eactor->SetMapper(emapper);
  eactor->GetProperty()->SetColor(EDGE_RGB[0], EDGE_RGB[1], EDGE_RGB[2]);
  eactor->GetProperty()->SetLineWidth(1.1);
  eactor->GetProperty()->LightingOff();
  ren->AddActor(eactor);
}

} // namespace

int main(int argc, char *argv[]) {
  std::vector<std::string> pos;
  for (int i = 1; i < argc; ++i)
    pos.emplace_back(argv[i]);
  if (pos.empty()) {
    std::fprintf(stderr,
                 "Usage: %s MESH [cuts] [explode]\n"
                 "  MESH     .stl / .obj closed surface\n"
                 "  cuts     axis-aligned cuts per axis   [default 6]\n"
                 "  explode  outward displacement frac    [default 0.0]\n",
                 argv[0]);
    return 1;
  }
  const std::string mesh_path = pos[0];
  const int cuts = pos.size() > 1 ? std::atoi(pos[1].c_str()) : 6;
  const Real explode = pos.size() > 2 ? Real(std::atof(pos[2].c_str())) : Real(0);

  auto solid = load_normalized(mesh_path);
  std::printf("%s: %zu tris (normalised)\n", mesh_path.c_str(),
              solid.faces().size());

  auto planes = make_planes(cuts);
  std::printf("cutting planes: %zu (%d per axis)\n", planes.size(), cuts);

  // forms[0] is the solid; forms[1..] are the open cutting planes.
  std::vector<decltype(solid.polygons())> forms;
  forms.reserve(planes.size() + 1);
  forms.push_back(solid.polygons());
  for (auto &p : planes)
    forms.push_back(p.polygons());

  tf::tick();
  auto graph = tf::make_csg_graph(tf::make_range(forms));
  tf::tock("make_csg_graph");

  // op(0) = "inside the solid": keeps exactly the interior box cells.
  tf::tick();
  auto [cells, ids] = tf::make_csg_domains(graph, tf::csg::op(0));
  tf::tock("make_csg_domains");
  std::printf("cells: %zu  (interior of solid)\n", cells.size());

  vtkNew<vtkOpenGLRenderer> ren;
  ren->SetBackground(1, 1, 1);
  ren->UseFXAAOn();
  std::size_t shown = 0;
  for (auto &cell : cells) {
    if (cell.points().size() == 0 || cell.faces().size() == 0)
      continue;
    if (explode != Real(0))
      explode_in_place(cell, explode);
    add_cell_actors(ren, cell, shown++);
  }

  vtkNew<vtkLightKit> lk;
  lk->SetKeyLightIntensity(0.9);
  lk->SetKeyToFillRatio(2.2);
  lk->SetKeyToHeadRatio(3.5);
  lk->SetKeyToBackRatio(3.0);
  lk->AddLightsToRenderer(ren);

  vtkNew<vtkRenderWindow> win;
  win->SetMultiSamples(8);
  win->AddRenderer(ren);
  win->SetSize(1400, 1400);
  win->SetWindowName("csg_fracture");

  vtkNew<vtkRenderWindowInteractor> iren;
  iren->SetRenderWindow(win);
  vtkNew<vtkInteractorStyleTrackballCamera> style;
  iren->SetInteractorStyle(style);

  ren->ResetCamera();
  win->Render();
  iren->Start();
  return 0;
}
