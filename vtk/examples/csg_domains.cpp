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
// Volumetric domain extraction from a set of input OBJ files, via the CSG
// path (make_csg_graph -> make_csg_domains) rather than the arrangement +
// make_domain_labels + split_into_domains path used by domains.cpp.
//
// Usage:
//   csg_domains <dir>                 # load every .obj from <dir>
//   csg_domains a.obj b.obj ...       # load specific files
//   csg_domains --tri=refined <dir>   # refined-cdt store (default: cdt)
//
// Pipeline:
//   read_obj per file -> triangulated -> cleaned -> make_csg_graph ->
//   make_csg_domains (default: drop the universe, fuse open fragments).
//   Each kept 3D domain comes back as its own watertight, outward-oriented
//   mesh -- no domain_labels, no split, no single merged arrangement.
//
// Viewer (split-screen):
//   Left  -- every domain cell, the current one opaque, the rest ghosted.
//   Right -- the current domain cell on its own, opaque.
//
//   Left / Right keys cycle the current domain. Cells are ordered by their
//   max-Z vertex (top -> bottom).

#include <trueform/trueform.hpp>
#include <trueform/vtk.hpp>

#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkNew.h>
#include <vtkObjectFactory.h>
#include <vtkOpenGLActor.h>
#include <vtkOpenGLPolyDataMapper.h>
#include <vtkOpenGLRenderer.h>
#include <vtkPolyData.h>
#include <vtkProperty.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

using Index = int;
using Real = double;
using Mesh = tf::polygons_buffer<Index, Real, 3, 3>;
using MeshVec = std::vector<Mesh, tf::allocator<Mesh>>;

// Weld tolerance applied to each input mesh on load. Matches domains.cpp.
constexpr Real WELD_TOLERANCE = 1e-6;

// Opacity for the non-current cells in the left viewport.
constexpr double GHOST_OPACITY = 0.12;

// Polydera color scheme -- dark teal-tinted backgrounds + warm/cool accents.
namespace palette {

constexpr std::array<double, 3> BG_LEFT = {0.047, 0.082, 0.075};  // bg-0
constexpr std::array<double, 3> BG_RIGHT = {0.071, 0.122, 0.114}; // bg-1

// Mesh palette -- distinguishable on the dark teal backdrop.
constexpr std::array<std::array<unsigned char, 3>, 8> MESH = {{
    {45, 223, 206},  // teal-400
    {196, 69, 105},  // rose
    {242, 191, 26},  // amber
    {155, 115, 214}, // violet
    {110, 200, 230}, // sky
    {109, 204, 138}, // green
    {231, 110, 116}, // coral
    {180, 180, 188}, // neutral
}};

auto mesh_color(Index i) -> std::array<double, 3> {
  auto c = MESH[static_cast<std::size_t>(i) % MESH.size()];
  return {c[0] / 255.0, c[1] / 255.0, c[2] / 255.0};
}

} // namespace palette

// -----------------------------------------------------------------------------
// argv -> list of .obj paths (single directory, or explicit list of files)
// -----------------------------------------------------------------------------
auto collect_obj_paths(const std::vector<std::string> &args)
    -> std::vector<std::string> {
  std::vector<std::string> paths;
  if (args.size() == 1) {
    std::filesystem::path p(args[0]);
    if (std::filesystem::is_directory(p)) {
      for (auto &entry : std::filesystem::directory_iterator(p))
        if (entry.path().extension() == ".obj")
          paths.push_back(entry.path().string());
      std::sort(paths.begin(), paths.end());
      return paths;
    }
  }
  return args;
}

// -----------------------------------------------------------------------------
// Pipeline. All trueform calls operate on trueform types; the conversion
// to vtkPolyData only happens at display time.
// -----------------------------------------------------------------------------
struct PipelineResult {
  MeshVec components;
  tf::buffer<Index> comp_labels;
};

// Load each .obj as a triangulated double-precision Mesh, welded at
// `tolerance` so the input is already deduplicated.
auto load_meshes(const std::vector<std::string> &paths, Real tolerance)
    -> std::vector<Mesh> {
  std::printf("loading %zu meshes...\n", paths.size());
  std::vector<Mesh> meshes;
  meshes.reserve(paths.size());
  tf::tick();
  for (auto &p : paths) {
    auto raw = tf::read_obj<Index, Real>(p);
    auto tri = tf::triangulated(raw.polygons());
    Mesh m = tf::cleaned(tri.polygons(), tolerance);
    std::printf("  loaded %s: %zu pts, %zu tris\n", p.c_str(),
                m.points().size(), m.faces().size());
    meshes.push_back(std::move(m));
  }
  tf::tock("loading:");
  return meshes;
}

// Order cells by max-Z descending (top -> bottom).
auto sort_for_display(MeshVec &components,
                      tf::buffer<Index> &comp_labels) -> void {
  if (components.empty())
    return;

  std::vector<Real> max_z(components.size());
  for (std::size_t i = 0; i < components.size(); ++i)
    max_z[i] = tf::aabb_from(components[i].polygons()).max[2];

  std::vector<Index> order(components.size());
  for (std::size_t i = 0; i < order.size(); ++i)
    order[i] = Index(i);
  std::sort(order.begin(), order.end(),
            [&](Index a, Index b) { return max_z[a] > max_z[b]; });

  MeshVec reordered;
  reordered.reserve(components.size());
  tf::buffer<Index> reordered_labels;
  reordered_labels.allocate(components.size());
  for (std::size_t i = 0; i < order.size(); ++i) {
    reordered.push_back(std::move(components[order[i]]));
    reordered_labels[i] = comp_labels[order[i]];
  }
  components = std::move(reordered);
  comp_labels = std::move(reordered_labels);
}

auto build_pipeline(const std::vector<std::string> &paths,
                    tf::triangulation_type tri) -> PipelineResult {
  auto meshes = load_meshes(paths, WELD_TOLERANCE);

  std::vector<decltype(meshes[0].polygons())> forms;
  forms.reserve(meshes.size());
  for (auto &m : meshes)
    forms.push_back(m.polygons());

  std::printf("running make_csg_graph on %zu meshes (%s)...\n", forms.size(),
              tri == tf::triangulation_type::refined_cdt ? "refined_cdt"
                                                         : "cdt");
  tf::tick();
  auto graph = tf::make_csg_graph(tf::make_range(forms), tri);
  tf::tock("graph:");

  std::printf("running make_csg_domains...\n");
  tf::tick();
  auto [cells, ids] = tf::make_csg_domains(graph);
  tf::tock("domains:");
  std::printf("  %zu domains\n", cells.size());

  PipelineResult pr{std::move(cells), std::move(ids)};
  sort_for_display(pr.components, pr.comp_labels);
  return pr;
}

// -----------------------------------------------------------------------------
// Interactor -- Left/Right cycle the current domain
// -----------------------------------------------------------------------------
class domain_cycle_interactor : public vtkInteractorStyleTrackballCamera {
public:
  static auto New() -> domain_cycle_interactor *;
  vtkTypeMacro(domain_cycle_interactor, vtkInteractorStyleTrackballCamera);

  auto initialize(std::vector<vtkSmartPointer<vtkOpenGLActor>> left_actors,
                  vtkOpenGLPolyDataMapper *right_mapper,
                  vtkOpenGLActor *right_actor, vtkOpenGLRenderer *right_ren,
                  PipelineResult *pr) -> void {
    _left_actors = std::move(left_actors);
    _right_mapper = right_mapper;
    _right_actor = right_actor;
    _right_ren = right_ren;
    _pr = pr;
    refresh();
  }

  auto OnKeyPress() -> void override {
    auto key = std::string(this->Interactor->GetKeySym());
    auto n = Index(_pr->components.size());
    if (n == 0) {
      vtkInteractorStyleTrackballCamera::OnKeyPress();
      return;
    }
    if (key == "Right") {
      _idx = (_idx + 1) % n;
      refresh();
      this->Interactor->Render();
    } else if (key == "Left") {
      _idx = (_idx - 1 + n) % n;
      refresh();
      this->Interactor->Render();
    } else {
      vtkInteractorStyleTrackballCamera::OnKeyPress();
    }
  }

private:
  auto refresh() -> void {
    if (_pr->components.empty())
      return;

    // Left: every cell shown, the current one opaque, the rest ghosted.
    for (std::size_t i = 0; i < _left_actors.size(); ++i) {
      auto rgb = palette::mesh_color(_pr->comp_labels[i]);
      _left_actors[i]->GetProperty()->SetColor(rgb[0], rgb[1], rgb[2]);
      _left_actors[i]->GetProperty()->SetOpacity(
          Index(i) == _idx ? 1.0 : GHOST_OPACITY);
    }

    // Right: the current cell on its own.
    auto label = _pr->comp_labels[_idx];
    auto poly = tf::vtk::make_vtk_polydata(_pr->components[_idx]);
    _right_mapper->SetInputData(poly);
    auto rgb = palette::mesh_color(label);
    _right_actor->GetProperty()->SetColor(rgb[0], rgb[1], rgb[2]);
    _right_ren->ResetCamera();

    std::printf("domain %d  (%d/%d)\n", int(label), int(_idx + 1),
                int(_pr->components.size()));
  }

  std::vector<vtkSmartPointer<vtkOpenGLActor>> _left_actors;
  vtkOpenGLPolyDataMapper *_right_mapper{nullptr};
  vtkOpenGLActor *_right_actor{nullptr};
  vtkOpenGLRenderer *_right_ren{nullptr};
  PipelineResult *_pr{nullptr};
  Index _idx{0};
};
vtkStandardNewMacro(domain_cycle_interactor);

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------
int main(int argc, char *argv[]) {
  auto tri = tf::triangulation_type::cdt;
  std::vector<std::string> args;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--tri=refined" || a == "--refined")
      tri = tf::triangulation_type::refined_cdt;
    else if (a == "--tri=cdt")
      tri = tf::triangulation_type::cdt;
    else
      args.push_back(std::move(a));
  }
  auto paths = collect_obj_paths(args);
  if (paths.empty()) {
    std::fprintf(stderr,
                 "Usage:\n  %s [--tri=cdt|--tri=refined] <dir>\n  %s "
                 "[--tri=refined] a.obj b.obj ...\n",
                 argv[0], argv[0]);
    return 1;
  }

  auto pr = build_pipeline(paths, tri);
  if (pr.components.empty()) {
    std::fprintf(stderr, "no domains extracted\n");
    return 1;
  }

  // ---- Left viewport: one actor per domain cell ----
  vtkNew<vtkOpenGLRenderer> left_ren;
  std::vector<vtkSmartPointer<vtkOpenGLActor>> left_actors;
  left_actors.reserve(pr.components.size());
  for (auto &cell : pr.components) {
    auto poly = tf::vtk::make_vtk_polydata(cell);
    vtkNew<vtkOpenGLPolyDataMapper> mapper;
    mapper->SetInputData(poly);
    auto actor = vtkSmartPointer<vtkOpenGLActor>::New();
    actor->SetMapper(mapper);
    actor->GetProperty()->EdgeVisibilityOn();
    actor->GetProperty()->SetEdgeColor(0.0, 0.0, 0.0);
    actor->GetProperty()->SetLineWidth(0.5);
    left_ren->AddActor(actor);
    left_actors.push_back(actor);
  }
  left_ren->SetBackground(palette::BG_LEFT[0], palette::BG_LEFT[1],
                          palette::BG_LEFT[2]);
  left_ren->SetViewport(0.0, 0.0, 0.5, 1.0);

  // ---- Right viewport: current domain cell ----
  vtkNew<vtkOpenGLPolyDataMapper> right_mapper;
  vtkNew<vtkOpenGLActor> right_actor;
  right_actor->SetMapper(right_mapper);
  right_actor->GetProperty()->EdgeVisibilityOn();
  right_actor->GetProperty()->SetEdgeColor(0.0, 0.0, 0.0);
  right_actor->GetProperty()->SetLineWidth(1.0);

  vtkNew<vtkOpenGLRenderer> right_ren;
  right_ren->AddActor(right_actor);
  right_ren->SetBackground(palette::BG_RIGHT[0], palette::BG_RIGHT[1],
                           palette::BG_RIGHT[2]);
  right_ren->SetViewport(0.5, 0.0, 1.0, 1.0);

  vtkNew<vtkRenderWindow> window;
  window->AddRenderer(left_ren);
  window->AddRenderer(right_ren);
  window->SetSize(1600, 900);

  vtkNew<vtkRenderWindowInteractor> interactor;
  interactor->SetRenderWindow(window);

  vtkNew<domain_cycle_interactor> style;
  style->initialize(std::move(left_actors), right_mapper, right_actor, right_ren,
                    &pr);
  interactor->SetInteractorStyle(style);

  left_ren->ResetCamera();
  right_ren->ResetCamera();
  window->Render();
  interactor->Start();
  return 0;
}
