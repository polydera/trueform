/*
* Copyright (c) 2025 XLAB
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
// Volumetric domain extraction from a set of input OBJ files.
//
// Usage:
//   domains <dir>           # load every .obj from <dir>
//   domains a.obj b.obj ... # load specific files
//
// Pipeline:
//   read_obj per file → make_mesh_arrangements → cleaned → make_domain_labels
//   (ignore_open_fragments) → split_into_domains.
//
// Viewer (split-screen):
//   Left  — full arrangement, faces ghosted except those bounding the
//           current domain. Non-manifold edges drawn as warm tubes.
//   Right — current domain submesh, opaque, outward-oriented.
//
//   Left / Right keys cycle the current domain. The outer shell (most
//   negative signed_volume) is shown first; remaining domains are
//   ordered top→bottom by their max-Z vertex.

#include <trueform/trueform.hpp>
#include <trueform/vtk.hpp>

#include <vtkCellData.h>
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
#include <vtkUnsignedCharArray.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

using Index = int;
using Real = double;
using Mesh = tf::polygons_buffer<Index, Real, 3, 3>;

// Weld tolerance applied to each input mesh on load and to the
// arrangement output. Matches what test_domains.cpp / the Python example
// use for the geological-model inputs.
constexpr Real WELD_TOLERANCE = 1e-6;

// Cross-section sides for the non-manifold-edge tubes drawn in the
// left viewport.
constexpr Index NM_TUBE_SEGMENTS = 12;

// Polydera color scheme — dark teal-tinted backgrounds + warm/cool
// accents tuned to read well against them.
namespace palette {

constexpr std::array<double, 3> BG_LEFT  = {0.047, 0.082, 0.075};  // bg-0
constexpr std::array<double, 3> BG_RIGHT = {0.071, 0.122, 0.114};  // bg-1
constexpr std::array<double, 3> TUBE     = {0.949, 0.271, 0.412};  // rose accent
constexpr std::array<unsigned char, 3> SENTINEL = {30, 50, 47};

// Mesh palette — designed to be distinguishable on the dark teal
// backdrop. Cycles through teal, rose, amber, violet, sky, green, coral,
// neutral.
constexpr std::array<std::array<unsigned char, 3>, 8> MESH = {{
    { 45, 223, 206},  // teal-400
    {196,  69, 105},  // rose
    {242, 191,  26},  // amber
    {155, 115, 214},  // violet
    {110, 200, 230},  // sky
    {109, 204, 138},  // green
    {231, 110, 116},  // coral
    {180, 180, 188},  // neutral
}};

auto mesh_color(Index i) -> std::array<unsigned char, 3> {
  return MESH[static_cast<std::size_t>(i) % MESH.size()];
}

} // namespace palette

// -----------------------------------------------------------------------------
// argv → list of .obj paths (single directory, or explicit list of files)
// -----------------------------------------------------------------------------
auto collect_obj_paths(int argc, char *argv[]) -> std::vector<std::string> {
  std::vector<std::string> paths;
  if (argc == 2) {
    std::filesystem::path p(argv[1]);
    if (std::filesystem::is_directory(p)) {
      for (auto &entry : std::filesystem::directory_iterator(p))
        if (entry.path().extension() == ".obj")
          paths.push_back(entry.path().string());
      std::sort(paths.begin(), paths.end());
      return paths;
    }
  }
  for (int i = 1; i < argc; ++i)
    paths.emplace_back(argv[i]);
  return paths;
}

// -----------------------------------------------------------------------------
// Pipeline. All trueform calls operate on trueform types; the conversion
// to vtkPolyData only happens at display time.
// -----------------------------------------------------------------------------
struct PipelineResult {
  Mesh arrangement;
  tf::domain_labels<Index> labels;
  std::vector<Mesh> components;
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

// Arrangement of N meshes + clean to weld intersection-introduced
// duplicates.
auto build_arrangement(const std::vector<Mesh> &meshes, Real tolerance)
    -> Mesh {
  std::vector<decltype(meshes[0].polygons())> forms;
  forms.reserve(meshes.size());
  for (auto &m : meshes)
    forms.push_back(m.polygons());

  std::printf("running make_mesh_arrangements on %zu meshes...\n",
              forms.size());
  tf::tick();
  auto [mesh, _tags, _faces] = tf::make_mesh_arrangements(tf::make_range(forms));
  tf::tock("arrangement:");
  std::printf("  arrangement: %zu pts, %zu tris\n", mesh.points().size(),
              mesh.faces().size());

  std::printf("running cleaned...\n");
  tf::tick();
  mesh = tf::cleaned(mesh.polygons(), tolerance);
  tf::tock("cleaned:");
  std::printf("  cleaned: %zu pts, %zu tris\n", mesh.points().size(),
              mesh.faces().size());

  return mesh;
}

// Label volumetric domains and split into per-domain submeshes.
auto extract_domains(const Mesh &arrangement)
    -> std::tuple<tf::domain_labels<Index>, std::vector<Mesh>,
                  tf::buffer<Index>> {
  std::printf("running make_domain_labels...\n");
  tf::tick();
  auto labels = tf::make_domain_labels(
      arrangement.polygons(), tf::domain_config::ignore_open_fragments);
  tf::tock("domains:");
  std::printf("  %d domains\n", int(labels.n_domains));

  std::printf("running split_into_domains...\n");
  tf::tick();
  auto [components, comp_labels] =
      tf::split_into_domains(arrangement.polygons(), labels);
  tf::tock("split:");
  std::printf("  %zu components\n", components.size());

  return {std::move(labels), std::move(components), std::move(comp_labels)};
}

// Reorder components for display: outer shell (most negative
// signed_volume) first, then by max-Z descending (top→bottom).
auto sort_for_display(std::vector<Mesh> &components,
                      tf::buffer<Index> &comp_labels) -> void {
  if (components.empty())
    return;

  std::vector<Real> volumes(components.size());
  std::vector<Real> max_z(components.size());
  for (std::size_t i = 0; i < components.size(); ++i) {
    volumes[i] = static_cast<Real>(
        tf::signed_volume(components[i].polygons()));
    max_z[i] = tf::aabb_from(components[i].polygons()).max[2];
  }
  auto outer_idx = Index(
      std::min_element(volumes.begin(), volumes.end()) - volumes.begin());

  std::vector<Index> order;
  order.reserve(components.size());
  order.push_back(outer_idx);
  for (std::size_t i = 0; i < components.size(); ++i)
    if (Index(i) != outer_idx)
      order.push_back(Index(i));
  std::sort(order.begin() + 1, order.end(),
            [&](Index a, Index b) { return max_z[a] > max_z[b]; });

  std::vector<Mesh> reordered;
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

auto build_pipeline(const std::vector<std::string> &paths) -> PipelineResult {
  auto meshes = load_meshes(paths, WELD_TOLERANCE);
  auto arrangement = build_arrangement(meshes, WELD_TOLERANCE);
  auto [labels, components, comp_labels] = extract_domains(arrangement);
  sort_for_display(components, comp_labels);
  return {std::move(arrangement), std::move(labels), std::move(components),
          std::move(comp_labels)};
}

// -----------------------------------------------------------------------------
// VTK display helpers
// -----------------------------------------------------------------------------
auto apply_highlight(vtkPolyData *poly, const tf::domain_labels<Index> &labels,
                     Index current) -> void {
  auto n_faces = poly->GetNumberOfPolys();
  vtkNew<vtkUnsignedCharArray> colors;
  colors->SetNumberOfComponents(4);
  colors->SetNumberOfTuples(n_faces);
  auto n_dom = labels.n_domains;
  for (vtkIdType f = 0; f < n_faces; ++f) {
    auto d0 = labels.labels[Index(f)][0];
    auto d1 = labels.labels[Index(f)][1];
    bool in_dom = (d0 == current) || (d1 == current);
    Index base = (d0 < n_dom) ? d0 : (d1 < n_dom ? d1 : Index(0));
    auto base_rgb = (d0 >= n_dom && d1 >= n_dom)
                        ? palette::SENTINEL
                        : palette::mesh_color(base);
    auto cur_rgb = palette::mesh_color(current);
    auto rgb = in_dom ? cur_rgb : base_rgb;
    colors->SetTuple4(f, rgb[0], rgb[1], rgb[2], in_dom ? 255 : 30);
  }
  poly->GetCellData()->SetScalars(colors);
  poly->GetCellData()->Modified();
}

auto build_nm_tube_actor(const Mesh &mesh, Real radius)
    -> vtkSmartPointer<vtkOpenGLActor> {
  auto nm = tf::make_non_manifold_edges(mesh.polygons());
  auto paths = tf::connect_edges_to_paths(tf::make_edges(nm));
  if (paths.size() == 0)
    return nullptr;

  auto tubes = tf::make_tube_mesh(tf::make_curves(paths, mesh.points()),
                                  radius, NM_TUBE_SEGMENTS);
  auto poly = tf::vtk::make_vtk_polydata(tubes);

  vtkNew<vtkOpenGLPolyDataMapper> mapper;
  mapper->SetInputData(poly);
  auto actor = vtkSmartPointer<vtkOpenGLActor>::New();
  actor->SetMapper(mapper);
  actor->GetProperty()->SetColor(palette::TUBE[0], palette::TUBE[1],
                                 palette::TUBE[2]);
  actor->GetProperty()->SetOpacity(0.7);
  actor->GetProperty()->LightingOff();
  return actor;
}

// -----------------------------------------------------------------------------
// Interactor — Left/Right cycle the current domain
// -----------------------------------------------------------------------------
class domain_cycle_interactor : public vtkInteractorStyleTrackballCamera {
public:
  static auto New() -> domain_cycle_interactor *;
  vtkTypeMacro(domain_cycle_interactor, vtkInteractorStyleTrackballCamera);

  auto initialize(vtkPolyData *left_poly, vtkOpenGLPolyDataMapper *right_mapper,
                  vtkOpenGLActor *right_actor, vtkOpenGLRenderer *right_ren,
                  PipelineResult *pr) -> void {
    _left_poly = left_poly;
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
    auto label = _pr->comp_labels[_idx];

    auto &sub = _pr->components[_idx];
    auto poly = tf::vtk::make_vtk_polydata(sub);
    _right_mapper->SetInputData(poly);
    auto rgb = palette::mesh_color(label);
    _right_actor->GetProperty()->SetColor(rgb[0] / 255.0, rgb[1] / 255.0,
                                          rgb[2] / 255.0);
    _right_ren->ResetCamera();

    apply_highlight(_left_poly, _pr->labels, label);

    std::printf("domain %d  (%d/%d of n_domains=%d)\n", int(label),
                int(_idx + 1), int(_pr->components.size()),
                int(_pr->labels.n_domains));
  }

  vtkPolyData *_left_poly{nullptr};
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
  auto paths = collect_obj_paths(argc, argv);
  if (paths.empty()) {
    std::fprintf(stderr,
                 "Usage:\n  %s <dir>\n  %s a.obj b.obj ...\n",
                 argv[0], argv[0]);
    return 1;
  }

  auto pr = build_pipeline(paths);
  Real diag = tf::aabb_from(pr.arrangement.polygons()).diagonal().length();
  Real tube_radius = std::max(diag * Real(0.001), Real(1e-6));

  // ---- Left viewport: arrangement + NM-edge tubes ----
  auto left_poly = tf::vtk::make_vtk_polydata(pr.arrangement);
  vtkNew<vtkOpenGLPolyDataMapper> left_mapper;
  left_mapper->SetInputData(left_poly);
  left_mapper->SetScalarModeToUseCellData();
  vtkNew<vtkOpenGLActor> left_actor;
  left_actor->SetMapper(left_mapper);
  left_actor->GetProperty()->EdgeVisibilityOn();
  left_actor->GetProperty()->SetEdgeColor(0.0, 0.0, 0.0);
  left_actor->GetProperty()->SetLineWidth(0.5);

  vtkNew<vtkOpenGLRenderer> left_ren;
  left_ren->AddActor(left_actor);
  if (auto tubes = build_nm_tube_actor(pr.arrangement, tube_radius); tubes)
    left_ren->AddActor(tubes);
  left_ren->SetBackground(palette::BG_LEFT[0], palette::BG_LEFT[1],
                          palette::BG_LEFT[2]);
  left_ren->SetViewport(0.0, 0.0, 0.5, 1.0);

  // ---- Right viewport: current domain submesh ----
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
  style->initialize(left_poly, right_mapper, right_actor, right_ren, &pr);
  interactor->SetInteractorStyle(style);

  left_ren->ResetCamera();
  right_ren->ResetCamera();
  window->Render();
  interactor->Start();
  return 0;
}
