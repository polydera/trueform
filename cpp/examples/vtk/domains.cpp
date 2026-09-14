/*
 * Volumetric domain extraction for two generated double-precision scenes.
 *
 * Left/Right cycle domains. Up/Down cycle the generated scenes.
 */
#include "util/vtk_examples.hpp"

#include <trueform/cpp/arrangement/mesh_arrangements.hpp>
#include <trueform/cpp/arrangement/polygon_arrangements.hpp>
#include <trueform/cpp/clean.hpp>
#include <trueform/cpp/core/cache.hpp>
#include <trueform/cpp/core/index_type.hpp>
#include <trueform/cpp/core/mesh.hpp>
#include <trueform/cpp/geometry/make_plane_mesh.hpp>
#include <trueform/cpp/geometry/make_sphere_mesh.hpp>
#include <trueform/cpp/geometry/make_tube_mesh.hpp>
#include <trueform/cpp/reindex/split_into_domains.hpp>
#include <trueform/cpp/topology.hpp>
#include <trueform/cpp/topology/domain_labels.hpp>

#include <vtkCellData.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkObjectFactory.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkTextActor.h>
#include <vtkUnsignedCharArray.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace examples = trueform::vtk_examples;
using index_t = tf::cpp::default_index_t;
using storage = tf::polygons_buffer<index_t, double, 3, 3>;
using cache_t = tf::cpp::cache<index_t, double>;
using mesh_t = tf::cpp::mesh<index_t, double>;
using domain_labels = tf::cpp::domain_labels_result<index_t>;

constexpr std::array<std::array<double, 3>, 8> palette{{
    {0.85, 0.20, 0.20},
    {0.20, 0.40, 0.85},
    {0.20, 0.75, 0.20},
    {0.95, 0.75, 0.10},
    {0.75, 0.30, 0.85},
    {0.20, 0.85, 0.85},
    {0.95, 0.55, 0.10},
    {0.55, 0.55, 0.60},
}};
constexpr std::array<unsigned char, 3> sentinel_rgb{77, 77, 82};

struct scene {
  std::string name;
  std::vector<storage> parts;
};

/// The arrangement is storage this result holds, so it holds the cache that
/// remembers it too, and `arrangement_mesh()` is the assembly the reads take.
struct pipeline_result {
  storage arrangement;
  mutable cache_t cache;
  domain_labels labels;
  std::vector<storage> components;
  std::vector<std::int32_t> component_labels;

  auto arrangement_mesh() const -> mesh_t {
    return {arrangement.faces(), arrangement.points(), cache};
  }
};

auto color(std::int32_t label) -> std::array<double, 3> {
  return palette[static_cast<std::size_t>(label) % palette.size()];
}

auto color_u8(std::int32_t label) -> std::array<unsigned char, 3> {
  const auto value = color(label);
  return {static_cast<unsigned char>(value[0] * 255.0),
          static_cast<unsigned char>(value[1] * 255.0),
          static_cast<unsigned char>(value[2] * 255.0)};
}

auto translate(storage value, double dx, double dy, double dz) -> storage {
  for (auto point : value.points()) {
    point[0] += dx;
    point[1] += dy;
    point[2] += dz;
  }
  return value;
}

auto swap_axes(storage value, const std::array<int, 3> &permutation)
    -> storage {
  for (auto point : value.points()) {
    const std::array<double, 3> source{point[0], point[1], point[2]};
    for (std::size_t axis = 0; axis != 3; ++axis)
      point[axis] = source[static_cast<std::size_t>(permutation[axis])];
  }
  return value;
}

auto sphere(double radius) -> storage {
  return tf::cpp::make_sphere_mesh(radius, 32, 32);
}

auto plane(double side) -> storage {
  return tf::cpp::make_plane_mesh(side, side);
}

auto make_scene(std::size_t index) -> scene {
  if (index == 0) {
    std::vector<storage> parts;
    parts.push_back(sphere(1.0));
    parts.push_back(plane(3.0));
    parts.push_back(translate(sphere(0.2), 0.0, 0.0, -0.5));
    return {"sphere + plane + small sphere (z=-0.5)", std::move(parts)};
  }

  std::vector<storage> parts;
  parts.push_back(sphere(1.0));
  parts.push_back(translate(plane(3.0), 0.0, 0.0, 0.3));
  parts.push_back(translate(plane(3.0), 0.0, 0.0, -0.3));
  parts.push_back(swap_axes(plane(3.0), {0, 2, 1}));
  parts.push_back(swap_axes(plane(3.0), {2, 1, 0}));
  return {"sphere + 4 planes", std::move(parts)};
}

auto build_pipeline(const std::vector<storage> &parts) -> pipeline_result {
  // One cache per operand, held here for as long as the operands are read.
  std::vector<cache_t> caches(parts.size());
  std::vector<mesh_t> operands;
  operands.reserve(parts.size());
  for (std::size_t part = 0; part != parts.size(); ++part)
    operands.push_back(
        {parts[part].faces(), parts[part].points(), caches[part]});
  const auto arrangement_result = tf::cpp::mesh_arrangements(operands);

  cache_t raw_cache;
  const mesh_t raw{arrangement_result.mesh.faces(),
                   arrangement_result.mesh.points(), raw_cache};

  pipeline_result result;
  result.arrangement = tf::cpp::cleaned_mesh(raw, 1.0e-6);
  result.labels = tf::cpp::make_domain_labels(
      result.arrangement_mesh(), tf::domain_config::ignore_open_fragments);
  auto split =
      tf::cpp::split_into_domains(result.arrangement_mesh(), result.labels);
  result.components = std::move(split.components);
  result.component_labels.assign(split.labels.begin(), split.labels.end());
  return result;
}

auto aabb_diagonal(const tf::points_buffer<double, 3> &points) -> double {
  const auto [lower, upper] = examples::point_bounds(points);
  const auto dx = upper[0] - lower[0];
  const auto dy = upper[1] - lower[1];
  const auto dz = upper[2] - lower[2];
  return std::sqrt(dx * dx + dy * dy + dz * dz);
}

auto make_arrangement_actor(const storage &value)
    -> std::pair<vtkSmartPointer<vtkActor>, vtkSmartPointer<vtkPolyData>> {
  auto poly = examples::to_vtk_polydata(value);
  auto actor = examples::make_actor(poly);
  auto *mapper = vtkPolyDataMapper::SafeDownCast(actor->GetMapper());
  mapper->SetScalarModeToUseCellData();
  actor->GetProperty()->EdgeVisibilityOn();
  actor->GetProperty()->SetEdgeColor(0.0, 0.0, 0.0);
  actor->GetProperty()->SetLineWidth(0.5);
  return {std::move(actor), std::move(poly)};
}

auto make_domain_actor(const storage &value, std::int32_t label)
    -> vtkSmartPointer<vtkActor> {
  auto actor =
      examples::make_actor(examples::to_vtk_polydata(value), color(label));
  actor->GetProperty()->EdgeVisibilityOn();
  actor->GetProperty()->SetEdgeColor(0.0, 0.0, 0.0);
  actor->GetProperty()->SetLineWidth(1.0);
  return actor;
}

auto make_tube_actor(const pipeline_result &value)
    -> vtkSmartPointer<vtkActor> {
  auto edges = tf::cpp::non_manifold_edges(value.arrangement_mesh());
  if (edges.shape_at(0) == 0)
    return nullptr;
  auto paths = tf::cpp::connect_edges_to_paths(edges);
  if (paths.size() == 0)
    return nullptr;
  const auto radius = std::max(
      aabb_diagonal(value.arrangement.points_buffer()) * 0.001, 1.0e-6);
  auto tubes = tf::cpp::make_tube_mesh(
      paths, examples::points_array(value.arrangement.points_buffer()), radius,
      12);
  auto actor =
      examples::make_actor(examples::to_vtk_polydata(tubes), {1.0, 0.85, 0.0});
  actor->GetProperty()->LightingOff();
  return actor;
}

auto apply_highlight(vtkPolyData *poly, const domain_labels &labels,
                     std::int32_t current_domain) -> void {
  auto colors = vtkSmartPointer<vtkUnsignedCharArray>::New();
  colors->SetNumberOfComponents(4);
  colors->SetNumberOfTuples(poly->GetNumberOfCells());
  const auto number_of_domains = labels.number_of_domains();
  for (vtkIdType face = 0; face != poly->GetNumberOfCells(); ++face) {
    const auto d0 = labels.label(static_cast<std::int32_t>(face), 0);
    const auto d1 = labels.label(static_cast<std::int32_t>(face), 1);
    const auto selected = d0 == current_domain || d1 == current_domain;
    auto rgb = sentinel_rgb;
    if (d0 < number_of_domains || d1 < number_of_domains) {
      const auto base = d0 < number_of_domains ? d0 : d1;
      rgb = color_u8(base);
    }
    if (selected)
      rgb = color_u8(current_domain);
    colors->SetTuple4(face, rgb[0], rgb[1], rgb[2], selected ? 255 : 30);
  }
  poly->GetCellData()->SetScalars(colors);
  poly->GetCellData()->Modified();
  poly->Modified();
}

class domains_interactor : public vtkInteractorStyleTrackballCamera {
public:
  static auto New() -> domains_interactor *;
  vtkTypeMacro(domains_interactor, vtkInteractorStyleTrackballCamera);

  auto initialize(vtkRenderer *left, vtkRenderer *right, vtkTextActor *text)
      -> void {
    _left = left;
    _right = right;
    _text = text;
    load_current_scene();
  }

  auto OnKeyPress() -> void override {
    const auto key = std::string(this->Interactor->GetKeySym());
    if (key == "Right" && !_pipeline.components.empty()) {
      _domain = (_domain + 1) % _pipeline.components.size();
      refresh_current();
      this->Interactor->Render();
      return;
    }
    if (key == "Left" && !_pipeline.components.empty()) {
      _domain = (_domain + _pipeline.components.size() - 1) %
                _pipeline.components.size();
      refresh_current();
      this->Interactor->Render();
      return;
    }
    if (key == "Up" || key == "Down") {
      _scene = key == "Up" ? (_scene + 1) % 2 : (_scene + 2 - 1) % 2;
      load_current_scene();
      return;
    }
    vtkInteractorStyleTrackballCamera::OnKeyPress();
  }

private:
  auto clear() -> void {
    for (const auto &actor : _left_actors)
      _left->RemoveActor(actor);
    for (const auto &actor : _right_actors)
      _right->RemoveActor(actor);
    _left_actors.clear();
    _right_actors.clear();
  }

  auto load_current_scene() -> void {
    clear();
    auto current = make_scene(_scene);
    _scene_name = current.name;
    _pipeline = build_pipeline(current.parts);
    _domain = 0;

    auto [arrangement_actor, arrangement_poly] =
        make_arrangement_actor(_pipeline.arrangement);
    _left_poly = arrangement_poly;
    _left_actors.push_back(arrangement_actor);
    if (auto tubes = make_tube_actor(_pipeline))
      _left_actors.push_back(tubes);
    for (const auto &actor : _left_actors)
      _left->AddActor(actor);

    refresh_current();
    _left->ResetCamera();
    _right->ResetCamera();
    if (this->Interactor)
      this->Interactor->Render();
  }

  auto refresh_current() -> void {
    for (const auto &actor : _right_actors)
      _right->RemoveActor(actor);
    _right_actors.clear();

    if (!_pipeline.components.empty()) {
      const auto label = _pipeline.component_labels[_domain];
      auto actor = make_domain_actor(_pipeline.components[_domain], label);
      _right_actors.push_back(actor);
      _right->AddActor(actor);
      apply_highlight(_left_poly, _pipeline.labels, label);
    }
    update_text();
  }

  auto update_text() -> void {
    std::string message;
    if (_pipeline.components.empty()) {
      message = "[" + std::to_string(_scene + 1) + "/2] " + _scene_name +
                "  —  no domains";
    } else {
      const auto label = _pipeline.component_labels[_domain];
      message = "[" + std::to_string(_scene + 1) + "/2] " + _scene_name +
                "  —  domain " + std::to_string(label) + "  (" +
                std::to_string(_domain + 1) + "/" +
                std::to_string(_pipeline.components.size()) + " of n_domains=" +
                std::to_string(_pipeline.labels.number_of_domains()) +
                ")    [Left/Right: cycle domain   Up/Down: cycle scene]";
    }
    _text->SetInput(message.c_str());
  }

  vtkRenderer *_left = nullptr;
  vtkRenderer *_right = nullptr;
  vtkTextActor *_text = nullptr;
  std::size_t _scene = 0;
  std::size_t _domain = 0;
  std::string _scene_name;
  pipeline_result _pipeline;
  vtkSmartPointer<vtkPolyData> _left_poly;
  std::vector<vtkSmartPointer<vtkActor>> _left_actors;
  std::vector<vtkSmartPointer<vtkActor>> _right_actors;
};

vtkStandardNewMacro(domains_interactor);

} // namespace

int main() {
  try {
    auto left = vtkSmartPointer<vtkRenderer>::New();
    left->SetViewport(0.0, 0.0, 0.5, 1.0);
    left->SetBackground(0.15, 0.10, 0.10);

    auto right = vtkSmartPointer<vtkRenderer>::New();
    right->SetViewport(0.5, 0.0, 1.0, 1.0);
    right->SetBackground(0.08, 0.12, 0.08);

    auto text_renderer = vtkSmartPointer<vtkRenderer>::New();
    text_renderer->SetViewport(0.0, 0.93, 1.0, 1.0);
    text_renderer->SetBackground(0.05, 0.05, 0.05);
    text_renderer->InteractiveOff();
    auto text = examples::make_text_actor("loading…", 22, {0.02, 0.30});
    text_renderer->AddViewProp(text);

    auto window = vtkSmartPointer<vtkRenderWindow>::New();
    window->AddRenderer(left);
    window->AddRenderer(right);
    window->AddRenderer(text_renderer);
    window->SetSize(1600, 900);

    auto interactor = vtkSmartPointer<vtkRenderWindowInteractor>::New();
    interactor->SetRenderWindow(window);
    auto style = vtkSmartPointer<domains_interactor>::New();
    style->initialize(left, right, text);
    interactor->SetInteractorStyle(style);

    window->Render();
    interactor->Start();
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
