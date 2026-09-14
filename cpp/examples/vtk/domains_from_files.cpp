/*
 * Volumetric domain extraction from OBJ files using the public C++ facade.
 *
 * Usage:
 *   domains_from_files DIRECTORY
 *   domains_from_files a.obj b.obj ...
 *
 * Left/Right cycle the extracted domains.
 */
#include "util/vtk_examples.hpp"

#include <trueform/cpp/arrangement/mesh_arrangements.hpp>
#include <trueform/cpp/arrangement/polygon_arrangements.hpp>
#include <trueform/cpp/clean.hpp>
#include <trueform/cpp/core/cache.hpp>
#include <trueform/cpp/core/index_type.hpp>
#include <trueform/cpp/core/mesh.hpp>
#include <trueform/cpp/geometry/make_tube_mesh.hpp>
#include <trueform/cpp/geometry/signed_volume.hpp>
#include <trueform/cpp/io/read_obj.hpp>
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
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
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

auto milliseconds(double seconds) -> std::string {
  std::ostringstream stream;
  stream << std::fixed << std::setprecision(1) << seconds * 1000.0 << " ms";
  return stream.str();
}

auto lower(std::string value) -> std::string {
  std::transform(
      value.begin(), value.end(), value.begin(),
      [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

auto collect_obj_paths(const std::vector<std::string> &arguments)
    -> std::vector<std::filesystem::path> {
  if (arguments.size() == 1 &&
      std::filesystem::is_directory(arguments.front())) {
    std::vector<std::filesystem::path> result;
    for (const auto &entry :
         std::filesystem::directory_iterator(arguments.front()))
      if (lower(entry.path().extension().string()) == ".obj")
        result.push_back(entry.path());
    std::sort(result.begin(), result.end());
    return result;
  }
  std::vector<std::filesystem::path> result;
  result.reserve(arguments.size());
  for (const auto &argument : arguments)
    result.emplace_back(argument);
  return result;
}

auto load_meshes(const std::vector<std::filesystem::path> &paths)
    -> std::vector<storage> {
  std::vector<storage> meshes;
  meshes.reserve(paths.size());
  const auto total_start = std::chrono::steady_clock::now();
  for (const auto &path : paths) {
    const auto start = std::chrono::steady_clock::now();
    auto value = tf::cpp::read_obj<index_t, double, 3>(path.string());
    std::cout << "  loaded " << path.filename().string() << ": "
              << value.points_buffer().size() << " pts, " << value.size()
              << " tris  (" << milliseconds(examples::elapsed_seconds(start))
              << ")\n";
    meshes.push_back(std::move(value));
  }
  std::cout << "  total read_obj: "
            << milliseconds(examples::elapsed_seconds(total_start)) << '\n';
  return meshes;
}

auto max_z(const storage &value) -> double {
  auto result = -std::numeric_limits<double>::infinity();
  for (const auto point : value.points())
    result = std::max(result, point[2]);
  return result;
}

auto build_pipeline(const std::vector<storage> &parts) -> pipeline_result {
  std::cout << "running mesh_arrangements on " << parts.size()
            << " meshes...\n";
  auto start = std::chrono::steady_clock::now();
  // One cache per operand, held here for as long as the operands are read.
  std::vector<cache_t> caches(parts.size());
  std::vector<mesh_t> operands;
  operands.reserve(parts.size());
  for (std::size_t part = 0; part != parts.size(); ++part)
    operands.push_back(
        {parts[part].faces(), parts[part].points(), caches[part]});
  const auto arrangement_result = tf::cpp::mesh_arrangements(operands);
  std::cout << "  arrangement: "
            << arrangement_result.mesh.points_buffer().size() << " pts, "
            << arrangement_result.mesh.size() << " tris  ("
            << milliseconds(examples::elapsed_seconds(start)) << ")\n";

  std::cout << "running cleaned...\n";
  start = std::chrono::steady_clock::now();
  cache_t raw_cache;
  const mesh_t raw{arrangement_result.mesh.faces(),
                   arrangement_result.mesh.points(), raw_cache};
  pipeline_result result;
  result.arrangement = tf::cpp::cleaned_mesh(raw, 1.0e-6);
  std::cout << "  cleaned: " << result.arrangement.points_buffer().size()
            << " pts, " << result.arrangement.size() << " tris  ("
            << milliseconds(examples::elapsed_seconds(start)) << ")\n";

  std::cout << "running domain_labels...\n";
  start = std::chrono::steady_clock::now();
  result.labels = tf::cpp::make_domain_labels(
      result.arrangement_mesh(), tf::domain_config::ignore_open_fragments);
  std::cout << "  " << result.labels.number_of_domains() << " domains  ("
            << milliseconds(examples::elapsed_seconds(start)) << ")\n";

  std::cout << "running split_into_domains...\n";
  start = std::chrono::steady_clock::now();
  auto split =
      tf::cpp::split_into_domains(result.arrangement_mesh(), result.labels);
  std::cout << "  " << split.components.size() << " components  ("
            << milliseconds(examples::elapsed_seconds(start)) << ")\n";

  std::vector<std::int32_t> component_labels(split.labels.begin(),
                                             split.labels.end());
  if (!split.components.empty()) {
    std::vector<std::size_t> order(split.components.size());
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::vector<double> volumes(split.components.size());
    std::vector<double> maxima(split.components.size());
    for (std::size_t index = 0; index != split.components.size(); ++index) {
      const auto &component = split.components[index];
      cache_t component_cache;
      volumes[index] = tf::cpp::signed_volume(
          mesh_t{component.faces(), component.points(), component_cache});
      maxima[index] = max_z(component);
    }
    const auto outer = *std::min_element(
        order.begin(), order.end(), [&](std::size_t left, std::size_t right) {
          return volumes[left] < volumes[right];
        });
    order.erase(std::find(order.begin(), order.end(), outer));
    std::sort(order.begin(), order.end(),
              [&](std::size_t left, std::size_t right) {
                return maxima[left] > maxima[right];
              });
    order.insert(order.begin(), outer);

    std::vector<storage> reordered;
    std::vector<std::int32_t> reordered_labels;
    reordered.reserve(order.size());
    reordered_labels.reserve(order.size());
    for (const auto index : order) {
      reordered.push_back(std::move(split.components[index]));
      reordered_labels.push_back(component_labels[index]);
    }
    split.components = std::move(reordered);
    component_labels = std::move(reordered_labels);
  }

  result.components = std::move(split.components);
  result.component_labels = std::move(component_labels);
  return result;
}

auto aabb_diagonal(const tf::points_buffer<double, 3> &points) -> double {
  const auto [lower_bound, upper_bound] = examples::point_bounds(points);
  const auto dx = upper_bound[0] - lower_bound[0];
  const auto dy = upper_bound[1] - lower_bound[1];
  const auto dz = upper_bound[2] - lower_bound[2];
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
  actor->GetProperty()->SetOpacity(0.6);
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
    if (d0 < number_of_domains || d1 < number_of_domains)
      rgb = color_u8(d0 < number_of_domains ? d0 : d1);
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

  auto initialize(vtkRenderer *left, vtkRenderer *right, vtkTextActor *text,
                  pipeline_result pipeline, std::string scene_name) -> void {
    _left = left;
    _right = right;
    _text = text;
    _pipeline = std::move(pipeline);
    _scene_name = std::move(scene_name);

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
    vtkInteractorStyleTrackballCamera::OnKeyPress();
  }

private:
  auto refresh_current() -> void {
    for (const auto &actor : _right_actors)
      _right->RemoveActor(actor);
    _right_actors.clear();
    if (!_pipeline.components.empty()) {
      const auto label = _pipeline.component_labels[_domain];
      auto actor = make_domain_actor(_pipeline.components[_domain], label);
      _right_actors.push_back(actor);
      _right->AddActor(actor);
      _right->ResetCamera();
      apply_highlight(_left_poly, _pipeline.labels, label);
    }
    update_text();
  }

  auto update_text() -> void {
    std::string message;
    if (_pipeline.components.empty()) {
      message = _scene_name + "  —  no domains";
    } else {
      const auto label = _pipeline.component_labels[_domain];
      message = _scene_name + "  —  domain " + std::to_string(label) + "  (" +
                std::to_string(_domain + 1) + "/" +
                std::to_string(_pipeline.components.size()) + " of n_domains=" +
                std::to_string(_pipeline.labels.number_of_domains()) +
                ")    [Left/Right: cycle domain]";
    }
    _text->SetInput(message.c_str());
  }

  vtkRenderer *_left = nullptr;
  vtkRenderer *_right = nullptr;
  vtkTextActor *_text = nullptr;
  std::size_t _domain = 0;
  std::string _scene_name;
  pipeline_result _pipeline;
  vtkSmartPointer<vtkPolyData> _left_poly;
  std::vector<vtkSmartPointer<vtkActor>> _left_actors;
  std::vector<vtkSmartPointer<vtkActor>> _right_actors;
};

vtkStandardNewMacro(domains_interactor);

void print_usage(const char *program) {
  std::cerr << "Usage:\n"
            << "  " << program << " /path/to/inputs/\n"
            << "  " << program << " foo.obj bar.obj baz.obj\n";
}

auto directory_name(const std::string &argument) -> std::string {
  auto path = std::filesystem::path(argument).lexically_normal();
  while (!path.empty() && (path.filename().empty() || path.filename() == "."))
    path = path.parent_path();
  return path.filename().string();
}

} // namespace

int main(int argc, char **argv) {
  if (argc == 2 &&
      (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help")) {
    print_usage(argv[0]);
    return 0;
  }
  if (argc < 2) {
    print_usage(argv[0]);
    return 1;
  }

  try {
    std::vector<std::string> arguments;
    for (int index = 1; index != argc; ++index)
      arguments.emplace_back(argv[index]);
    const auto paths = collect_obj_paths(arguments);
    if (paths.empty()) {
      std::cerr << "no .obj files found\n";
      return 1;
    }

    std::cout << "loading " << paths.size() << " files...\n";
    const auto meshes = load_meshes(paths);
    auto pipeline = build_pipeline(meshes);
    const auto scene_name =
        arguments.size() == 1 &&
                std::filesystem::is_directory(arguments.front())
            ? directory_name(arguments.front())
            : std::to_string(paths.size()) + " files";

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
    style->initialize(left, right, text, std::move(pipeline), scene_name);
    interactor->SetInteractorStyle(style);

    window->Render();
    interactor->Start();
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
