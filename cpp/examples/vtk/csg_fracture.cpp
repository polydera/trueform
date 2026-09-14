/*
 * Cellular fracture from one reusable CSG arrangement.
 *
 * Usage: csg_fracture MESH [cuts] [explode] [--all] [--png]
 *   cuts    axis-aligned cuts per axis                         [default 12]
 *   explode outward displacement as a fraction of extent       [default 0.0]
 *   --all   render every bounded domain instead of operand 0's interior
 *   --png   render RGBA offscreen to the temporary fracture.png path
 */
#include "util/vtk_examples.hpp"

#include <trueform/cpp/clean.hpp>
#include <trueform/cpp/core/cache.hpp>
#include <trueform/cpp/core/index_type.hpp>
#include <trueform/cpp/core/mesh.hpp>
#include <trueform/cpp/csg/csg_domains.hpp>
#include <trueform/cpp/csg/csg_graph.hpp>
#include <trueform/cpp/geometry/make_plane_mesh.hpp>
#include <trueform/cpp/io/read_obj.hpp>
#include <trueform/cpp/io/read_stl.hpp>
#include <trueform/cpp/topology.hpp>

#include <vtkCamera.h>
#include <vtkErrorCode.h>
#include <vtkFeatureEdges.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkLightKit.h>
#include <vtkObjectFactory.h>
#include <vtkPNGWriter.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkWindowToImageFilter.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace examples = trueform::vtk_examples;
using index_t = tf::cpp::default_index_t;
using storage = tf::polygons_buffer<index_t, double, 3, 3>;
using cache_t = tf::cpp::cache<index_t, double>;
using mesh_t = tf::cpp::mesh<index_t, double>;
using point = examples::point3<double>;

constexpr std::array<std::array<unsigned char, 3>, 12> palette{{
    {0x8F, 0xE3, 0xD6},
    {0xEF, 0xA9, 0xC4},
    {0x8F, 0xB4, 0xD8},
    {0xF6, 0xBE, 0x84},
    {0x8F, 0xC8, 0x82},
    {0xB9, 0xA0, 0xDC},
    {0xEF, 0x9A, 0x98},
    {0x6F, 0xC3, 0xB4},
    {0xF2, 0xCE, 0x86},
    {0xA9, 0xC4, 0xE6},
    {0xA6, 0xDD, 0xBE},
    {0xD6, 0xA9, 0xCE},
}};
constexpr std::array<double, 3> edge_rgb{0.18, 0.20, 0.24};

struct camera_settings {
  point position;
  point focal_point;
  point view_up;
  double view_angle;
  std::array<int, 2> size;
};

// Replace nullopt with values printed by the C key to reproduce a chosen view.
const std::optional<camera_settings> camera_override = std::nullopt;

struct options {
  std::filesystem::path mesh_path;
  int cuts = 12;
  double explode = 0.0;
  bool all_domains = false;
  bool png = false;
};

auto lower(std::string value) -> std::string {
  std::transform(
      value.begin(), value.end(), value.begin(),
      [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

auto parse_int(const std::string &text) -> int {
  std::size_t consumed = 0;
  const auto value = std::stoi(text, &consumed);
  if (consumed != text.size())
    throw std::invalid_argument("invalid cuts value '" + text + "'");
  return value;
}

auto parse_double(const std::string &text) -> double {
  std::size_t consumed = 0;
  const auto value = std::stod(text, &consumed);
  if (consumed != text.size())
    throw std::invalid_argument("invalid explode value '" + text + "'");
  return value;
}

void print_usage(const char *program, std::ostream &stream) {
  stream
      << "Usage: " << program << " MESH [cuts] [explode] [--all] [--png]\n\n"
      << "  MESH     closed .stl or .obj surface\n"
      << "  cuts     number of axis-aligned cuts per axis [default 12]\n"
      << "  explode  outward cell displacement fraction  [default 0.0]\n"
      << "  --all    render every domain, skip the interior filter\n"
      << "  --png    render offscreen to the temporary fracture.png path\n\n"
      << "Interactive: drag orbit, scroll zoom, 'c' print camera, 'q' quit.\n";
}

auto parse_options(int argc, char **argv) -> std::optional<options> {
  options result;
  std::vector<std::string> positional;
  for (int index = 1; index != argc; ++index) {
    const auto argument = std::string(argv[index]);
    if (argument == "--all")
      result.all_domains = true;
    else if (argument == "--png")
      result.png = true;
    else if (argument == "--help" || argument == "-h") {
      print_usage(argv[0], std::cout);
      return std::nullopt;
    } else if (argument.rfind("--", 0) == 0)
      throw std::invalid_argument("unknown option '" + argument + "'");
    else
      positional.push_back(argument);
  }
  if (positional.empty())
    throw std::invalid_argument("MESH is required");
  if (positional.size() > 3)
    throw std::invalid_argument("too many positional arguments");
  result.mesh_path = positional[0];
  if (positional.size() > 1)
    result.cuts = parse_int(positional[1]);
  if (positional.size() > 2)
    result.explode = parse_double(positional[2]);
  return result;
}

auto stl_as_double(const std::filesystem::path &path) -> storage {
  const auto source = tf::cpp::read_stl(path.string());
  const auto &source_coordinates = source.points_buffer().data_buffer();
  storage result;
  result.faces_buffer().data_buffer() = source.faces_buffer().data_buffer();
  auto &coordinates = result.points_buffer().data_buffer();
  coordinates.allocate(source_coordinates.size());
  std::copy(source_coordinates.begin(), source_coordinates.end(),
            coordinates.begin());
  return result;
}

auto load_closed(const std::filesystem::path &path) -> storage {
  const auto extension = lower(path.extension().string());
  storage source;
  if (extension == ".stl")
    source = stl_as_double(path);
  else if (extension == ".obj")
    source = tf::cpp::read_obj<index_t, double, 3>(path.string());
  else
    throw std::invalid_argument("unsupported mesh extension '" + extension +
                                "' (use .stl or .obj)");

  cache_t source_cache;
  auto value =
      tf::cpp::cleaned_mesh(examples::mesh_over(source, source_cache), 1.0e-6);
  cache_t cache;
  if (!tf::cpp::is_closed(examples::mesh_over(value, cache)))
    throw std::invalid_argument(path.filename().string() +
                                " is not watertight; 'inside the solid' is "
                                "undefined on an open surface");

  const auto [lower_bound, upper_bound] =
      examples::point_bounds(value.points_buffer());
  const point center{(lower_bound[0] + upper_bound[0]) / 2.0,
                     (lower_bound[1] + upper_bound[1]) / 2.0,
                     (lower_bound[2] + upper_bound[2]) / 2.0};
  const auto extent = std::max({upper_bound[0] - lower_bound[0],
                                upper_bound[1] - lower_bound[1],
                                upper_bound[2] - lower_bound[2]});
  if (!(extent > 0.0))
    throw std::invalid_argument("mesh has zero extent");
  // The coordinates move after the last read of them, so nothing the cache
  // holds is stale by the time this is handed back.
  for (auto vertex : value.points())
    for (std::size_t axis = 0; axis != 3; ++axis)
      vertex[axis] = (vertex[axis] - center[axis]) / extent;
  return value;
}

using matrix3 = std::array<double, 9>;

auto rotation_z_to(const point &normal) -> matrix3 {
  if (std::abs(normal[0]) < 1.0e-12 && std::abs(normal[1]) < 1.0e-12 &&
      std::abs(normal[2] - 1.0) < 1.0e-12)
    return {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  if (std::abs(normal[0]) < 1.0e-12 && std::abs(normal[1]) < 1.0e-12 &&
      std::abs(normal[2] + 1.0) < 1.0e-12)
    return {1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0};

  point axis{-normal[1], normal[0], 0.0};
  const auto axis_length = std::sqrt(axis[0] * axis[0] + axis[1] * axis[1]);
  for (auto &coordinate : axis)
    coordinate /= axis_length;
  const auto cosine = normal[2];
  const auto sine = std::sqrt(std::max(0.0, 1.0 - cosine * cosine));
  const matrix3 k{0.0,      -axis[2], axis[1], axis[2], 0.0,
                  -axis[0], -axis[1], axis[0], 0.0};
  matrix3 kk{};
  for (int row = 0; row != 3; ++row)
    for (int column = 0; column != 3; ++column)
      for (int inner = 0; inner != 3; ++inner)
        kk[static_cast<std::size_t>(row * 3 + column)] +=
            k[static_cast<std::size_t>(row * 3 + inner)] *
            k[static_cast<std::size_t>(inner * 3 + column)];
  matrix3 result{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  for (std::size_t index = 0; index != result.size(); ++index)
    result[index] += sine * k[index] + (1.0 - cosine) * kk[index];
  return result;
}

auto make_plane(const point &center, const point &normal, double side = 4.0)
    -> storage {
  auto value = tf::cpp::make_plane_mesh(side, side);
  const auto rotation = rotation_z_to(normal);
  const auto local_center = examples::point_mean(value.points_buffer());
  for (auto vertex : value.points()) {
    const point local{vertex[0] - local_center[0], vertex[1] - local_center[1],
                      vertex[2] - local_center[2]};
    for (std::size_t axis = 0; axis != 3; ++axis)
      vertex[axis] = rotation[axis * 3] * local[0] +
                     rotation[axis * 3 + 1] * local[1] +
                     rotation[axis * 3 + 2] * local[2] + center[axis];
  }
  return value;
}

auto make_planes(const tf::points_buffer<double, 3> &solid_points, int cuts)
    -> std::vector<storage> {
  const auto [lower_bound, upper_bound] = examples::point_bounds(solid_points);
  const point middle{(lower_bound[0] + upper_bound[0]) / 2.0,
                     (lower_bound[1] + upper_bound[1]) / 2.0,
                     (lower_bound[2] + upper_bound[2]) / 2.0};
  std::vector<storage> planes;
  if (cuts > 0)
    planes.reserve(static_cast<std::size_t>(cuts * 3));
  for (int axis = 0; axis != 3; ++axis) {
    point normal{0.0, 0.0, 0.0};
    normal[static_cast<std::size_t>(axis)] = 1.0;
    for (int index = 0; index < cuts; ++index) {
      const auto t =
          static_cast<double>(index + 1) / static_cast<double>(cuts + 1);
      auto center = middle;
      center[static_cast<std::size_t>(axis)] =
          lower_bound[static_cast<std::size_t>(axis)] +
          t * (upper_bound[static_cast<std::size_t>(axis)] -
               lower_bound[static_cast<std::size_t>(axis)]);
      planes.push_back(make_plane(center, normal));
    }
  }
  return planes;
}

auto cell_color(std::size_t index) -> std::array<double, 3> {
  const auto &value = palette[(index * 5) % palette.size()];
  return {value[0] / 255.0, value[1] / 255.0, value[2] / 255.0};
}

auto point_string(const double *value) -> std::string {
  std::ostringstream stream;
  stream << std::fixed << std::setprecision(4) << '(' << value[0] << ", "
         << value[1] << ", " << value[2] << ')';
  return stream.str();
}

class camera_interactor : public vtkInteractorStyleTrackballCamera {
public:
  static auto New() -> camera_interactor *;
  vtkTypeMacro(camera_interactor, vtkInteractorStyleTrackballCamera);

  auto initialize(vtkCamera *camera, vtkRenderWindow *window) -> void {
    _camera = camera;
    _window = window;
  }

  auto OnKeyPress() -> void override {
    const auto key = std::string(this->Interactor->GetKeySym());
    if (key == "c" || key == "C") {
      const auto *size = _window->GetSize();
      std::cout << "\ncamera_settings{\n"
                << "  " << point_string(_camera->GetPosition()) << ",\n"
                << "  " << point_string(_camera->GetFocalPoint()) << ",\n"
                << "  " << point_string(_camera->GetViewUp()) << ",\n"
                << "  " << std::fixed << std::setprecision(3)
                << _camera->GetViewAngle() << ", {" << size[0] << ", "
                << size[1] << "}}\n\n";
      return;
    }
    vtkInteractorStyleTrackballCamera::OnKeyPress();
  }

private:
  vtkCamera *_camera = nullptr;
  vtkRenderWindow *_window = nullptr;
};

vtkStandardNewMacro(camera_interactor);

void set_camera(vtkRenderer *renderer, vtkRenderWindow *window,
                vtkCamera *camera) {
  if (camera_override) {
    camera->SetPosition(camera_override->position.data());
    camera->SetFocalPoint(camera_override->focal_point.data());
    camera->SetViewUp(camera_override->view_up.data());
    camera->SetViewAngle(camera_override->view_angle);
    window->SetSize(camera_override->size[0], camera_override->size[1]);
  } else {
    window->SetSize(1400, 1400);
    camera->SetViewUp(0.0, 0.0, 1.0);
    camera->SetFocalPoint(0.0, 0.0, 0.0);
    camera->SetPosition(0.55, -1.0, 0.32);
    renderer->ResetCamera();
    camera->Zoom(1.35);
  }
  renderer->ResetCameraClippingRange();
}

void add_cell_actors(vtkRenderer *renderer, const storage &cell,
                     std::size_t index) {
  auto poly = examples::to_vtk_polydata(cell);
  auto mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
  mapper->SetInputData(poly);
  auto actor = vtkSmartPointer<vtkActor>::New();
  actor->SetMapper(mapper);
  auto *property = actor->GetProperty();
  const auto fill_color = cell_color(index);
  property->SetColor(fill_color[0], fill_color[1], fill_color[2]);
  property->SetEdgeVisibility(0);
  property->SetInterpolationToPhong();
  property->SetAmbient(0.32);
  property->SetDiffuse(0.68);
  property->SetSpecular(0.18);
  property->SetSpecularPower(28.0);
  renderer->AddActor(actor);

  auto feature_edges = vtkSmartPointer<vtkFeatureEdges>::New();
  feature_edges->SetInputData(poly);
  feature_edges->BoundaryEdgesOn();
  feature_edges->FeatureEdgesOn();
  feature_edges->SetFeatureAngle(25.0);
  feature_edges->ManifoldEdgesOff();
  feature_edges->NonManifoldEdgesOff();
  feature_edges->Update();
  auto edge_mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
  edge_mapper->SetInputConnection(feature_edges->GetOutputPort());
  edge_mapper->ScalarVisibilityOff();
  auto edge_actor = vtkSmartPointer<vtkActor>::New();
  edge_actor->SetMapper(edge_mapper);
  edge_actor->GetProperty()->SetColor(edge_rgb[0], edge_rgb[1], edge_rgb[2]);
  edge_actor->GetProperty()->SetLineWidth(1.1);
  edge_actor->GetProperty()->LightingOff();
  renderer->AddActor(edge_actor);
}

} // namespace

int main(int argc, char **argv) {
  try {
    const auto parsed = parse_options(argc, argv);
    if (!parsed)
      return 0;
    const auto &arguments = *parsed;

    const auto solid = load_closed(arguments.mesh_path);
    const auto name = arguments.mesh_path.filename().string();
    const auto [lower_bound, upper_bound] =
        examples::point_bounds(solid.points_buffer());
    std::cout << name << ": " << solid.size() << " tris, bbox ["
              << lower_bound[0] << ' ' << lower_bound[1] << ' '
              << lower_bound[2] << "] .. [" << upper_bound[0] << ' '
              << upper_bound[1] << ' ' << upper_bound[2] << "]\n";

    const auto planes = make_planes(solid.points_buffer(), arguments.cuts);

    // The graph reads its operands for as long as it lives: the storage and
    // the caches below outlive it, and every cut is one more reading.
    std::vector<cache_t> caches(planes.size() + 1);
    std::vector<mesh_t> operands;
    operands.reserve(caches.size());
    operands.push_back(examples::mesh_over(solid, caches.front()));
    for (std::size_t plane = 0; plane != planes.size(); ++plane)
      operands.push_back(examples::mesh_over(planes[plane], caches[plane + 1]));

    const auto graph = tf::cpp::make_csg_graph(std::move(operands));
    constexpr auto domain_config = tf::domain_config::exclude_outer_shell |
                                   tf::domain_config::ignore_open_fragments;
    auto domains =
        arguments.all_domains
            ? tf::cpp::make_csg_domains(graph, domain_config)
            : tf::cpp::make_csg_domains(graph, tf::csg::op(0), domain_config);
    std::cout << "cells: " << domains.meshes.size()
              << (arguments.all_domains ? "  (ALL)" : "  (interior of solid)")
              << '\n';

    const auto scene_center = examples::point_mean(solid.points_buffer());
    std::vector<storage> cells;
    cells.reserve(domains.meshes.size());
    for (auto &cell : domains.meshes) {
      if (cell.points_buffer().size() == 0 || cell.size() < 1)
        continue;
      const auto center = examples::point_mean(cell.points_buffer());
      for (auto vertex : cell.points())
        for (std::size_t axis = 0; axis != 3; ++axis)
          vertex[axis] +=
              arguments.explode * (center[axis] - scene_center[axis]);
      cells.push_back(std::move(cell));
    }
    std::cout << "rendered domains: " << cells.size() << '\n';

    auto renderer = vtkSmartPointer<vtkRenderer>::New();
    renderer->SetBackground(1.0, 1.0, 1.0);
    if (!arguments.png)
      renderer->UseFXAAOn();
    for (std::size_t index = 0; index != cells.size(); ++index)
      add_cell_actors(renderer, cells[index], index);

    auto light_kit = vtkSmartPointer<vtkLightKit>::New();
    light_kit->SetKeyLightIntensity(0.9);
    light_kit->SetKeyToFillRatio(2.2);
    light_kit->SetKeyToHeadRatio(3.5);
    light_kit->SetKeyToBackRatio(3.0);
    light_kit->AddLightsToRenderer(renderer);

    auto window = vtkSmartPointer<vtkRenderWindow>::New();
    window->SetMultiSamples(8);
    window->AddRenderer(renderer);
    const auto title = std::string("csg_fracture: ") + name + " / " +
                       std::to_string(cells.size()) + " cells / " +
                       std::to_string(arguments.cuts) + " cuts";
    window->SetWindowName(title.c_str());
    auto *camera = renderer->GetActiveCamera();
    set_camera(renderer, window, camera);

    if (arguments.png) {
      window->SetAlphaBitPlanes(1);
      window->SetOffScreenRendering(1);
      window->Render();
      auto image = vtkSmartPointer<vtkWindowToImageFilter>::New();
      image->SetInput(window);
      image->SetInputBufferTypeToRGBA();
      image->ReadFrontBufferOff();
      image->Update();
      auto writer = vtkSmartPointer<vtkPNGWriter>::New();
#if defined(_WIN32)
      const auto png_path =
          std::filesystem::temp_directory_path() / "fracture.png";
#else
      const auto png_path = std::filesystem::path("/tmp/fracture.png");
#endif
      const auto png_path_string = png_path.string();
      writer->SetFileName(png_path_string.c_str());
      writer->SetInputConnection(image->GetOutputPort());
      writer->Write();
      if (writer->GetErrorCode() != vtkErrorCode::NoError)
        throw std::runtime_error(
            "failed to write " + png_path_string + ": " +
            vtkErrorCode::GetStringFromErrorCode(writer->GetErrorCode()));
      std::cout << "wrote " << png_path_string << " (RGBA)\n";
    } else {
      auto interactor = vtkSmartPointer<vtkRenderWindowInteractor>::New();
      interactor->SetRenderWindow(window);
      auto style = vtkSmartPointer<camera_interactor>::New();
      style->initialize(camera, window);
      interactor->SetInteractorStyle(style);
      window->Render();
      std::cout << "interactive: drag orbit, scroll zoom, 'c' print camera, "
                   "'q' quit\n";
      interactor->Start();
    }
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "error: " << error.what() << '\n';
    print_usage(argv[0], std::cerr);
    return 1;
  }
}
