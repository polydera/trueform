#include "util/vtk_examples.hpp"

#include <trueform/cpp/core/cache.hpp>
#include <trueform/cpp/core/index_type.hpp>
#include <trueform/cpp/core/mesh.hpp>
#include <trueform/cpp/geometry/mean_edge_length.hpp>
#include <trueform/cpp/remesh/decimated.hpp>
#include <trueform/cpp/remesh/isotropic_remeshed.hpp>
#include <trueform/cpp/remesh/simplified.hpp>

#include <vtkActor.h>
#include <vtkCallbackCommand.h>
#include <vtkCommand.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkPolyData.h>
#include <vtkProperty.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkSmartPointer.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace examples = trueform::vtk_examples;

constexpr std::array<double, 3> edge_color{0.15, 0.15, 0.18};
constexpr int base_height = 700;

struct arguments {
  std::filesystem::path mesh = std::filesystem::path(TRUEFORM_DATA_DIR) /
                               "benchmarks" / "data" / "dragon-500k.stl";
  float target = 0.05F;
};

auto usage(const char *program) -> void {
  std::cout << "Usage: " << program << " [mesh.stl] [-t TARGET]\n"
            << "Remesh pipeline: decimate then isotropic remesh\n\n"
            << "Options:\n"
            << "  -t, --target TARGET  Decimation target proportion (0.0-1.0). "
               "Default: 0.05\n";
}

auto parse_float(const std::string &text) -> float {
  std::size_t consumed = 0;
  const auto value = std::stof(text, &consumed);
  if (consumed != text.size())
    throw std::invalid_argument("invalid target proportion: " + text);
  return value;
}

auto parse_arguments(int argc, char **argv) -> arguments {
  arguments result;
  bool has_mesh = false;
  for (int i = 1; i != argc; ++i) {
    const std::string value(argv[i]);
    if (value == "-h" || value == "--help") {
      usage(argv[0]);
      std::exit(0);
    }
    if (value == "-t" || value == "--target") {
      if (++i == argc)
        throw std::invalid_argument(value + " requires a value");
      result.target = parse_float(argv[i]);
      continue;
    }
    constexpr auto target_prefix = "--target=";
    if (value.rfind(target_prefix, 0) == 0) {
      result.target = parse_float(
          value.substr(std::char_traits<char>::length(target_prefix)));
      continue;
    }
    if (has_mesh)
      throw std::invalid_argument("only one mesh path may be supplied");
    result.mesh = value;
    has_mesh = true;
  }
  return result;
}

auto grouped(std::size_t value) -> std::string {
  auto text = std::to_string(value);
  for (auto position = static_cast<std::ptrdiff_t>(text.size()) - 3;
       position > 0; position -= 3)
    text.insert(static_cast<std::size_t>(position), 1, ',');
  return text;
}

auto format_duration(double seconds) -> std::string {
  std::ostringstream stream;
  if (seconds < 1.0)
    stream << std::fixed << std::setprecision(1) << seconds * 1000.0 << " ms";
  else
    stream << std::fixed << std::setprecision(2) << seconds << " s";
  return stream.str();
}

using index_t = tf::cpp::default_index_t;
using mesh_t = tf::cpp::mesh<index_t, float>;
using triangles_t = tf::polygons_buffer<index_t, float, 3, 3>;

auto make_mesh_actor(const triangles_t &mesh, bool edges = false)
    -> vtkSmartPointer<vtkActor> {
  const auto polydata = examples::to_vtk_polydata(mesh);
  const auto actor = examples::make_actor(polydata, {0.85, 0.85, 0.88});
  auto *property = actor->GetProperty();
  property->SetAmbient(0.2);
  property->SetDiffuse(0.8);
  if (edges) {
    property->EdgeVisibilityOn();
    property->SetEdgeColor(edge_color.data());
    property->SetLineWidth(1.0);
  }
  return actor;
}

struct edge_width_data {
  std::array<vtkActor *, 2> actors{};
};

auto update_edge_width(vtkRenderWindow *window, edge_width_data &data) -> void {
  const auto height = window->GetSize()[1];
  const auto width = std::max(0.5, 0.7 * static_cast<double>(height) /
                                       static_cast<double>(base_height));
  for (auto *actor : data.actors)
    actor->GetProperty()->SetLineWidth(width);
}

} // namespace

int main(int argc, char **argv) {
  try {
    const auto options = parse_arguments(argc, argv);

    std::cout << "Loading: " << options.mesh.string() << '\n';
    const auto original = tf::cpp::read_stl(options.mesh.string());
    tf::cpp::cache<index_t, float> original_cache;
    const mesh_t original_mesh{original.faces(), original.points(),
                               original_cache};
    const auto original_faces = original.size();
    std::cout << "  " << grouped(original_faces) << " faces, "
              << grouped(original.points_buffer().size()) << " points\n";

    std::cout << "Decimating to " << std::fixed << std::setprecision(1)
              << options.target * 100.0F << "% ...\n";
    auto start = std::chrono::steady_clock::now();
    const auto decimated =
        tf::cpp::decimated(original_mesh, options.target).mesh;
    const auto decimation_time = examples::elapsed_seconds(start);
    const auto decimated_faces = decimated.size();
    std::cout << "  " << grouped(decimated_faces) << " faces, "
              << grouped(decimated.points_buffer().size()) << " points  ("
              << std::fixed << std::setprecision(1) << decimation_time * 1000.0
              << " ms)\n";

    // Every stage is new storage, so every stage brings its own cache.
    tf::cpp::cache<index_t, float> decimated_cache;
    const mesh_t decimated_mesh{decimated.faces(), decimated.points(),
                                decimated_cache};
    const auto target_length = tf::cpp::mean_edge_length(decimated_mesh);
    std::cout << "Isotropic remesh (target length " << std::fixed
              << std::setprecision(4) << target_length << ") ...\n";
    auto remesh_config = tf::isotropic_remesh_config<float>(target_length);
    remesh_config.use_quadric = true;
    start = std::chrono::steady_clock::now();
    const auto remeshed =
        tf::cpp::isotropic_remeshed(decimated_mesh, remesh_config).mesh;
    const auto remesh_time = examples::elapsed_seconds(start);
    const auto remeshed_faces = remeshed.size();
    std::cout << "  " << grouped(remeshed_faces) << " faces, "
              << grouped(remeshed.points_buffer().size()) << " points  ("
              << std::fixed << std::setprecision(1) << remesh_time * 1000.0
              << " ms)\n";

    auto original_renderer = vtkSmartPointer<vtkRenderer>::New();
    auto decimated_renderer = vtkSmartPointer<vtkRenderer>::New();
    auto remeshed_renderer = vtkSmartPointer<vtkRenderer>::New();
    auto text_renderer = vtkSmartPointer<vtkRenderer>::New();

    constexpr double text_height = 0.10;
    constexpr double half = 0.5;
    constexpr double middle_y = text_height + (1.0 - text_height) / 2.0;
    original_renderer->SetViewport(0.0, text_height, half, 1.0);
    decimated_renderer->SetViewport(half, middle_y, 1.0, 1.0);
    remeshed_renderer->SetViewport(half, text_height, 1.0, middle_y);
    text_renderer->SetViewport(0.0, 0.0, 1.0, text_height);
    text_renderer->InteractiveOff();

    constexpr std::array<double, 3> background{27.0 / 255.0, 43.0 / 255.0,
                                               52.0 / 255.0};
    for (auto *renderer :
         {original_renderer.GetPointer(), decimated_renderer.GetPointer(),
          remeshed_renderer.GetPointer()})
      renderer->SetBackground(background.data());
    text_renderer->SetBackground(0.090, 0.143, 0.173);

    const auto original_actor = make_mesh_actor(original);
    const auto decimated_actor = make_mesh_actor(decimated, true);
    const auto remeshed_actor = make_mesh_actor(remeshed, true);
    original_renderer->AddActor(original_actor);
    decimated_renderer->AddActor(decimated_actor);
    remeshed_renderer->AddActor(remeshed_actor);

    original_renderer->AddViewProp(examples::make_text_actor(
        "Original: " + grouped(original_faces) + " faces", 18, {0.03, 0.03}));
    {
      std::ostringstream label;
      label << "Decimated: " << grouped(decimated_faces) << " faces ("
            << std::fixed << std::setprecision(0) << options.target * 100.0F
            << "%)";
      decimated_renderer->AddViewProp(
          examples::make_text_actor(label.str(), 14, {0.03, 0.05}));
    }
    remeshed_renderer->AddViewProp(examples::make_text_actor(
        "Remeshed: " + grouped(remeshed_faces) + " faces", 14, {0.03, 0.05}));

    const auto summary = "Decimate " + format_duration(decimation_time) +
                         "  |  Remesh " + format_duration(remesh_time) +
                         "  |  " + grouped(original_faces) + " → " +
                         grouped(decimated_faces) + " → " +
                         grouped(remeshed_faces) + " faces";
    text_renderer->AddViewProp(
        examples::make_text_actor(summary, 34, {0.02, 0.5}));

    auto window = vtkSmartPointer<vtkRenderWindow>::New();
    window->AddRenderer(original_renderer);
    window->AddRenderer(decimated_renderer);
    window->AddRenderer(remeshed_renderer);
    window->AddRenderer(text_renderer);
    window->SetSize(1600, 800);

    auto interactor = vtkSmartPointer<vtkRenderWindowInteractor>::New();
    interactor->SetRenderWindow(window);
    auto style = vtkSmartPointer<vtkInteractorStyleTrackballCamera>::New();
    interactor->SetInteractorStyle(style);

    edge_width_data edge_data{{decimated_actor, remeshed_actor}};
    update_edge_width(window, edge_data);
    auto resize_callback = vtkSmartPointer<vtkCallbackCommand>::New();
    resize_callback->SetClientData(&edge_data);
    resize_callback->SetCallback(
        [](vtkObject *caller, unsigned long, void *client_data, void *) {
          auto *render_window = vtkRenderWindow::SafeDownCast(caller);
          auto &data = *static_cast<edge_width_data *>(client_data);
          update_edge_width(render_window, data);
        });
    const auto resize_observer =
        window->AddObserver(vtkCommand::ModifiedEvent, resize_callback);

    decimated_renderer->SetActiveCamera(original_renderer->GetActiveCamera());
    remeshed_renderer->SetActiveCamera(original_renderer->GetActiveCamera());
    original_renderer->ResetCamera();

    window->Render();
    interactor->Start();
    window->RemoveObserver(resize_observer);
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "remeshing: " << error.what() << '\n';
    return 1;
  }
}
