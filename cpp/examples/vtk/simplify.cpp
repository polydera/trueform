#include "util/vtk_examples.hpp"

#include <trueform/cpp/core/cache.hpp>
#include <trueform/cpp/core/index_type.hpp>
#include <trueform/cpp/core/mesh.hpp>
#include <trueform/cpp/io/read_obj.hpp>
#include <trueform/cpp/remesh/decimated.hpp>
#include <trueform/cpp/remesh/isotropic_remeshed.hpp>
#include <trueform/cpp/remesh/simplified.hpp>

#include <vtkActor.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkProperty.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkSmartPointer.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
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

constexpr std::array<float, 5> error_rel_values{0.0005F, 0.001F, 0.002F, 0.004F,
                                                0.008F};
constexpr float minimum_quality = 0.3F;
constexpr float feature_angle_degrees = 60.0F;
constexpr int iterations = 3;
constexpr int optimize_iterations = 3;
constexpr std::array<double, 3> mesh_color{0.78, 0.80, 0.85};
constexpr std::array<double, 3> edge_color{0.10, 0.12, 0.14};
constexpr std::array<double, 3> background{27.0 / 255.0, 43.0 / 255.0,
                                           52.0 / 255.0};

struct panel {
  std::string label;
  vtkSmartPointer<vtkPolyData> polydata;
};

auto grouped(std::size_t value) -> std::string {
  auto text = std::to_string(value);
  for (auto position = static_cast<std::ptrdiff_t>(text.size()) - 3;
       position > 0; position -= 3)
    text.insert(static_cast<std::size_t>(position), 1, ',');
  return text;
}

auto scalar_text(float value) -> std::string {
  std::ostringstream stream;
  stream << std::defaultfloat << value;
  return stream.str();
}

auto expand_user(std::filesystem::path path) -> std::filesystem::path {
  const auto text = path.string();
  if (text == "~" || text.rfind("~/", 0) == 0) {
    if (const auto *home = std::getenv("HOME"))
      return text == "~" ? std::filesystem::path(home)
                         : std::filesystem::path(home) / text.substr(2);
  }
  return path;
}

using triangles = tf::polygons_buffer<tf::cpp::default_index_t, float, 3, 3>;

auto load_mesh(const std::filesystem::path &path) -> triangles {
  auto extension = path.extension().string();
  std::transform(extension.begin(), extension.end(), extension.begin(),
                 [](unsigned char value) {
                   return static_cast<char>(std::tolower(value));
                 });
  if (extension == ".stl")
    return tf::cpp::read_stl(path.string());
  if (extension == ".obj")
    return tf::cpp::read_obj<tf::cpp::default_index_t, float, 3>(path.string());
  throw std::invalid_argument("unsupported mesh extension '" + extension +
                              "' (use .stl or .obj)");
}

auto to_double_vtk_polydata(const triangles &mesh)
    -> vtkSmartPointer<vtkPolyData> {
  auto result = examples::to_vtk_polydata(mesh);
  auto points = vtkSmartPointer<vtkPoints>::New();
  points->SetDataTypeToDouble();
  const auto source = mesh.points();
  points->SetNumberOfPoints(static_cast<vtkIdType>(source.size()));
  for (std::size_t point = 0; point != source.size(); ++point)
    points->SetPoint(static_cast<vtkIdType>(point), source[point][0],
                     source[point][1], source[point][2]);
  result->SetPoints(points);
  return result;
}

auto make_actor(vtkPolyData *polydata) -> vtkSmartPointer<vtkActor> {
  const auto actor = examples::make_actor(polydata, mesh_color);
  auto *property = actor->GetProperty();
  property->SetAmbient(0.2);
  property->SetDiffuse(0.85);
  property->EdgeVisibilityOn();
  property->SetEdgeColor(edge_color.data());
  property->SetLineWidth(1.0);
  return actor;
}

auto show(const std::vector<panel> &panels) -> void {
  constexpr int columns = 3;
  constexpr int rows = 2;
  auto window = vtkSmartPointer<vtkRenderWindow>::New();
  window->SetWindowName("simplify: fixed quality, increasing error_rel");
  std::vector<vtkSmartPointer<vtkRenderer>> renderers;
  renderers.reserve(panels.size());

  for (std::size_t i = 0; i != panels.size(); ++i) {
    auto renderer = vtkSmartPointer<vtkRenderer>::New();
    renderer->AddActor(make_actor(panels[i].polydata));
    renderer->AddViewProp(
        examples::make_text_actor(panels[i].label, 14, {0.04, 0.05}));
    renderer->SetBackground(background.data());
    const auto column = static_cast<int>(i) % columns;
    const auto row = static_cast<int>(i) / columns;
    const auto y1 = 1.0 - static_cast<double>(row) / rows;
    renderer->SetViewport(static_cast<double>(column) / columns,
                          y1 - 1.0 / rows,
                          static_cast<double>(column + 1) / columns, y1);
    window->AddRenderer(renderer);
    renderers.push_back(std::move(renderer));
  }
  window->SetSize(1500, 1000);

  auto interactor = vtkSmartPointer<vtkRenderWindowInteractor>::New();
  interactor->SetRenderWindow(window);
  auto style = vtkSmartPointer<vtkInteractorStyleTrackballCamera>::New();
  interactor->SetInteractorStyle(style);

  renderers.front()->ResetCamera();
  for (std::size_t i = 1; i != renderers.size(); ++i)
    renderers[i]->SetActiveCamera(renderers.front()->GetActiveCamera());
  window->Render();
  interactor->Start();
}

} // namespace

int main(int argc, char **argv) {
  try {
    const auto default_mesh = std::filesystem::path(TRUEFORM_DATA_DIR) /
                              "benchmarks" / "data" / "dragon-500k.stl";
    if (argc > 2 || (argc == 2 && (std::string(argv[1]) == "-h" ||
                                   std::string(argv[1]) == "--help"))) {
      std::cout << "Usage: " << argv[0] << " [mesh.stl|mesh.obj]\n"
                << "Error-budget simplification sweep (tf::cpp::simplified)\n";
      return argc > 2 ? 2 : 0;
    }
    auto path =
        expand_user(argc == 2 ? std::filesystem::path(argv[1]) : default_mesh);
    if (!std::filesystem::exists(path))
      throw std::invalid_argument("mesh not found: " + path.string());

    const auto source = load_mesh(path);
    tf::cpp::cache<tf::cpp::default_index_t, float> cache;
    const tf::cpp::mesh<tf::cpp::default_index_t, float> mesh{
        source.faces(), source.points(), cache};
    const auto source_faces = source.size();
    std::cout << "loaded " << path.filename().string() << ": "
              << grouped(source_faces) << " faces, "
              << grouped(source.points_buffer().size()) << " points\n";

    std::vector<panel> panels;
    panels.reserve(6);
    panels.push_back({"input\n" + grouped(source_faces) + " faces",
                      to_double_vtk_polydata(source)});

    for (const auto error_rel : error_rel_values) {
      tf::simplify_config<float> config;
      config.error_rel = error_rel;
      config.min_quality = minimum_quality;
      config.feature_angle = tf::rad<float>(
          static_cast<float>(static_cast<double>(feature_angle_degrees) *
                             std::acos(-1.0) / 180.0));
      config.iterations = iterations;
      config.optimize_iterations = optimize_iterations;

      const auto start = std::chrono::steady_clock::now();
      auto simplified = tf::cpp::simplified(mesh, config).mesh;
      const auto seconds = examples::elapsed_seconds(start);
      const auto face_count = simplified.size();
      const auto percentage = 100.0 * static_cast<double>(face_count) /
                              static_cast<double>(source_faces);

      std::cout << "  error_rel=" << std::left << std::setw(7)
                << scalar_text(error_rel) << " -> " << std::right
                << std::setw(8) << grouped(face_count) << " faces ("
                << std::fixed << std::setw(5) << std::setprecision(1)
                << percentage << "%)  " << std::setw(7) << seconds * 1000.0
                << " ms\n";

      std::ostringstream label;
      label << "error_rel = " << scalar_text(error_rel) << '\n'
            << grouped(face_count) << " faces (" << std::fixed
            << std::setprecision(1) << percentage << "%)";
      panels.push_back({label.str(), to_double_vtk_polydata(simplified)});
    }

    show(panels);
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "simplify: " << error.what() << '\n';
    return 1;
  }
}
