#include "util/mesh_interactor.hpp"
#include "util/vtk_examples.hpp"

#include <trueform/cpp/geometry/triangulate.hpp>
#include <trueform/cpp/io/read_stl.hpp>
#include <trueform/cpp/iso/isocontours.hpp>
#include <trueform/cpp/spatial/distance.hpp>
#include <trueform/cpp/spatial/primitive.hpp>

#include <vtkActor.h>
#include <vtkObjectFactory.h>
#include <vtkPolyData.h>
#include <vtkProperty.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkSmartPointer.h>
#include <vtkTextActor.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace ex = trueform::vtk_examples;

namespace {

auto print_usage(const char *program) -> void {
  std::cout << "Usage: " << program << " MESH\n"
            << "Interactive filled cross-section extraction.\n\n"
            << "Controls:\n"
            << "  N              Randomize cutting plane\n"
            << "  Ctrl + Scroll  Move cutting plane\n"
            << "  Mouse drag     Rotate camera\n";
}

auto parse_arguments(int argc, char **argv) -> std::filesystem::path {
  if (argc == 2 &&
      (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help")) {
    print_usage(argv[0]);
    std::exit(0);
  }
  if (argc != 2)
    throw std::invalid_argument("exactly one mesh file is required");
  return argv[1];
}

class cross_section_interactor final : public ex::mesh_interactor {
public:
  static auto New() -> cross_section_interactor *;
  vtkTypeMacro(cross_section_interactor, ex::mesh_interactor);

  auto initialize(ex::mesh_actor_data &surface, vtkPolyData *fill,
                  vtkPolyData *curves, vtkTextActor *timing_text) -> void {
    surface_ = &surface;
    fill_ = fill;
    curves_ = curves;
    timing_text_ = timing_text;
    reset_plane();
  }

  auto compute_cross_section() -> void {
    const auto start = std::chrono::steady_clock::now();
    const auto contours =
        tf::cpp::isocontours(surface_->mesh(), scalars_, cut_value_);

    if (contours.size() != 0 && contours.points_buffer().size() != 0) {
      const auto triangulated = tf::cpp::triangulate<std::int32_t, float, 3>(
          ex::as_offset_blocks(contours.paths_buffer()),
          ex::points_array(contours.points_buffer()));
      update_timing(start);
      const auto fill = ex::to_vtk_polydata(triangulated);
      ex::replace_polydata(fill_, fill);
      const auto lines = ex::to_vtk_polydata(contours);
      ex::replace_polydata(curves_, lines);
    } else {
      update_timing(start);
      ex::replace_polydata(fill_, nullptr);
      ex::replace_polydata(curves_, nullptr);
    }
    if (this->Interactor)
      this->Interactor->Render();
  }

  auto OnKeyPress() -> void override {
    auto key = std::string(this->Interactor->GetKeySym());
    if (key == "n" || key == "N") {
      randomize_plane();
      compute_cross_section();
      return;
    }
    vtkInteractorStyleTrackballCamera::OnKeyPress();
  }

  auto OnMouseWheelForward() -> void override {
    if (!this->Interactor->GetControlKey()) {
      vtkInteractorStyleTrackballCamera::OnMouseWheelForward();
      return;
    }
    const auto range = max_distance_ - min_distance_;
    const auto margin = range * 0.01F;
    cut_value_ = std::min(max_distance_ - margin, cut_value_ + range * 0.005F);
    compute_cross_section();
  }

  auto OnMouseWheelBackward() -> void override {
    if (!this->Interactor->GetControlKey()) {
      vtkInteractorStyleTrackballCamera::OnMouseWheelBackward();
      return;
    }
    const auto range = max_distance_ - min_distance_;
    const auto margin = range * 0.01F;
    cut_value_ = std::max(min_distance_ + margin, cut_value_ - range * 0.005F);
    compute_cross_section();
  }

private:
  auto update_timing(std::chrono::steady_clock::time_point start) -> void {
    times_.add(ex::elapsed_seconds(start));
    const auto timing =
        "Cross-section time: " + ex::format_milliseconds(times_.average());
    timing_text_->SetInput(timing.c_str());
  }

  auto set_plane(const ex::point3<float> &point, ex::point3<float> normal)
      -> void {
    const auto length = std::sqrt(
        normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);
    if (!(length > 0.0F))
      throw std::runtime_error("cutting-plane normal has zero length");
    for (auto &coordinate : normal)
      coordinate /= length;

    const auto plane_d =
        -(normal[0] * point[0] + normal[1] * point[1] + normal[2] * point[2]);
    const auto plane = tf::cpp::primitive<float>(
        tf::cpp::primitive_kind::plane,
        ex::make_array<float>({normal[0], normal[1], normal[2], plane_d}, {4}));
    const auto point_batch = tf::cpp::primitive<float>(
        tf::cpp::primitive_kind::point,
        ex::points_array(surface_->polygons.points_buffer()));
    scalars_ = tf::cpp::distance(point_batch, plane).batch();
    const auto limits = std::minmax_element(scalars_.begin(), scalars_.end());
    min_distance_ = *limits.first;
    max_distance_ = *limits.second;
    cut_value_ = (min_distance_ + max_distance_) * 0.5F;
  }

  auto reset_plane() -> void {
    set_plane(ex::point_mean(surface_->polygons.points_buffer()),
              {1.0F, 2.0F, 1.0F});
  }

  auto randomize_plane() -> void {
    static thread_local std::mt19937 generator(std::random_device{}());
    std::normal_distribution<float> normal_distribution;
    ex::point3<float> normal{normal_distribution(generator),
                             normal_distribution(generator),
                             normal_distribution(generator)};
    const auto points = surface_->polygons.points();
    std::uniform_int_distribution<std::size_t> point_distribution(
        0, points.size() - 1);
    const auto point = points[point_distribution(generator)];
    set_plane({point[0], point[1], point[2]}, normal);
  }

  ex::mesh_actor_data *surface_ = nullptr;
  vtkPolyData *fill_ = nullptr;
  vtkPolyData *curves_ = nullptr;
  vtkTextActor *timing_text_ = nullptr;
  tf::cpp::nd_array<float> scalars_;
  float cut_value_ = 0.0F;
  float min_distance_ = 0.0F;
  float max_distance_ = 1.0F;
  ex::rolling_average times_{100};
};

vtkStandardNewMacro(cross_section_interactor);

} // namespace

int main(int argc, char **argv) {
  try {
    const auto mesh_file = parse_arguments(argc, argv);
    auto polygons = tf::cpp::read_stl(mesh_file.string());
    ex::materialize_center_and_scale(polygons, 10.0F);
    auto surface = ex::make_mesh_actor_data(std::move(polygons),
                                            ex::identity_matrix<float>());

    auto renderer = vtkSmartPointer<vtkRenderer>::New();
    auto text_renderer = ex::make_text_strip(renderer);
    surface.actor->GetProperty()->SetColor(0.5, 0.5, 0.55);
    surface.actor->GetProperty()->SetOpacity(0.25);
    renderer->AddActor(surface.actor);

    auto fill = vtkSmartPointer<vtkPolyData>::New();
    auto fill_actor = ex::make_actor(fill, {0.0, 0.659, 0.604});
    renderer->AddActor(fill_actor);

    auto curves = vtkSmartPointer<vtkPolyData>::New();
    auto curve_actor = ex::make_actor(curves, {0.0, 0.835, 0.745});
    curve_actor->GetProperty()->SetRenderLinesAsTubes(true);
    curve_actor->GetProperty()->SetLineWidth(8.0);
    renderer->AddActor(curve_actor);

    auto timing_text =
        ex::make_text_actor("Cross-section time: 0 ms", 38, {0.03, 0.5});
    text_renderer->AddViewProp(timing_text);

    auto window = vtkSmartPointer<vtkRenderWindow>::New();
    window->AddRenderer(renderer);
    window->AddRenderer(text_renderer);
    window->SetSize(800, 600);
    auto interactor = vtkSmartPointer<vtkRenderWindowInteractor>::New();
    interactor->SetRenderWindow(window);
    auto style = vtkSmartPointer<cross_section_interactor>::New();
    style->initialize(surface, fill, curves, timing_text);
    interactor->SetInteractorStyle(style);

    style->compute_cross_section();
    window->Render();
    interactor->Start();
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "cross_section: " << error.what() << '\n';
    return 1;
  }
}
