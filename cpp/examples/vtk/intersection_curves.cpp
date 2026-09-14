#include "util/mesh_interactor.hpp"
#include "util/vtk_examples.hpp"

#include <trueform/cpp/core/build_face_membership.hpp>
#include <trueform/cpp/core/build_manifold_edge_link.hpp>
#include <trueform/cpp/core/build_tree.hpp>
#include <trueform/cpp/intersect/intersection_curves.hpp>
#include <trueform/cpp/intersect/self_intersection_curves.hpp>

#include <vtkActor.h>
#include <vtkObjectFactory.h>
#include <vtkPolyData.h>
#include <vtkProperty.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkSmartPointer.h>
#include <vtkTextActor.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace ex = trueform::vtk_examples;

namespace {

struct arguments {
  std::filesystem::path first;
  std::filesystem::path second;
};

auto print_usage(const char *program) -> void {
  std::cout << "Usage: " << program << " MESH1 [MESH2]\n"
            << "Interactive exact intersection-curve extraction.\n\n"
            << "Controls:\n"
            << "  N           Randomize mesh orientations\n"
            << "  Mouse drag  Move meshes / rotate camera\n";
}

auto parse_arguments(int argc, char **argv) -> arguments {
  if (argc == 2 &&
      (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help")) {
    print_usage(argv[0]);
    std::exit(0);
  }
  if (argc != 2 && argc != 3)
    throw std::invalid_argument("one or two mesh files are required");
  return {argv[1], argc == 3 ? std::filesystem::path(argv[2])
                             : std::filesystem::path(argv[1])};
}

class intersection_curves_interactor final : public ex::mesh_interactor {
public:
  static auto New() -> intersection_curves_interactor *;
  vtkTypeMacro(intersection_curves_interactor, ex::mesh_interactor);

  auto initialize(ex::mesh_actor_data &first, ex::mesh_actor_data &second,
                  vtkPolyData *curves, vtkTextActor *timing_text) -> void {
    first_ = &first;
    second_ = &second;
    curves_ = curves;
    timing_text_ = timing_text;
    add_mesh(first);
    add_mesh(second);
    set_dragged_callback(
        [this](ex::mesh_actor_data &, const ex::point3<float> &) {
          compute_curves();
        });
    set_released_callback(
        [this](ex::mesh_actor_data *) { this->Interactor->Render(); });
  }

  auto compute_curves() -> void {
    const auto start = std::chrono::steady_clock::now();
    const auto result =
        tf::cpp::intersection_curves(first_->mesh(), second_->mesh());
    times_.add(ex::elapsed_seconds(start));
    const auto timing =
        "Intersection curve time: " + ex::format_milliseconds(times_.average());
    timing_text_->SetInput(timing.c_str());
    if (result.size() != 0 && result.points_buffer().size() != 0) {
      const auto polydata = ex::to_vtk_polydata(result);
      ex::replace_polydata(curves_, polydata);
    } else {
      ex::replace_polydata(curves_, nullptr);
    }
  }

  auto OnKeyPress() -> void override {
    const auto key = std::string(this->Interactor->GetKeySym());
    if (key == "n" || key == "N") {
      randomize_orientations();
      compute_curves();
      this->Interactor->Render();
      return;
    }
    vtkInteractorStyleTrackballCamera::OnKeyPress();
  }

private:
  auto randomize_orientations() -> void {
    for (auto *data : {first_, second_}) {
      const auto current = ex::from_vtk_matrix<float>(data->matrix);
      const auto center = ex::transform_point(
          current, ex::point_mean(data->polygons.points_buffer()));
      const auto next = ex::rotate_around_world_point(
          current, center, ex::random_rotation_matrix<float>());
      ex::synchronize(*data, next);
    }
  }

  ex::mesh_actor_data *first_ = nullptr;
  ex::mesh_actor_data *second_ = nullptr;
  vtkPolyData *curves_ = nullptr;
  vtkTextActor *timing_text_ = nullptr;
  ex::rolling_average times_{100};
};

vtkStandardNewMacro(intersection_curves_interactor);

} // namespace

int main(int argc, char **argv) {
  try {
    const auto args = parse_arguments(argc, argv);
    auto first = ex::load_stl_actor(args.first, {0.0F, 0.0F, 0.0F}, true);
    auto second = ex::load_stl_actor(args.second, {15.0F, 0.0F, 0.0F}, true);

    // Prebuild what every frame will read, so the first one does not pay for
    // it. Each verb builds what its structure stands on.
    tf::cpp::build_tree(first.mesh());
    tf::cpp::build_tree(second.mesh());
    tf::cpp::build_face_membership(first.mesh());
    tf::cpp::build_face_membership(second.mesh());
    tf::cpp::build_manifold_edge_link(first.mesh());
    tf::cpp::build_manifold_edge_link(second.mesh());

    auto renderer = vtkSmartPointer<vtkRenderer>::New();
    auto text_renderer = ex::make_text_strip(renderer);
    renderer->AddActor(first.actor);
    renderer->AddActor(second.actor);

    auto curves = vtkSmartPointer<vtkPolyData>::New();
    auto curve_actor = ex::make_actor(curves, {1.0, 0.1, 0.1});
    curve_actor->GetProperty()->SetRenderLinesAsTubes(true);
    curve_actor->GetProperty()->SetLineWidth(8.0);
    renderer->AddActor(curve_actor);

    auto timing_text =
        ex::make_text_actor("Intersection curve time: 0 ms", 38, {0.03, 0.5});
    text_renderer->AddViewProp(timing_text);

    auto window = vtkSmartPointer<vtkRenderWindow>::New();
    window->AddRenderer(renderer);
    window->AddRenderer(text_renderer);
    window->SetSize(800, 600);
    auto interactor = vtkSmartPointer<vtkRenderWindowInteractor>::New();
    interactor->SetRenderWindow(window);
    auto style = vtkSmartPointer<intersection_curves_interactor>::New();
    style->initialize(first, second, curves, timing_text);
    style->set_interaction_renderer(renderer);
    interactor->SetInteractorStyle(style);

    renderer->ResetCamera();
    style->compute_curves();
    window->Render();
    interactor->Start();
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "intersection_curves: " << error.what() << '\n';
    return 1;
  }
}
