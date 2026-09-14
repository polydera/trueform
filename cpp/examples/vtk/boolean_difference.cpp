/*
 * Interactive boolean difference using the public C++ facade and upstream VTK.
 *
 * Usage: boolean_difference MESH1 [MESH2]
 *
 * The left viewport shows the two movable inputs and their intersection curves.
 * The right viewport shows MESH1 - MESH2. Press N to randomize both poses.
 */
#include "util/mesh_interactor.hpp"
#include "util/vtk_examples.hpp"

#include <trueform/cpp/core/build_face_membership.hpp>
#include <trueform/cpp/core/build_manifold_edge_link.hpp>
#include <trueform/cpp/core/build_tree.hpp>
#include <trueform/cpp/csg/make_boolean.hpp>

#include <vtkCamera.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkObjectFactory.h>
#include <vtkPolyData.h>
#include <vtkProperty.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkTextActor.h>

#include <chrono>
#include <exception>
#include <filesystem>
#include <iostream>
#include <string>

namespace {

namespace examples = trueform::vtk_examples;

class boolean_difference_interactor : public examples::mesh_interactor {
public:
  static auto New() -> boolean_difference_interactor *;
  vtkTypeMacro(boolean_difference_interactor, examples::mesh_interactor);

  auto initialize(examples::mesh_actor_data *first,
                  examples::mesh_actor_data *second, vtkPolyData *result,
                  vtkPolyData *curves, vtkTextActor *text) -> void {
    _first = first;
    _second = second;
    _result = result;
    _curves = curves;
    _text = text;
    add_mesh(*_first);
    add_mesh(*_second);
    set_dragged_callback(
        [this](examples::mesh_actor_data &, const examples::point3<float> &) {
          compute_boolean();
        });
  }

  auto compute_boolean() -> void {
    const auto start = std::chrono::steady_clock::now();
    auto output = tf::cpp::make_boolean_with_curves(
        _first->mesh(), _second->mesh(), tf::boolean_op::left_difference);
    _times.add(examples::elapsed_seconds(start));

    auto result_poly = examples::to_vtk_polydata(output.mesh);
    auto curve_poly = examples::to_vtk_polydata(output.curves);
    examples::replace_polydata(_result, result_poly);
    examples::replace_polydata(_curves, curve_poly);

    const auto message = std::string("Boolean difference time: ") +
                         examples::format_milliseconds(_times.average());
    _text->SetInput(message.c_str());
    if (this->Interactor)
      this->Interactor->Render();
  }

  auto OnKeyPress() -> void override {
    const auto key = std::string(this->Interactor->GetKeySym());
    if (key == "n" || key == "N") {
      randomize_orientations();
      return;
    }
    vtkInteractorStyleTrackballCamera::OnKeyPress();
  }

private:
  auto randomize(examples::mesh_actor_data &data) -> void {
    const auto current = data.placement;
    const auto local_center =
        examples::point_mean(data.polygons.points_buffer());
    const auto world_center = examples::transform_point(current, local_center);
    const auto next = examples::rotate_around_world_point(
        current, world_center, examples::random_rotation_matrix<float>());
    examples::synchronize(data, next);
  }

  auto randomize_orientations() -> void {
    randomize(*_first);
    randomize(*_second);
    compute_boolean();
  }

  examples::mesh_actor_data *_first = nullptr;
  examples::mesh_actor_data *_second = nullptr;
  vtkPolyData *_result = nullptr;
  vtkPolyData *_curves = nullptr;
  vtkTextActor *_text = nullptr;
  examples::rolling_average _times{100};
};

vtkStandardNewMacro(boolean_difference_interactor);

void print_usage(const char *program) {
  std::cerr << "Usage: " << program << " MESH1 [MESH2]\n\n"
            << "Interactive boolean difference\n\n"
            << "Controls:\n"
            << "  N            Randomize mesh orientations\n"
            << "  Mouse drag   Move meshes / rotate camera\n";
}

} // namespace

int main(int argc, char **argv) {
  auto help = false;
  for (int index = 1; index != argc; ++index)
    help = help || std::string(argv[index]) == "--help" ||
           std::string(argv[index]) == "-h";
  if (argc < 2 || argc > 3 || help) {
    print_usage(argv[0]);
    return help ? 0 : 1;
  }

  try {
    auto first = examples::load_stl_actor(std::filesystem::path(argv[1]),
                                          {0.0F, 0.0F, 0.0F}, true, 10.0F);
    tf::cpp::build_tree(first.mesh());
    tf::cpp::build_face_membership(first.mesh());
    tf::cpp::build_manifold_edge_link(first.mesh());

    auto second =
        argc == 3 ? examples::load_stl_actor(std::filesystem::path(argv[2]),
                                             {15.0F, 0.0F, 0.0F}, true, 10.0F)
                  : examples::share_mesh_actor(first, {15.0F, 0.0F, 0.0F}, true,
                                               10.0F);
    if (argc == 3) {
      tf::cpp::build_tree(second.mesh());
      tf::cpp::build_face_membership(second.mesh());
      tf::cpp::build_manifold_edge_link(second.mesh());
    }

    auto left = vtkSmartPointer<vtkRenderer>::New();
    auto right = vtkSmartPointer<vtkRenderer>::New();
    auto text_renderer = vtkSmartPointer<vtkRenderer>::New();
    left->SetViewport(0.0, 0.12, 0.5, 1.0);
    right->SetViewport(0.5, 0.12, 1.0, 1.0);
    text_renderer->SetViewport(0.0, 0.0, 1.0, 0.12);
    text_renderer->InteractiveOff();
    left->SetBackground(27.0 / 255.0, 43.0 / 255.0, 52.0 / 255.0);
    right->SetBackground(27.0 / 255.0, 43.0 / 255.0, 52.0 / 255.0);
    text_renderer->SetBackground(0.090, 0.143, 0.173);

    left->AddActor(first.actor);
    left->AddActor(second.actor);

    auto curve_poly = vtkSmartPointer<vtkPolyData>::New();
    curve_poly->Initialize();
    auto curve_actor = examples::make_actor(curve_poly, {1.0, 0.0, 0.0});
    curve_actor->GetProperty()->SetRenderLinesAsTubes(true);
    curve_actor->GetProperty()->SetLineWidth(8.0);
    left->AddActor(curve_actor);

    auto result_poly = vtkSmartPointer<vtkPolyData>::New();
    result_poly->Initialize();
    auto result_actor = examples::make_actor(result_poly, {0.8, 0.8, 0.8});
    right->AddActor(result_actor);

    auto text = examples::make_text_actor("Boolean difference time: 0 ms", 38,
                                          {0.03, 0.50});
    text_renderer->AddViewProp(text);

    auto window = vtkSmartPointer<vtkRenderWindow>::New();
    window->AddRenderer(left);
    window->AddRenderer(right);
    window->AddRenderer(text_renderer);
    window->SetSize(1200, 600);

    auto interactor = vtkSmartPointer<vtkRenderWindowInteractor>::New();
    interactor->SetRenderWindow(window);
    auto style = vtkSmartPointer<boolean_difference_interactor>::New();
    style->initialize(&first, &second, result_poly, curve_poly, text);
    style->set_interaction_renderer(left);
    interactor->SetInteractorStyle(style);

    right->SetActiveCamera(left->GetActiveCamera());
    left->ResetCamera();
    right->ResetCamera();
    style->compute_boolean();

    window->Render();
    interactor->Start();
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
