#include "util/vtk_examples.hpp"

#include <trueform/cpp/core/build_tree.hpp>
#include <trueform/cpp/core/cache.hpp>
#include <trueform/cpp/core/index_type.hpp>
#include <trueform/cpp/core/mesh.hpp>
#include <trueform/cpp/iso/isocontours.hpp>
#include <trueform/cpp/spatial/distance.hpp>
#include <trueform/cpp/spatial/primitive.hpp>

#include <vtkActor.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkMatrix4x4.h>
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
#include <cstddef>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace examples = trueform::vtk_examples;
/// A field is cut on the mesh the reader gave, and a mixed carrier is what a
/// mesh read from a file is.
using index_t = tf::cpp::default_index_t;
using mixed_storage = tf::polygons_buffer<index_t, float, 3, tf::dynamic_size>;
using mixed_cache = tf::cpp::cache<index_t, float, 3, tf::dynamic_size>;
using mixed_mesh = tf::cpp::mesh<index_t, float, 3, tf::dynamic_size>;

/// What this example holds: the storage the reader gave and the cache that
/// remembers it. `mesh()` is the assembly every entry takes.
struct surface_source {
  mixed_storage polygons;
  mutable mixed_cache cache;

  auto mesh() const -> mixed_mesh {
    return {polygons.faces(), polygons.points(), cache};
  }
};

class isocontour_interactor final : public vtkInteractorStyleTrackballCamera {
public:
  static auto New() -> isocontour_interactor *;
  vtkTypeMacro(isocontour_interactor, vtkInteractorStyleTrackballCamera);

  auto initialize(const surface_source &value, vtkPolyData *curves,
                  vtkTextActor *timing) -> void {
    _surface = &value;
    _curve_polydata = curves;
    _timing = timing;
    reset_plane();
  }

  auto compute_curves() -> void {
    constexpr int number_of_levels = 10;
    auto wrapped_offset = std::fmod(_distance, _spacing);
    if (wrapped_offset < 0.0F)
      wrapped_offset += _spacing;

    std::vector<float> values;
    values.reserve(number_of_levels + 3);
    for (int i = -1; i != number_of_levels + 2; ++i)
      values.push_back(_minimum + wrapped_offset +
                       static_cast<float>(i) * _spacing);
    const auto thresholds =
        examples::make_array(std::move(values), {number_of_levels + 3});

    const auto start = std::chrono::steady_clock::now();
    const auto curves =
        tf::cpp::isocontours(_surface->mesh(), _scalars, thresholds);
    _times.add(examples::elapsed_seconds(start));
    const auto label = std::string("Isocontours time: ") +
                       examples::format_milliseconds(_times.average());
    _timing->SetInput(label.c_str());

    const auto polydata = examples::to_vtk_polydata(curves);
    examples::replace_polydata(_curve_polydata, polydata);
    if (this->Interactor)
      this->Interactor->Render();
  }

  auto OnKeyPress() -> void override {
    if (std::string(this->Interactor->GetKeySym()) == "n") {
      reset_plane();
      compute_curves();
      return;
    }
    vtkInteractorStyleTrackballCamera::OnKeyPress();
  }

  auto OnMouseWheelForward() -> void override {
    if (this->Interactor->GetControlKey()) {
      _distance += _spacing * 0.1F;
      compute_curves();
      return;
    }
    vtkInteractorStyleTrackballCamera::OnMouseWheelForward();
  }

  auto OnMouseWheelBackward() -> void override {
    if (this->Interactor->GetControlKey()) {
      _distance -= _spacing * 0.1F;
      compute_curves();
      return;
    }
    vtkInteractorStyleTrackballCamera::OnMouseWheelBackward();
  }

protected:
  isocontour_interactor() = default;
  ~isocontour_interactor() override = default;

private:
  auto reset_plane() -> void {
    if (!_surface || _surface->polygons.points_buffer().size() == 0)
      throw std::invalid_argument("isocontours requires a non-empty mesh");

    std::normal_distribution<float> normal_distribution(0.0F, 1.0F);
    examples::point3<float> normal{normal_distribution(_generator),
                                   normal_distribution(_generator),
                                   normal_distribution(_generator)};
    const auto normal_length = std::sqrt(
        normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);
    for (auto &coordinate : normal)
      coordinate /= normal_length;

    const auto points = _surface->polygons.points();
    std::uniform_int_distribution<std::size_t> point_distribution(
        0, points.size() - 1);
    const auto source = points[point_distribution(_generator)];
    set_plane({source[0], source[1], source[2]}, normal);
  }

  auto set_plane(const examples::point3<float> &point,
                 const examples::point3<float> &normal) -> void {
    const auto plane_d =
        -(normal[0] * point[0] + normal[1] * point[1] + normal[2] * point[2]);
    const auto plane = tf::cpp::primitive<float>(
        tf::cpp::primitive_kind::plane,
        examples::make_array<float>({normal[0], normal[1], normal[2], plane_d},
                                    {4}));
    const auto points = tf::cpp::primitive<float>(
        tf::cpp::primitive_kind::point,
        examples::points_array(_surface->polygons.points_buffer()));
    _scalars = tf::cpp::distance(points, plane).batch();
    _distance = 0.0F;
    const auto limits = std::minmax_element(_scalars.begin(), _scalars.end());
    _minimum = *limits.first;
    _maximum = *limits.second;
    _spacing = (_maximum - _minimum) / 10.0F;
  }

  const surface_source *_surface = nullptr;
  vtkPolyData *_curve_polydata = nullptr;
  vtkTextActor *_timing = nullptr;
  tf::cpp::nd_array<float> _scalars;
  float _distance = 0.0F;
  float _minimum = 0.0F;
  float _maximum = 0.0F;
  float _spacing = 0.1F;
  examples::rolling_average _times{100};
  std::mt19937 _generator{std::random_device{}()};
};

vtkStandardNewMacro(isocontour_interactor);

} // namespace

int main(int argc, char **argv) {
  try {
    const auto help = argc == 2 && (std::string(argv[1]) == "-h" ||
                                    std::string(argv[1]) == "--help");
    if (argc != 2 || help) {
      auto &stream = help ? std::cout : std::cerr;
      stream << "Usage: " << argv[0] << " mesh.stl\n\n"
             << "Controls:\n"
             << "  N              Randomize cutting plane\n"
             << "  Ctrl + Scroll  Move isocontour levels\n"
             << "  Mouse drag     Rotate camera\n";
      return help ? 0 : 2;
    }

    surface_source surface{examples::as_mixed_mesh(tf::cpp::read_stl(argv[1])),
                           {}};
    examples::materialize_center_and_scale(surface.polygons, 10.0F);
    tf::cpp::build_tree(surface.mesh());
    const auto source_polydata = examples::to_vtk_polydata(surface.polygons);

    auto renderer = vtkSmartPointer<vtkRenderer>::New();
    const auto text_renderer = examples::make_text_strip(renderer);

    const auto mesh_actor = examples::make_actor(source_polydata);
    mesh_actor->GetProperty()->SetColor(0.8, 0.8, 0.8);
    auto identity = vtkSmartPointer<vtkMatrix4x4>::New();
    identity->Identity();
    mesh_actor->SetUserMatrix(identity);
    renderer->AddActor(mesh_actor);

    auto curve_polydata = vtkSmartPointer<vtkPolyData>::New();
    curve_polydata->Initialize();
    const auto curve_actor = examples::make_actor(curve_polydata);
    curve_actor->GetProperty()->SetColor(1.0, 0.1, 0.1);
    curve_actor->GetProperty()->SetRenderLinesAsTubes(true);
    curve_actor->GetProperty()->SetLineWidth(8.0);
    renderer->AddActor(curve_actor);

    const auto timing =
        examples::make_text_actor("Isocontours time: 0 ms", 38, {0.03, 0.5});
    text_renderer->AddViewProp(timing);

    auto window = vtkSmartPointer<vtkRenderWindow>::New();
    window->AddRenderer(renderer);
    window->AddRenderer(text_renderer);
    window->SetSize(800, 600);

    auto interactor = vtkSmartPointer<vtkRenderWindowInteractor>::New();
    interactor->SetRenderWindow(window);
    auto style = vtkSmartPointer<isocontour_interactor>::New();
    style->initialize(surface, curve_polydata, timing);
    interactor->SetInteractorStyle(style);

    style->compute_curves();
    window->Render();
    interactor->Start();
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "isocontours: " << error.what() << '\n';
    return 1;
  }
}
