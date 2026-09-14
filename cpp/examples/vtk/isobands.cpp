#include "util/vtk_examples.hpp"

#include <trueform/cpp/core/cache.hpp>
#include <trueform/cpp/core/index_type.hpp>
#include <trueform/cpp/core/mesh.hpp>
#include <trueform/cpp/iso/isobands.hpp>
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
#include <cstdint>
#include <iostream>
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

class isoband_interactor final : public vtkInteractorStyleTrackballCamera {
public:
  static auto New() -> isoband_interactor *;
  vtkTypeMacro(isoband_interactor, vtkInteractorStyleTrackballCamera);

  auto initialize(const surface_source &value, vtkPolyData *bands,
                  vtkPolyData *curves, vtkTextActor *timing) -> void {
    _surface = &value;
    _band_polydata = bands;
    _curve_polydata = curves;
    _timing = timing;
    reset_plane();
  }

  auto compute_bands() -> void {
    constexpr int number_of_levels = 10;
    auto wrapped_offset = std::fmod(_distance, _spacing);
    if (wrapped_offset < 0.0F)
      wrapped_offset += _spacing;

    std::vector<float> cut_values;
    cut_values.reserve(number_of_levels + 3);
    for (int i = -1; i != number_of_levels + 2; ++i)
      cut_values.push_back(_minimum + wrapped_offset +
                           static_cast<float>(i) * _spacing);
    const auto cuts =
        examples::make_array(std::move(cut_values), {number_of_levels + 3});

    const auto period =
        static_cast<long long>(std::floor(_distance / _spacing));
    const auto parity = static_cast<int>((period % 2 + 2) % 2);
    std::vector<std::int32_t> selected_values;
    for (int band = 0; band != number_of_levels + 2; ++band)
      if ((band & 1) == parity)
        selected_values.push_back(static_cast<std::int32_t>(band));
    const auto selected =
        examples::make_array(std::move(selected_values),
                             {static_cast<int>((number_of_levels + 2) / 2)});

    const auto start = std::chrono::steady_clock::now();
    const auto result = tf::cpp::isobands_with_curves_selected(
        _surface->mesh(), _scalars, cuts, selected);
    _times.add(examples::elapsed_seconds(start));
    const auto label = std::string("Isobands time: ") +
                       examples::format_milliseconds(_times.average());
    _timing->SetInput(label.c_str());

    const auto band_polydata = examples::to_vtk_polydata(result.mesh);
    examples::replace_polydata(_band_polydata, band_polydata);
    const auto curve_polydata = examples::to_vtk_polydata(result.curves);
    examples::replace_polydata(_curve_polydata, curve_polydata);
    if (this->Interactor)
      this->Interactor->Render();
  }

  auto OnKeyPress() -> void override {
    if (std::string(this->Interactor->GetKeySym()) == "n") {
      randomize_plane();
      compute_bands();
      return;
    }
    vtkInteractorStyleTrackballCamera::OnKeyPress();
  }

  auto OnMouseWheelForward() -> void override {
    if (this->Interactor->GetControlKey()) {
      _distance += _spacing * 0.1F;
      compute_bands();
      return;
    }
    vtkInteractorStyleTrackballCamera::OnMouseWheelForward();
  }

  auto OnMouseWheelBackward() -> void override {
    if (this->Interactor->GetControlKey()) {
      _distance -= _spacing * 0.1F;
      compute_bands();
      return;
    }
    vtkInteractorStyleTrackballCamera::OnMouseWheelBackward();
  }

protected:
  isoband_interactor() = default;
  ~isoband_interactor() override = default;

private:
  auto reset_plane() -> void {
    if (!_surface || _surface->polygons.points_buffer().size() == 0)
      throw std::invalid_argument("isobands requires a non-empty mesh");
    auto normal = examples::point3<float>{1.0F, 2.0F, 1.0F};
    const auto length = std::sqrt(6.0F);
    for (auto &coordinate : normal)
      coordinate /= length;
    set_plane(examples::point_mean(_surface->polygons.points_buffer()), normal);
  }

  auto randomize_plane() -> void {
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
  vtkPolyData *_band_polydata = nullptr;
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

vtkStandardNewMacro(isoband_interactor);

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
             << "  Ctrl + Scroll  Move isoband levels\n"
             << "  Mouse drag     Rotate camera\n";
      return help ? 0 : 2;
    }

    surface_source surface{examples::as_mixed_mesh(tf::cpp::read_stl(argv[1])),
                           {}};
    examples::materialize_center_and_scale(surface.polygons, 10.0F);
    const auto source_polydata = examples::to_vtk_polydata(surface.polygons);

    auto renderer = vtkSmartPointer<vtkRenderer>::New();
    const auto text_renderer = examples::make_text_strip(renderer);

    const auto mesh_actor = examples::make_actor(source_polydata);
    mesh_actor->GetProperty()->SetColor(0.5, 0.5, 0.55);
    mesh_actor->GetProperty()->SetOpacity(0.25);
    auto identity = vtkSmartPointer<vtkMatrix4x4>::New();
    identity->Identity();
    mesh_actor->SetUserMatrix(identity);
    renderer->AddActor(mesh_actor);

    auto band_polydata = vtkSmartPointer<vtkPolyData>::New();
    band_polydata->Initialize();
    const auto band_actor = examples::make_actor(band_polydata);
    band_actor->GetProperty()->SetColor(0.0, 0.659, 0.604);
    renderer->AddActor(band_actor);

    auto curve_polydata = vtkSmartPointer<vtkPolyData>::New();
    curve_polydata->Initialize();
    const auto curve_actor = examples::make_actor(curve_polydata);
    curve_actor->GetProperty()->SetColor(0.0, 0.835, 0.745);
    curve_actor->GetProperty()->SetRenderLinesAsTubes(true);
    curve_actor->GetProperty()->SetLineWidth(8.0);
    renderer->AddActor(curve_actor);

    const auto timing =
        examples::make_text_actor("Isobands time: 0 ms", 38, {0.03, 0.5});
    text_renderer->AddViewProp(timing);

    auto window = vtkSmartPointer<vtkRenderWindow>::New();
    window->AddRenderer(renderer);
    window->AddRenderer(text_renderer);
    window->SetSize(800, 600);

    auto interactor = vtkSmartPointer<vtkRenderWindowInteractor>::New();
    interactor->SetRenderWindow(window);
    auto style = vtkSmartPointer<isoband_interactor>::New();
    style->initialize(surface, band_polydata, curve_polydata, timing);
    interactor->SetInteractorStyle(style);

    style->compute_bands();
    window->Render();
    interactor->Start();
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "isobands: " << error.what() << '\n';
    return 1;
  }
}
