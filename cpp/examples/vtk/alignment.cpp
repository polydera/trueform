#include "util/mesh_interactor.hpp"
#include "util/vtk_examples.hpp"

#include <trueform/core/points_buffer.hpp>
#include <trueform/core/transformation_view.hpp>
#include <trueform/core/unit_vectors.hpp>
#include <trueform/cpp/core/build_tree.hpp>
#include <trueform/cpp/core/cache.hpp>
#include <trueform/cpp/core/index_type.hpp>
#include <trueform/cpp/core/nd_array.hpp>
#include <trueform/cpp/core/point_cloud.hpp>
#include <trueform/cpp/core/point_cloud_cache.hpp>
#include <trueform/cpp/geometry/chamfer_error.hpp>
#include <trueform/cpp/geometry/fit_icp.hpp>
#include <trueform/cpp/geometry/fit_obb.hpp>
#include <trueform/cpp/geometry/normals.hpp>
#include <trueform/cpp/geometry/point_normals.hpp>
#include <trueform/cpp/geometry/taubin_smoothed.hpp>
#include <trueform/cpp/io/read_stl.hpp>

#include <vtkActor.h>
#include <vtkObjectFactory.h>
#include <vtkProperty.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkSmartPointer.h>
#include <vtkTextActor.h>

#include <array>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ex = trueform::vtk_examples;

namespace {

constexpr ex::point3<double> source_color{0.0, 0.659, 0.604};
constexpr ex::point3<double> source_highlight{0.0, 0.75, 0.68};
constexpr ex::point3<double> target_color{0.2, 0.35, 0.33};
constexpr ex::point3<double> aligned_color{0.0, 0.835, 0.745};

struct arguments {
  std::filesystem::path source;
  std::filesystem::path target;
};

auto print_usage(const char *program) -> void {
  std::cout
      << "Usage: " << program << " [--source FILE] [--target FILE]\n"
      << "Interactive point-cloud alignment (OBB + point-to-plane ICP).\n\n"
      << "Controls:\n"
      << "  Left-drag on source mesh  Translate\n"
      << "  Right-drag                Rotate source mesh\n"
      << "  A                         Run alignment\n"
      << "  R                         Randomize source pose\n";
}

auto parse_arguments(int argc, char **argv) -> arguments {
  const auto data =
      std::filesystem::path(TRUEFORM_DATA_DIR) / "benchmarks" / "data";
  arguments result{data / "dragon-500k.stl", data / "dragon-50k.stl"};
  for (int i = 1; i < argc; ++i) {
    const std::string option = argv[i];
    if (option == "-h" || option == "--help") {
      print_usage(argv[0]);
      std::exit(0);
    }
    if (option == "--source" || option == "--target") {
      if (++i == argc)
        throw std::invalid_argument(option + " requires a file path");
      (option == "--source" ? result.source : result.target) = argv[i];
      continue;
    }
    constexpr auto source_prefix = "--source=";
    constexpr auto target_prefix = "--target=";
    if (option.rfind(source_prefix, 0) == 0) {
      result.source =
          option.substr(std::char_traits<char>::length(source_prefix));
      continue;
    }
    if (option.rfind(target_prefix, 0) == 0) {
      result.target =
          option.substr(std::char_traits<char>::length(target_prefix));
      continue;
    }
    throw std::invalid_argument("unknown argument: " + option);
  }
  return result;
}

/// What this example holds when it shows a point cloud: the points, the
/// normals they carry when they have some, the cache that remembers them, and
/// the placement this instance is drawn at. `cloud()` is the assembly every
/// entry takes.
struct cloud_source {
  tf::points_buffer<float, 3> points;
  tf::cpp::nd_array<float> normals;
  mutable tf::cpp::point_cloud_cache<float> cache;
  ex::matrix4<float> placement = ex::identity_matrix<float>();

  auto cloud() const -> tf::cpp::point_cloud<float> {
    return {points.points(), tf::make_unit_vectors<3>(normals.make_range()),
            cache, tf::make_transformation_view<3>(placement.data())};
  }
};

auto sampled_points(const tf::points_buffer<float, 3> &storage, int stride)
    -> tf::points_buffer<float, 3> {
  const auto source = storage.points();
  tf::points_buffer<float, 3> result;
  auto &values = result.data_buffer();
  values.reserve((source.size() + static_cast<std::size_t>(stride) - 1) * 3 /
                 static_cast<std::size_t>(stride));
  for (std::size_t point = 0; point < source.size();
       point += static_cast<std::size_t>(stride))
    for (std::size_t axis = 0; axis != 3; ++axis)
      values.push_back(source[point][axis]);
  return result;
}

class alignment_interactor final : public ex::mesh_interactor {
public:
  static auto New() -> alignment_interactor *;
  vtkTypeMacro(alignment_interactor, ex::mesh_interactor);

  auto initialize(ex::mesh_actor_data &source, cloud_source &cloud,
                  cloud_source &sample_cloud, cloud_source &target,
                  vtkTextActor *instructions, vtkTextActor *chamfer) -> void {
    source_ = &source;
    source_cloud_ = &cloud;
    source_sample_cloud_ = &sample_cloud;
    target_cloud_ = &target;
    instructions_ = instructions;
    chamfer_ = chamfer;
    set_colors(source_color, source_highlight);
    add_mesh(source);
    set_dragged_callback(
        [this](ex::mesh_actor_data &, const ex::point3<float> &) {
          synchronize_clouds();
          update_chamfer();
        });
    set_released_callback([this](ex::mesh_actor_data *) {
      synchronize_clouds();
      update_chamfer();
      this->Interactor->Render();
    });
    update_chamfer();
  }

  auto OnRightButtonDown() -> void override {
    const auto *position = this->Interactor->GetEventPosition();
    last_position_ = {position[0], position[1]};
    const auto center = ex::point_mean(source_->polygons.points_buffer());
    rotation_center_ = ex::transform_point(
        ex::from_vtk_matrix<float>(source_->matrix), center);
    rotating_ = true;
    this->Interactor->GetRenderWindow()->HideCursor();
  }

  auto OnRightButtonUp() -> void override {
    if (!rotating_) {
      vtkInteractorStyleTrackballCamera::OnRightButtonUp();
      return;
    }
    rotating_ = false;
    this->Interactor->GetRenderWindow()->ShowCursor();
    synchronize_clouds();
    update_chamfer();
  }

  auto OnMouseMove() -> void override {
    if (!rotating_) {
      ex::mesh_interactor::OnMouseMove();
      return;
    }
    const auto *position = this->Interactor->GetEventPosition();
    const auto dx = position[0] - last_position_[0];
    const auto dy = position[1] - last_position_[1];
    last_position_ = {position[0], position[1]};
    constexpr float radians_per_degree = 0.01745329251994329576923690768489F;
    const auto rotation = ex::multiply(
        ex::rotation_y(static_cast<float>(dx) * 0.5F * radians_per_degree),
        ex::rotation_x(static_cast<float>(dy) * 0.5F * radians_per_degree));
    const auto next = ex::rotate_around_world_point(
        ex::from_vtk_matrix<float>(source_->matrix), rotation_center_,
        rotation);
    ex::synchronize(*source_, next);
    synchronize_clouds();
    update_chamfer();
    this->Interactor->Render();
  }

  auto OnKeyPress() -> void override {
    auto key = std::string(this->Interactor->GetKeySym());
    for (auto &character : key)
      character = static_cast<char>(
          std::tolower(static_cast<unsigned char>(character)));
    if (key == "a") {
      run_alignment();
      return;
    }
    if (key == "r") {
      randomize_source();
      return;
    }
    vtkInteractorStyleTrackballCamera::OnKeyPress();
  }

private:
  auto synchronize_clouds() -> void {
    const auto transform = ex::from_vtk_matrix<float>(source_->matrix);
    source_cloud_->placement = transform;
    source_sample_cloud_->placement = transform;
  }

  auto update_chamfer() -> void {
    const auto error = tf::cpp::chamfer_error(source_sample_cloud_->cloud(),
                                              target_cloud_->cloud());
    std::ostringstream text;
    text << std::fixed << std::setprecision(4) << "Chamfer error: " << error;
    chamfer_->SetInput(text.str().c_str());
  }

  auto randomize_source() -> void {
    const auto current = ex::from_vtk_matrix<float>(source_->matrix);
    const auto center_local = ex::point_mean(source_->polygons.points_buffer());
    const auto center_world = ex::transform_point(current, center_local);
    static thread_local std::mt19937 generator(std::random_device{}());
    std::uniform_real_distribution<float> offset(-5.0F, 5.0F);
    const ex::point3<float> destination{center_world[0] + offset(generator),
                                        center_world[1] + offset(generator),
                                        center_world[2] + offset(generator)};
    const auto next = ex::multiply(
        ex::translation(destination[0], destination[1], destination[2]),
        ex::multiply(
            ex::random_rotation_matrix<float>(),
            ex::multiply(ex::translation(-center_world[0], -center_world[1],
                                         -center_world[2]),
                         current)));
    ex::synchronize(*source_, next);
    synchronize_clouds();
    source_->actor->GetProperty()->SetColor(source_color[0], source_color[1],
                                            source_color[2]);
    update_chamfer();
    this->Interactor->Render();
  }

  auto run_alignment() -> void {
    instructions_->SetInput("Aligning...");
    this->Interactor->Render();
    const auto start = std::chrono::steady_clock::now();

    const auto source_transform = source_cloud_->placement;
    const auto obb_delta = ex::from_nd_array(
        tf::cpp::fit_obb(source_cloud_->cloud(), target_cloud_->cloud()));
    const auto obb_transform = ex::multiply(obb_delta, source_transform);
    source_cloud_->placement = obb_transform;

    tf::cpp::fit_icp_options<float> options;
    options.max_iterations = 50;
    options.n_samples = 1000;
    options.k = 1;
    const auto icp_delta = ex::from_nd_array(tf::cpp::fit_icp(
        source_cloud_->cloud(), target_cloud_->cloud(), options));
    const auto final_transform = ex::multiply(icp_delta, obb_transform);

    ex::synchronize(*source_, final_transform);
    source_cloud_->placement = final_transform;
    source_sample_cloud_->placement = final_transform;
    source_->actor->GetProperty()->SetColor(aligned_color[0], aligned_color[1],
                                            aligned_color[2]);
    const auto message =
        "Aligned in " + ex::format_milliseconds(ex::elapsed_seconds(start));
    instructions_->SetInput(message.c_str());
    update_chamfer();
    this->Interactor->Render();
  }

  ex::mesh_actor_data *source_ = nullptr;
  cloud_source *source_cloud_ = nullptr;
  cloud_source *source_sample_cloud_ = nullptr;
  cloud_source *target_cloud_ = nullptr;
  vtkTextActor *instructions_ = nullptr;
  vtkTextActor *chamfer_ = nullptr;
  ex::point3<float> rotation_center_{};
  std::array<int, 2> last_position_{};
  bool rotating_ = false;
};

vtkStandardNewMacro(alignment_interactor);

} // namespace

int main(int argc, char **argv) {
  try {
    const auto args = parse_arguments(argc, argv);

    std::cout << "Loading source mesh: " << args.source << '\n';
    auto source_mesh = tf::cpp::read_stl(args.source.string());
    std::cout << "  " << source_mesh.size() << " faces, "
              << source_mesh.points_buffer().size() << " points\n";

    std::cout << "Loading target mesh: " << args.target << '\n';
    const auto read_target = tf::cpp::read_stl(args.target.string());
    std::cout << "  " << read_target.size() << " faces, "
              << read_target.points_buffer().size() << " points\n"
              << "Smoothing target mesh (50 Taubin iterations)...\n";
    tf::cpp::cache<tf::cpp::default_index_t, float> read_target_cache;
    auto target_mesh = tf::cpp::taubin_smoothed(
        ex::mesh_over(read_target, read_target_cache), 50, 0.9F);
    tf::cpp::cache<tf::cpp::default_index_t, float> smoothed_cache;
    auto target_normals =
        tf::cpp::point_normals(ex::mesh_over(target_mesh, smoothed_cache));
    std::cout << "  Done.\n";

    const auto source_center =
        ex::center_and_scale_matrix(source_mesh.points_buffer(), 10.0F);
    const auto target_transform =
        ex::center_and_scale_matrix(target_mesh.points_buffer(), 10.0F);
    const auto source_transform = ex::multiply(
        ex::translation(5.0F, 3.0F, 2.0F),
        ex::multiply(ex::random_rotation_matrix<float>(), source_center));

    // A cloud is points, a cache and a placement, exactly as a mesh is: these
    // read copies of the meshes' coordinates, so the meshes may be moved on.
    cloud_source source_cloud{ex::copied_points(source_mesh.points_buffer()),
                              {},
                              {},
                              source_transform};
    tf::cpp::build_tree(source_cloud.cloud());
    cloud_source source_sample_cloud{
        sampled_points(source_mesh.points_buffer(), 10),
        {},
        {},
        source_transform};

    cloud_source target_cloud{ex::copied_points(target_mesh.points_buffer()),
                              std::move(target_normals),
                              {},
                              target_transform};
    tf::cpp::build_tree(target_cloud.cloud());

    auto source_data =
        ex::make_mesh_actor_data(std::move(source_mesh), source_transform);
    tf::cpp::build_tree(source_data.mesh());
    source_data.actor->GetProperty()->SetColor(source_color[0], source_color[1],
                                               source_color[2]);
    auto target_data =
        ex::make_mesh_actor_data(std::move(target_mesh), target_transform);
    tf::cpp::build_tree(target_data.mesh());
    target_data.actor->GetProperty()->SetColor(target_color[0], target_color[1],
                                               target_color[2]);
    target_data.actor->GetProperty()->SetOpacity(0.4);

    auto renderer = vtkSmartPointer<vtkRenderer>::New();
    renderer->SetBackground(0.1, 0.12, 0.15);
    renderer->SetViewport(0.0, 0.1, 1.0, 1.0);
    renderer->AddActor(source_data.actor);
    renderer->AddActor(target_data.actor);
    auto text_renderer = ex::make_text_strip(renderer, 0.1);
    text_renderer->SetBackground(0.08, 0.1, 0.12);

    auto instructions = ex::make_text_actor(
        "Drag: move | Right-drag: rotate | A: align | R: randomize", 24,
        {0.02, 0.5});
    auto chamfer =
        ex::make_text_actor("Chamfer error: --", 24, {0.98, 0.5}, true);
    text_renderer->AddViewProp(instructions);
    text_renderer->AddViewProp(chamfer);

    auto window = vtkSmartPointer<vtkRenderWindow>::New();
    window->AddRenderer(renderer);
    window->AddRenderer(text_renderer);
    window->SetSize(1200, 800);
    window->SetWindowName("Interactive Alignment - trueform");
    auto interactor = vtkSmartPointer<vtkRenderWindowInteractor>::New();
    interactor->SetRenderWindow(window);
    auto style = vtkSmartPointer<alignment_interactor>::New();
    style->initialize(source_data, source_cloud, source_sample_cloud,
                      target_cloud, instructions, chamfer);
    style->set_interaction_renderer(renderer);
    interactor->SetInteractorStyle(style);

    renderer->ResetCamera();
    window->Render();
    std::cout << "\nControls:\n"
              << "  Left-drag on source mesh: translate\n"
              << "  Right-drag on source mesh: rotate\n"
              << "  A: run alignment (OBB + ICP)\n"
              << "  R: randomize source position\n\n";
    interactor->Start();
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "alignment: " << error.what() << '\n';
    return 1;
  }
}
