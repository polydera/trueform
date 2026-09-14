#include "util/mesh_interactor.hpp"
#include "util/vtk_examples.hpp"

#include <trueform/cpp/core/build_tree.hpp>
#include <trueform/cpp/spatial/intersects.hpp>

#include <vtkActor.h>
#include <vtkObjectFactory.h>
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
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace ex = trueform::vtk_examples;

namespace {

constexpr ex::point3<double> normal_color{0.8, 0.8, 0.8};
constexpr ex::point3<double> highlight_color{0.85, 0.85, 0.9};
constexpr ex::point3<double> colliding_color{0.8, 1.0, 1.0};

auto print_usage(const char *program) -> void {
  std::cout << "Usage: " << program << " MESH [MESH ...]\n"
            << "Interactive collision detection in a 5x5 grid.\n\n"
            << "Controls:\n"
            << "  Hover       Highlight mesh\n"
            << "  Mouse drag  Move mesh (colliding meshes turn cyan)\n";
}

auto parse_arguments(int argc, char **argv)
    -> std::vector<std::filesystem::path> {
  std::vector<std::filesystem::path> paths;
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    if (argument == "-h" || argument == "--help") {
      print_usage(argv[0]);
      std::exit(0);
    }
    if (!argument.empty() && argument.front() == '-')
      throw std::invalid_argument("unknown option: " + argument);
    paths.emplace_back(argument);
  }
  if (paths.empty())
    throw std::invalid_argument("at least one mesh file is required");
  return paths;
}

class collision_interactor final : public ex::mesh_interactor {
public:
  static auto New() -> collision_interactor *;
  vtkTypeMacro(collision_interactor, ex::mesh_interactor);

  auto initialize(std::vector<ex::mesh_actor_data> &meshes,
                  vtkTextActor *timing_text) -> void {
    meshes_ = &meshes;
    timing_text_ = timing_text;
    set_colors(normal_color, highlight_color);
    for (auto &mesh : meshes)
      add_mesh(mesh);
    set_dragged_callback(
        [this](ex::mesh_actor_data &, const ex::point3<float> &) {
          check_collisions();
        });
    set_released_callback([this](ex::mesh_actor_data *) {
      collisions_.clear();
      this->Interactor->Render();
    });
  }

private:
  auto check_collisions() -> void {
    auto *selected = selected_mesh();
    if (!selected)
      return;
    collisions_.clear();
    const auto start = std::chrono::steady_clock::now();
    for (auto &other : *meshes_) {
      if (&other == selected)
        continue;
      if (tf::cpp::intersects(selected->mesh(), other.mesh()).scalar())
        collisions_.insert(&other);
    }
    times_.add(ex::elapsed_seconds(start));
    const auto timing =
        "Collision time: " + ex::format_microseconds(times_.average());
    timing_text_->SetInput(timing.c_str());

    for (auto &mesh : *meshes_) {
      const auto color = &mesh == selected
                             ? highlight_color
                             : (collisions_.count(&mesh) != 0 ? colliding_color
                                                              : normal_color);
      mesh.actor->GetProperty()->SetColor(color[0], color[1], color[2]);
    }
  }

  std::vector<ex::mesh_actor_data> *meshes_ = nullptr;
  vtkTextActor *timing_text_ = nullptr;
  std::unordered_set<ex::mesh_actor_data *> collisions_;
  ex::rolling_average times_{1000};
};

vtkStandardNewMacro(collision_interactor);

} // namespace

int main(int argc, char **argv) {
  try {
    const auto mesh_files = parse_arguments(argc, argv);

    std::map<std::filesystem::path, ex::mesh_actor_data> sources;
    for (const auto &filename : mesh_files) {
      if (sources.count(filename) != 0)
        continue;
      auto source =
          ex::load_stl_actor(filename, {0.0F, 0.0F, 0.0F}, false, 10.0F);
      tf::cpp::build_tree(source.mesh());
      sources.emplace(filename, std::move(source));
    }

    constexpr int grid_size = 5;
    constexpr float spacing = 15.0F;
    std::vector<ex::mesh_actor_data> meshes;
    meshes.reserve(grid_size * grid_size);
    int mesh_index = 0;
    for (int i = 0; i != grid_size; ++i)
      for (int j = 0; j != grid_size; ++j) {
        const auto x = static_cast<float>(i) * spacing -
                       static_cast<float>(grid_size - 1) * spacing / 2.0F;
        const auto y = static_cast<float>(j) * spacing -
                       static_cast<float>(grid_size - 1) * spacing / 2.0F;
        const auto &filename = mesh_files[static_cast<std::size_t>(mesh_index) %
                                          mesh_files.size()];
        auto instance =
            ex::share_mesh_actor(sources.at(filename), {x, y, 0.0F}, true);
        instance.actor->GetProperty()->SetColor(
            normal_color[0], normal_color[1], normal_color[2]);
        meshes.push_back(std::move(instance));
        ++mesh_index;
      }

    auto renderer = vtkSmartPointer<vtkRenderer>::New();
    auto text_renderer = ex::make_text_strip(renderer);
    for (auto &mesh : meshes)
      renderer->AddActor(mesh.actor);
    auto timing_text =
        ex::make_text_actor("Collision time: 0.0 us", 38, {0.03, 0.5});
    text_renderer->AddViewProp(timing_text);

    auto window = vtkSmartPointer<vtkRenderWindow>::New();
    window->AddRenderer(renderer);
    window->AddRenderer(text_renderer);
    window->SetSize(800, 600);
    auto interactor = vtkSmartPointer<vtkRenderWindowInteractor>::New();
    interactor->SetRenderWindow(window);
    auto style = vtkSmartPointer<collision_interactor>::New();
    style->initialize(meshes, timing_text);
    style->set_interaction_renderer(renderer);
    interactor->SetInteractorStyle(style);

    renderer->ResetCamera();
    window->Render();
    interactor->Start();
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "collision: " << error.what() << '\n';
    return 1;
  }
}
