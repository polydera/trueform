#include "./util/common.hpp"
#include "./util/data_bridge.hpp"
#include "trueform/random.hpp"
#include "trueform/spatial.hpp"
#include "vtkInteractorStyleTrackballCamera.h"
#include "vtkMatrix4x4.h"
#include "vtkOpenGLActor.h"
#include "vtkOpenGLPolyDataMapper.h"
#include "vtkProperty.h"
#include "vtkRenderWindow.h"
#include "vtkRenderWindowInteractor.h"
#include "vtkRenderer.h"
#include "vtkTextActor.h"
#include "vtkTextProperty.h"
#include <filesystem>
#include <set>
#include <string>
#include <string_view>

class geometry_handle_t {
public:
  auto push_back(vtkActor *actor, vtkPolyData *poly,
                 tf::tree<int, float, 3> &tree) -> void {
    map[actor] = actors.size();
    actors.push_back(actor);
    polys.push_back(poly);
    frames.emplace_back();
    frames.back().fill(actor->GetUserMatrix()->GetData());
    trees.push_back(&tree);
  }

  auto get_actors() const -> const auto & { return actors; }

  auto ray_hit(tf::ray<float, 3> ray)
      -> std::pair<vtkActor *, tf::point<float, 3>> {
    tf::tree_ray_info<int, tf::ray_cast_info<float>> result;
    tf::ray_config<float> config{};
    vtkActor *picked = nullptr;

    for (std::size_t i = 0; i < polys.size(); ++i) {
      auto form = tf::make_form(frames[i], *trees[i], get_triangles(polys[i]));
      auto res = tf::ray_cast(ray, form, config);
      if (res) {
        result = res;
        config.max_t = result.info.t;
        picked = actors[i];
      }
    }
    return std::make_pair(picked, ray.origin + result.info.t * ray.direction);
  }

  auto intersects_any(vtkActor *actor, std::set<vtkActor *> &colliding) {
    std::size_t id = map[actor];
    auto form0 =
        tf::make_form(frames[id], *trees[id], get_triangles(polys[id]));
    for (std::size_t i = 0; i < polys.size(); ++i) {
      if (i == id)
        continue;
      auto collision = tf::intersects(
          form0, tf::make_form(frames[i], *trees[i], get_triangles(polys[i])));
      if (collision)
        colliding.insert(actors[i]);
      else {
        colliding.erase(actors[i]);
      }
    }
  }

  auto update_frame(vtkActor *actor) -> void {
    auto id = map[actor];
    frames[id].fill(actors[id]->GetUserMatrix()->GetData());
  }

private:
  std::map<vtkActor *, int> map;
  std::vector<vtkPolyData *> polys;
  std::vector<vtkActor *> actors;
  std::vector<tf::frame<float, 3>> frames;
  std::vector<tf::tree<int, float, 3> *> trees;
};

class cursor_interactor : public vtkInteractorStyleTrackballCamera {
private:
  geometry_handle_t geometry_handle;
  std::vector<float> pick_times;
  std::vector<float> collide_times;
  int time_index = 0;
  vtkTextActor *pick_text;
  vtkTextActor *collide_text;
  std::array<double, 3> normal_mesh_color{0.8, 0.8, 0.8};
  std::array<double, 3> coliding_mesh_color{0.8, 1, 1};
  std::array<double, 3> selected_mesh_color{1, 0.9, 1};

  tf::plane<float, 3> moving_plane;
  tf::point<float, 3> last_point;
  tf::vector<float, 3> dx;
  vtkActor *selected_actor = nullptr;
  bool selected_mode = false;
  bool camera_mode = false;
  std::set<vtkActor *> colliding;

  auto add_time(std::vector<float> &times, float t) {
    if (times.size() < 100) {
      times.push_back(t);
    } else {
      times[time_index] = t;
    }
    time_index = (time_index + 1) % 100;
    float sum = 0;
    for (auto time : times)
      sum += time;
    return sum / times.size();
  }

  auto add_pick_time(float t) {
    auto pick_time = add_time(pick_times, t);
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "Picking time per frame  : %.1f mcs",
                  pick_time * 1000);
    pick_text->SetInput(buffer);
  }

  auto add_collide_time(float t) {
    auto collide_time = add_time(collide_times, t);
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "Collision time per frame: %.1f mcs",
                  collide_time * 1000);
    collide_text->SetInput(buffer);
  }

  auto get_ray_and_renderer() {
    auto x = this->Interactor->GetEventPosition()[0];
    auto y = this->Interactor->GetEventPosition()[1];
    auto renderer = this->Interactor->FindPokedRenderer(x, y);
    auto ray = get_world_point_ray(renderer, x, y);
    tf::normalize(ray.direction);
    tf::ray<float, 3> _ray{ray.origin, ray.direction};
    return std::make_pair(_ray, renderer);
  }

  auto make_moving_plane(tf::point<float, 3> origin, vtkRenderer *renderer) {
    vtkCamera *camera = renderer->GetActiveCamera();
    // Get the camera's position and focal point
    double cameraPosition[3];
    double focalPoint[3];

    camera->GetPosition(cameraPosition);
    camera->GetFocalPoint(focalPoint);
    auto normal = tf::make_unit_vector(
        tf::make_vector_view<3>(&focalPoint[0]).as<float>() -
        tf::make_vector_view<3>(&cameraPosition[0]).as<float>());
    moving_plane = tf::make_plane(normal, origin);
  }

  auto move_selected(vtkActor *selected_actor) {
    for (int i = 0; i < 3; ++i)
      selected_actor->GetUserMatrix()->Element[i][3] += dx[i];
    selected_actor->GetUserMatrix()->Modified();
    geometry_handle.update_frame(selected_actor);
  }

  auto handle_collisions() {
    tf::tick();
    geometry_handle.intersects_any(selected_actor, colliding);
    add_collide_time(tf::tock());
    for (auto actor : geometry_handle.get_actors()) {
      if (actor == selected_actor)
        continue;
      if (colliding.find(actor) == colliding.end()) {
        reset_active_color(actor);
      } else {
        set_colliding_color(actor);
      }
    }
  }

public:
  auto reset_active_color(vtkActor *selected_actor) -> void {
    if (!selected_actor)
      return;
    selected_actor->GetProperty()->SetColor(
        normal_mesh_color[0], normal_mesh_color[1], normal_mesh_color[2]);
  }

  auto set_active_color(vtkActor *selected_actor) -> void {
    selected_actor->GetProperty()->SetColor(
        selected_mesh_color[0], selected_mesh_color[1], selected_mesh_color[2]);
  }

  auto reset_colliding_colors() -> void {
    colliding.clear();
    for (auto actor : geometry_handle.get_actors()) {
      if (actor != selected_actor)
        reset_active_color(actor);
    }
  }

  auto set_colliding_color(vtkActor *selected_actor) -> void {
    selected_actor->GetProperty()->SetColor(
        coliding_mesh_color[0], coliding_mesh_color[1], coliding_mesh_color[2]);
  }

  auto set_text_actors(vtkTextActor *_pick_text, vtkTextActor *_collide_text)
      -> void {
    pick_text = _pick_text;
    collide_text = _collide_text;
  }

  auto push_back(vtkActor *actor, vtkPolyData *poly,
                 tf::tree<int, float, 3> &tree) -> void {
    geometry_handle.push_back(actor, poly, tree);
  }

  auto OnLeftButtonDown() -> void override {
    if (selected_actor) {
      selected_mode = true;
      this->Interactor->GetRenderWindow()->HideCursor();
      this->Interactor->Render();
    } else {
      camera_mode = true;
      vtkInteractorStyleTrackballCamera::OnLeftButtonDown();
    }
  }

  auto OnLeftButtonUp() -> void override {
    if (selected_mode) {
      selected_mode = false;
      reset_colliding_colors();
      this->Interactor->GetRenderWindow()->ShowCursor();
      this->Interactor->Render();
    } else if (camera_mode) {
      camera_mode = false;
      vtkInteractorStyleTrackballCamera::OnLeftButtonUp();
    }
  }

  auto OnMouseMove() -> void override {
    auto [ray, renderer] = get_ray_and_renderer();
    if (!selected_mode && !camera_mode) {
      tf::tick();
      auto [actor, point] = geometry_handle.ray_hit(ray);
      add_pick_time(tf::tock());
      if (actor) {
        make_moving_plane(point, renderer);
        if (selected_actor != actor) {
          reset_active_color(selected_actor);
          set_active_color(actor);
        }
        last_point = point;
      } else {
        reset_active_color(selected_actor);
      }
      selected_actor = actor;
      this->Interactor->Render();
    } else if (selected_mode) {
      auto next_point = tf::ray_hit(ray, moving_plane).point;
      dx = next_point - last_point;
      last_point = next_point;
      move_selected(selected_actor);
      handle_collisions();
      this->Interactor->Render();
    } else if (camera_mode) {
      vtkInteractorStyleTrackballCamera::OnMouseMove();
      this->Interactor->Render();
    }
  }

  static cursor_interactor *New();
  vtkTypeMacro(cursor_interactor, vtkInteractorStyleTrackballCamera)
};

vtkStandardNewMacro(cursor_interactor);

auto set_at(vtkMatrix4x4 *mat, tf::vector<float, 3> at) -> void {
  auto tr = tf::random_transformation(at);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 4; ++j)
      mat->Element[i][j] = tr(i, j);
  mat->Modified();
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: program <input1.stl> <input2.stl> ...\n";
    return 1;
  } else if (argc == 2 && (std::string_view(argv[1]) == "-h" ||
                           std::string_view(argv[1]) == "--help")) {
    std::cerr << "Usage: program <input1.stl> <input2.stl> ...\n";
    return 1;
  }

  std::vector<decltype(readSTL(argv[1]))> polys;
  std::vector<tf::tree<int, float, 3>> trees;

  for (int i = 1; i < argc; ++i) {
    std::filesystem::path path{argv[i]};

    if (!path.has_extension() || path.extension() != ".stl") {
      std::cerr << "Skipping file " << path.filename()
                << ": not an .stl file\n";
      continue;
    }

    std::cout << "Reading file: " << path.filename() << std::endl;
    auto poly = readSTL(argv[i]);
    center_and_scale(poly.get());
    polys.emplace_back(std::move(poly));
    trees.emplace_back();
    trees.back().build(get_triangles(polys.back().get()),
                       tf::config_tree(4, 4));
  }

  if (polys.empty()) {
    std::cerr << "No valid .stl files provided.\n";
    return 1;
  }

  auto inter = vtk_make_unique<cursor_interactor>();
  auto renderer = vtk_make_unique<vtkRenderer>();
  auto render_window = vtk_make_unique<vtkRenderWindow>();
  auto interactor = vtk_make_unique<vtkRenderWindowInteractor>();
  interactor->SetInteractorStyle(inter.get());
  int n_actors_in_dim = 5;
  std::size_t poly_index = 0;
  std::size_t total_polygons = 0;
  for (int i = 0; i < n_actors_in_dim; ++i) {
    for (int j = 0; j < n_actors_in_dim; ++j) {
      auto &poly = polys[poly_index];
      total_polygons += poly->GetNumberOfPolys();
      auto mapper = vtk_make_unique<vtkOpenGLPolyDataMapper>();
      auto actor = vtk_make_unique<vtkOpenGLActor>();
      actor->SetMapper(mapper.get());
      mapper->SetInputData(poly.get());
      auto matrix = vtk_make_unique<vtkMatrix4x4>();
      matrix->Identity();
      set_at(matrix.get(), {i * 15.f, j * 15.f, 0.f});
      actor->SetUserMatrix(matrix.get());
      inter->push_back(actor.get(), poly.get(), trees[poly_index]);
      renderer->AddActor(actor.get());
      inter->reset_active_color(actor.get());
      poly_index = (poly_index + 1) % polys.size();
    }
  }

  renderer->SetBackground(27. / 255, 43. / 255, 52. / 255);
  render_window->SetInteractor(interactor.get());
  render_window->AddRenderer(renderer.get());
  render_window->Render();

  renderer->SetViewport(0.0, 0.12, 1.0, 1.0); // top 80%
  auto renderer_text = vtk_make_unique<vtkRenderer>();
  renderer_text->SetViewport(0.0, 0.0, 1.0, 0.12); // bottom 20%
  renderer_text->InteractiveOff();

  auto text0 = vtk_make_unique<vtkTextActor>();
  text0->SetInput(
      ("Total polygons in scene : " + std::to_string(total_polygons)).c_str());
  auto textprop0 = text0->GetTextProperty();
  textprop0->SetFontSize(40);
  textprop0->SetColor(1.0, 1.0, 1.0);
  textprop0->SetJustificationToLeft();
  textprop0->SetVerticalJustificationToCentered();
  text0->SetDisplayPosition(40, 55);
  renderer_text->AddActor2D(text0.get());

  auto text1 = vtk_make_unique<vtkTextActor>();
  text1->SetInput("Picking time per frame  : 0 mcs");
  auto textprop1 = text1->GetTextProperty();
  textprop1->SetFontSize(40);
  textprop1->SetColor(1.0, 1.0, 1.0);
  textprop1->SetJustificationToLeft();
  textprop1->SetVerticalJustificationToCentered();
  text1->SetDisplayPosition(40, 115);
  renderer_text->AddActor2D(text1.get());

  auto text2 = vtk_make_unique<vtkTextActor>();
  text2->SetInput("Collision time per frame: 0 mcs");
  auto textprop2 = text2->GetTextProperty();
  textprop2->SetFontSize(40);
  textprop2->SetColor(1.0, 1.0, 1.0);
  textprop2->SetJustificationToLeft();
  textprop2->SetVerticalJustificationToCentered();
  text2->SetDisplayPosition(40, 175);
  renderer_text->AddActor2D(text2.get());

  auto text3 = vtk_make_unique<vtkTextActor>();
  text3->SetInput("Grab and drag a mesh to test.\n"
                  "Intersecting meshes are highlighted.\n"
                  "Powered by trueform.");
  auto textprop3 = text3->GetTextProperty();
  textprop3->SetFontSize(40);
  textprop3->SetColor(1.0, 1.0, 1.0);
  textprop3->SetJustificationToRight();
  textprop3->SetVerticalJustificationToCentered();
  textprop3->SetLineSpacing(1.5);
  text3->SetDisplayPosition(renderer->GetSize()[0] - 40, 120);
  auto aligner = vtk_make_unique<RightAlignTextUpdater>(render_window.get(),
                                                        text3.get(), 40, 120);

  render_window->AddObserver(vtkCommand::WindowResizeEvent, aligner.get());

  renderer_text->AddActor2D(text3.get());

  inter->set_text_actors(text1.get(), text2.get());

  renderer_text->SetBackground(0.090, 0.143, 0.173); // darker tone
  render_window->AddRenderer(renderer_text.get());

  render_window->Render();
  interactor->Start();

  return 0;
}
