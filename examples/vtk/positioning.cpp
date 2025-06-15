#include "./util/common.hpp"
#include "./util/data_bridge.hpp"
#include "trueform/form.hpp"
#include "trueform/frame.hpp"
#include "trueform/intersects.hpp"
#include "trueform/neighbor_search.hpp"
#include "trueform/plane.hpp"
#include "trueform/random_transformation.hpp"
#include "trueform/ray_config.hpp"
#include "trueform/ray_hit.hpp"
#include "trueform/tick_tock.hpp"
#include "trueform/transformation.hpp"
#include "trueform/tree.hpp"
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

  auto closest_metric_point_pair(vtkActor *actor0) {
    std::size_t id0 = map[actor0];
    std::size_t id1 = !id0;
    auto form0 =
        tf::make_form(frames[id0], *trees[id0], get_triangles(polys[id0]));
    auto form1 =
        tf::make_form(frames[id1], *trees[id1], get_triangles(polys[id1]));

    auto result0 = tf::neighbor_search(form0, form1);
    return result0;
  }

  auto intersects_other(vtkActor *actor0) {
    std::size_t id0 = map[actor0];
    std::size_t id1 = !id0;
    auto form0 =
        tf::make_form(frames[id0], *trees[id0], get_triangles(polys[id0]));
    auto form1 =
        tf::make_form(frames[id1], *trees[id1], get_triangles(polys[id1]));
    return tf::intersects(form0, form1);
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
  tf::point<double, 3> normal_mesh_color{{0.8, 0.8, 0.8}};
  tf::point<double, 3> selected_mesh_color{{1, 0.9, 1}};

  tf::plane<float, 3> moving_plane;
  tf::point<float, 3> last_point;
  tf::vector<float, 3> dx;
  vtkActor *selected_actor = nullptr;
  bool selected_mode = false;
  bool camera_mode = false;
  int last_mouse_x = 0;
  int last_mouse_y = 0;

  void dolly_to_position(double factor) {
    // 1) Get mouse display position and the renderer under it
    auto *rwi = this->Interactor;
    int *event_pos = rwi->GetEventPosition();
    int x = event_pos[0], y = event_pos[1];
    auto *renderer = rwi->FindPokedRenderer(x, y);
    if (!renderer)
      return;

    // 2) Access the camera
    auto *camera = renderer->GetActiveCamera();
    if (!camera)
      return;

    // 3) Project current focal point to display to fetch its depth
    double fp_world[4];
    camera->GetFocalPoint(fp_world);
    fp_world[3] = 1.;
    renderer->SetWorldPoint(fp_world);
    renderer->WorldToDisplay();
    double fp_disp[3];
    renderer->GetDisplayPoint(fp_disp);
    double depth = fp_disp[2];

    // 4) Unproject the cursor (x,y,depth) back into world space → pivot point
    renderer->SetDisplayPoint(x, y, depth);
    renderer->DisplayToWorld();
    double pivot[4];
    renderer->GetWorldPoint(pivot);
    if (pivot[3] == 0.)
      return;
    for (int i = 0; i < 3; ++i)
      pivot[i] /= pivot[3];

    // 5) Grab the old camera position & focal point
    double old_pos[3], old_fp[3];
    camera->GetPosition(old_pos);
    camera->GetFocalPoint(old_fp);

    // 6) Compute the vector from pivot→camera
    double v[3];
    for (int i = 0; i < 3; ++i)
      v[i] = old_pos[i] - pivot[i];

    // 7) Scale that vector by 1/factor to move camera closer/further
    double new_pos[3];
    for (int i = 0; i < 3; ++i)
      new_pos[i] = pivot[i] + v[i] / factor;

    // 8) Compute how much we shifted the camera by
    double delta[3];
    for (int i = 0; i < 3; ++i)
      delta[i] = new_pos[i] - old_pos[i];

    // 9) Move the focal point by the *same* shift so view direction stays the
    // same
    double new_fp[3];
    for (int i = 0; i < 3; ++i)
      new_fp[i] = old_fp[i] + delta[i];

    // 10) Apply, reset clipping, and redraw
    camera->SetPosition(new_pos);
    camera->SetFocalPoint(new_fp);
    renderer->ResetCameraClippingRange();
    rwi->Render();
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

  auto move_selected(vtkActor *selected_actor, tf::point<float, 3> pt) {
    for (int i = 0; i < 3; ++i)
      selected_actor->GetUserMatrix()->Element[i][3] += pt[i];
    selected_actor->GetUserMatrix()->Modified();
    geometry_handle.update_frame(selected_actor);
  }

  auto position_them() {
    if (geometry_handle.intersects_other(selected_actor)) {
      return;
    }
    auto neighbors = geometry_handle.closest_metric_point_pair(selected_actor);
    auto pt0 = neighbors.info.first;
    auto pt1 = neighbors.info.second;
    auto ray = tf::make_ray_between_points(pt0, pt1);
    auto eps = 0.1f;
    float t = 0;
    auto prev_pt = ray.origin;
    tf::tick();
    while (t < 1) {
      t = std::min(1.f, 1.75f * tf::tock() / 1000);
      auto t_use = std::min(t, 1.f - eps);
      auto s_t = 3 * t_use * t_use - 2 * t_use * t_use * t_use;
      auto pt = ray.origin + s_t * ray.direction;
      auto color = selected_mesh_color +
                   t_use * (normal_mesh_color - selected_mesh_color);
      move_selected(selected_actor, pt - prev_pt);
      selected_actor->GetProperty()->SetColor(color.data());
      selected_actor->Modified();
      prev_pt = pt;
      this->Interactor->Render();
    }
  }

public:
  // Override the mouse wheel events
  void OnMouseWheelForward() override {
    // A positive factor zooms in
    dolly_to_position(1.1);
  }

  void OnMouseWheelBackward() override {
    // A negative factor zooms out
    dolly_to_position(0.9);
  }

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

      last_mouse_x = this->Interactor->GetEventPosition()[0];
      last_mouse_y = this->Interactor->GetEventPosition()[1];
      /*vtkInteractorStyleTrackballCamera::OnLeftButtonDown();*/
    }
  }

  auto OnLeftButtonUp() -> void override {
    if (selected_mode) {
      selected_mode = false;
      position_them();
      reset_active_color(selected_actor);
      this->Interactor->GetRenderWindow()->ShowCursor();
      this->Interactor->Render();
    } else if (camera_mode) {
      camera_mode = false;
      /*vtkInteractorStyleTrackballCamera::OnLeftButtonUp();*/
    }
  }

  auto OnMouseMove() -> void override {

    auto [ray, renderer] = get_ray_and_renderer();
    if (!selected_mode && !camera_mode) {
      tf::tick();
      auto [actor, point] = geometry_handle.ray_hit(ray);
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
      this->Interactor->Render();
    } else if (camera_mode) {

      int *currPos = this->Interactor->GetEventPosition();
      int dx = currPos[0] - last_mouse_x;
      int dy = currPos[1] - last_mouse_y;

      last_mouse_x = currPos[0];
      last_mouse_y = currPos[1];

      auto *renderer =
          this->Interactor->FindPokedRenderer(currPos[0], currPos[1]);
      if (renderer) {
        auto *cam = renderer->GetActiveCamera();
        double SPEED = 5 / (cam->GetDistance());
        cam->Azimuth(-dx * SPEED);   // yaw
        cam->Elevation(-dy * SPEED); // pitch
        cam->OrthogonalizeViewUp();  // keep “up” consistent
        renderer->ResetCameraClippingRange();
        this->Interactor->Render();
      }
    }
  }

  static cursor_interactor *New();
  vtkTypeMacro(cursor_interactor, vtkInteractorStyleTrackballCamera)
};

vtkStandardNewMacro(cursor_interactor);

auto center_and_scale(vtkPolyData *poly) -> void {
  auto pts = get_points(poly);
  auto aabb = tf::aabb_from(tf::make_polygon(pts));
  auto center = aabb.center().as_vector();
  auto r = aabb.diagonal().length() / 2;
  for (auto pt : pts) {
    pt -= center;
    auto v = pt.as_vector_view();
    v *= 10 / r;
    pt = tf::make_point(v);
  }
}

auto set_at(vtkMatrix4x4 *mat, tf::vector<float, 3> at) -> void {
  auto tr = tf::random_transformation(at);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 4; ++j)
      mat->Element[i][j] = tr(i, j);
  mat->Modified();
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: program <input1.stl> <input2.stl>\n";
    return 1;
  } else if (argc == 2 && (std::string_view(argv[1]) == "-h" ||
                           std::string_view(argv[1]) == "--help")) {
    std::cerr << "Usage: program <input1.stl> <input2.stl> ...\n";
    return 1;
  }

  std::vector<decltype(readSTL(argv[1]))> polys;
  std::vector<tf::tree<int, float, 3>> trees;

  for (int i = 1; i < std::min(3, argc); ++i) {
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
  std::size_t poly_index = 0;
  for (int i = 0; i < 2; ++i) {
    auto &poly = polys[poly_index];
    auto mapper = vtk_make_unique<vtkOpenGLPolyDataMapper>();
    auto actor = vtk_make_unique<vtkOpenGLActor>();
    actor->SetMapper(mapper.get());
    mapper->SetInputData(poly.get());
    auto matrix = vtk_make_unique<vtkMatrix4x4>();
    matrix->Identity();
    set_at(matrix.get(), {{i * 15.f, (i + 2) * 15.f, 0.f}});
    actor->SetUserMatrix(matrix.get());
    inter->push_back(actor.get(), poly.get(), trees[poly_index]);
    renderer->AddActor(actor.get());
    inter->reset_active_color(actor.get());
    poly_index = (poly_index + 1) % polys.size();
  }

  renderer->SetBackground(27. / 255, 43. / 255, 52. / 255);
  render_window->SetInteractor(interactor.get());
  render_window->AddRenderer(renderer.get());
  render_window->Render();

  /*renderer->SetViewport(0.0, 0.12, 1.0, 1.0); // top 80%*/
  auto renderer_text = vtk_make_unique<vtkRenderer>();
  renderer_text->SetViewport(0, 0.0, 1.0, 0.12); // bottom 20%
  renderer_text->InteractiveOff();

  auto text3 = vtk_make_unique<vtkTextActor>();
  text3->SetInput("Drag one mesh away from the other.\n"
                  "Their nearest neighbors will be\n"
                  "brought back together on release.");
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

  renderer_text->SetBackground(0.090, 0.143, 0.173); // darker tone
  render_window->AddRenderer(renderer_text.get());

  render_window->Render();
  interactor->Start();

  return 0;
}

