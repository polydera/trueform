#include "./util/common.hpp"
#include "./util/data_bridge.hpp"
#include "trueform/random.hpp"
#include "trueform/trueform.hpp"
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
#include "vtkTubeFilter.h"
#include <filesystem>
#include <string>
#include <string_view>

class tf_bridge {
public:
  auto push_back(vtkActor *actor, vtkPolyData *poly) -> void {
    map[actor] = actors.size();
    actors.push_back(actor);
    polys.push_back(poly);
    frames.emplace_back();
    frames.back().fill(actor->GetUserMatrix()->GetData());
    trees.emplace_back();
    trees.back().build(get_triangles(poly), tf::config_tree(4, 4));
    face_memberships.emplace_back();
    face_memberships.back().build(get_triangles(poly));
    manifold_edge_links.emplace_back();
    manifold_edge_links.back().build(get_triangles(poly).faces(),
                                     face_memberships.back());
  }

  auto get_actors() const -> const auto & { return actors; }

  auto ray_hit(tf::ray<float, 3> ray)
      -> std::pair<vtkActor *, tf::point<float, 3>> {
    tf::tree_ray_info<int, tf::ray_cast_info<float>> result;
    tf::ray_config<float> config{};
    vtkActor *picked = nullptr;

    for (std::size_t i = 0; i < polys.size(); ++i) {
      auto form = tf::make_form(frames[i], trees[i], get_triangles(polys[i]));
      auto res = tf::ray_cast(ray, form, config);
      if (res) {
        result = res;
        config.max_t = result.info.t;
        picked = actors[i];
      }
    }
    return std::make_pair(picked, ray.origin + result.info.t * ray.direction);
  }

  auto update_frame(vtkActor *actor) -> void {
    auto id = map[actor];
    frames[id].fill(actors[id]->GetUserMatrix()->GetData());
  }

  auto compute_intersection_curves()  {
    auto form0 =
        tf::make_form(frames[0], trees[0],
                      tf::make_polygons(get_triangle_faces(polys[0]),
                                        get_points(polys[0]).as<double>())) //
        | tf::tag(face_memberships[0])                                      //
        | tf::tag(manifold_edge_links[0]);

    auto form1 =
        tf::make_form(frames[1], trees[1],
                      tf::make_polygons(get_triangle_faces(polys[1]),
                                        get_points(polys[1]).as<double>())) //
        | tf::tag(face_memberships[1])                                      //
        | tf::tag(manifold_edge_links[1]);
    return tf::make_intersection_curves(form0, form1);
  }

  auto get_actors() -> std::vector<vtkActor *> & { return actors; }

private:
  std::map<vtkActor *, int> map;
  std::vector<vtkPolyData *> polys;
  std::vector<vtkActor *> actors;
  std::vector<tf::frame<double, 3>> frames;
  std::vector<tf::tree<int, float, 3>> trees;
  std::vector<tf::face_membership<int>> face_memberships;
  std::vector<tf::manifold_edge_link<int, 3>> manifold_edge_links;
};

class cursor_interactor : public vtkInteractorStyleTrackballCamera {
private:
  tf_bridge bridge;
  std::vector<float> times;
  int time_index = 0;
  vtkTextActor *text;
  vtkPolyData *curve_poly;

  tf::plane<float, 3> moving_plane;
  tf::point<float, 3> last_point;
  tf::vector<float, 3> dx;
  vtkActor *selected_actor = nullptr;
  bool selected_mode = false;
  bool camera_mode = false;

  auto add_time(float t) {
    if (times.size() < 100) {
      times.push_back(t);
    } else {
      times[time_index] = t;
    }
    time_index = (time_index + 1) % 100;
    float sum = 0;
    for (auto time : times)
      sum += time;
    auto time = sum / times.size();
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer),
                  "Intersection curve time per scroll: %.1f ms", time);
    text->SetInput(buffer);
  }

  auto compute_curves() {
    tf::tick();
    auto curves = bridge.compute_intersection_curves();
    add_time(tf::tock());
    auto tmp_poly = curves_to_polydata(curves.curves());
    auto tubes = vtk_make_unique<vtkTubeFilter>();
    tubes->SetRadius(0.05);
    tubes->SetInputData(tmp_poly.get());
    tubes->Update();
    curve_poly->ShallowCopy(tubes->GetOutput());
    curve_poly->Modified();
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
    bridge.update_frame(selected_actor);
  }

  auto randomize_rotations() {
    for (vtkActor *actor : bridge.get_actors()) {
      vtkMatrix4x4 *mat = actor->GetUserMatrix();
      tf::vector<double, 3> at{mat->Element[0][3], mat->Element[1][3],
                               mat->Element[2][3]};
      auto tr = tf::random_transformation(at);
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 4; ++j)
          mat->Element[i][j] = tr(i, j);
      mat->Modified();
      actor->Modified();
      bridge.update_frame(actor);
    }
  }

public:
  auto push_back(vtkActor *actor, vtkPolyData *poly) -> void {
    bridge.push_back(actor, poly);
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
      auto [actor, point] = bridge.ray_hit(ray);
      if (actor) {
        make_moving_plane(point, renderer);
        last_point = point;
      }
      selected_actor = actor;
    } else if (selected_mode) {
      auto next_point = tf::ray_hit(ray, moving_plane).point;
      dx = next_point - last_point;
      last_point = next_point;
      move_selected(selected_actor);
      compute_curves();
      this->Interactor->Render();
    } else if (camera_mode) {
      vtkInteractorStyleTrackballCamera::OnMouseMove();
      this->Interactor->Render();
    }
  }

  auto initialize(vtkPolyData *_cpoly, vtkTextActor *_text) -> void {
    curve_poly = _cpoly;
    text = _text;
  }

  auto OnKeyPress() -> void override {
    std::string key = this->GetInteractor()->GetKeySym();
    if (key == "n") {
      randomize_rotations();
      compute_curves();
      this->Interactor->Render();
    } else {
      vtkInteractorStyleTrackballCamera::OnKeyPress();
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
    std::cerr << "Usage: program <input1.stl|obj> <input2.stl|obj> \n";
    return 1;
  } else if (argc == 2 && (std::string_view(argv[1]) == "-h" ||
                           std::string_view(argv[1]) == "--help")) {
    std::cerr << "Usage: program <input1.stl|obj> <input2.stl|obj> \n";
    return 1;
  }

  std::vector<decltype(read_mesh(argv[1]))> polys;

  for (int i = 1; i < argc; ++i) {
    std::filesystem::path path{argv[i]};

    std::cout << "Reading file: " << path.filename() << std::endl;
    auto poly = read_mesh(argv[i]);
    if (poly->GetPoints()) {
      center_and_scale(poly.get());
      polys.emplace_back(std::move(poly));
    }
    if (polys.size() == 2)
      break;
  }

  auto inter = vtk_make_unique<cursor_interactor>();
  auto renderer = vtk_make_unique<vtkRenderer>();
  auto render_window = vtk_make_unique<vtkRenderWindow>();
  auto interactor = vtk_make_unique<vtkRenderWindowInteractor>();
  interactor->SetInteractorStyle(inter.get());

  std::size_t total_polygons = 0;
  for (int i = 0; i < 2; ++i) {
    auto &poly = polys[i % int(polys.size())];
    total_polygons += poly->GetNumberOfPolys();
    auto mapper = vtk_make_unique<vtkOpenGLPolyDataMapper>();
    auto actor = vtk_make_unique<vtkOpenGLActor>();
    actor->SetMapper(mapper.get());
    mapper->SetInputData(poly.get());
    auto matrix = vtk_make_unique<vtkMatrix4x4>();
    matrix->Identity();
    set_at(matrix.get(), {i * 15.f, 0.f, 0.f});
    actor->SetUserMatrix(matrix.get());
    inter->push_back(actor.get(), poly.get());
    renderer->AddActor(actor.get());
  }

  auto curve_poly = vtk_make_unique<vtkPolyData>();
  curve_poly->Initialize();
  auto cmapper = vtk_make_unique<vtkOpenGLPolyDataMapper>();
  auto cactor = vtk_make_unique<vtkOpenGLActor>();
  cactor->SetMapper(cmapper.get());
  cmapper->SetInputData(curve_poly.get());
  cactor->GetProperty()->SetColor(1, 0.1, 0.1);
  renderer->AddActor(cactor.get());

  renderer->SetBackground(27. / 255, 43. / 255, 52. / 255);
  render_window->SetInteractor(interactor.get());
  render_window->AddRenderer(renderer.get());
  render_window->Render();

  renderer->SetViewport(0.0, 0.12, 1.0, 1.0); // top 80%
  auto renderer_text = vtk_make_unique<vtkRenderer>();
  renderer_text->SetBackground(0.090, 0.143, 0.173); // darker tone
  renderer_text->SetViewport(0.0, 0.0, 1.0, 0.12);   // bottom 20%
  renderer_text->InteractiveOff();
  render_window->AddRenderer(renderer_text.get());

  // text0
  auto text0 = vtk_make_unique<vtkTextActor>();
  text0->SetInput("Intersection curve time per scroll: 0 ms");
  auto textprop0 = text0->GetTextProperty();
  textprop0->SetFontSize(40);
  textprop0->SetColor(1.0, 1.0, 1.0);
  textprop0->SetJustificationToLeft();
  textprop0->SetVerticalJustificationToCentered();
  text0->GetPositionCoordinate()->SetCoordinateSystemToNormalizedViewport();
  text0->SetPosition(0.03, 0.30);
  renderer_text->AddActor2D(text0.get());

  // text1
  auto text1 = vtk_make_unique<vtkTextActor>();
  text1->SetInput("Press n to randomize the plane.");
  auto textprop1 = text1->GetTextProperty();
  textprop1->SetFontSize(40);
  textprop1->SetColor(1.0, 1.0, 1.0);
  textprop1->SetJustificationToLeft();
  textprop1->SetVerticalJustificationToCentered();
  text1->GetPositionCoordinate()->SetCoordinateSystemToNormalizedViewport();
  text1->SetPosition(0.03, 0.60);
  renderer_text->AddActor2D(text1.get());

  // text2
  auto text2 = vtk_make_unique<vtkTextActor>();
  text2->SetInput(
      ("Total polygons in scene : " + std::to_string(total_polygons)).c_str());
  auto textprop2 = text2->GetTextProperty();
  textprop2->SetFontSize(40);
  textprop2->SetColor(1.0, 1.0, 1.0);
  textprop2->SetJustificationToLeft();
  textprop2->SetVerticalJustificationToCentered();
  text2->GetPositionCoordinate()->SetCoordinateSystemToNormalizedViewport();
  text2->SetPosition(0.03, 0.88);
  renderer_text->AddActor2D(text2.get());

  // text3 (right-aligned, normalized)
  auto text3 = vtk_make_unique<vtkTextActor>();
  text3->SetInput("Grab and drag a mesh to test.\n"
                  "Intersection curves are shown.\n"
                  "Powered by trueform.");
  auto textprop3 = text3->GetTextProperty();
  textprop3->SetFontSize(40);
  textprop3->SetColor(1.0, 1.0, 1.0);
  textprop3->SetJustificationToRight();
  textprop3->SetVerticalJustificationToCentered();
  textprop3->SetLineSpacing(1.5);
  text3->GetPositionCoordinate()->SetCoordinateSystemToNormalizedViewport();
  text3->SetPosition(0.97, 0.60);
  renderer_text->AddActor2D(text3.get());

  inter->initialize(curve_poly.get(), text0.get());

  render_window->Render();
  interactor->Start();
  return 0;
}
