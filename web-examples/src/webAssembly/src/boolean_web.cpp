#include "trueform/core/curves_buffer.hpp"
#include "trueform/io/read_stl.hpp"
#include "trueform/random.hpp"
#include "trueform/trueform.hpp"
#include "trueform/spatial/form.hpp"
#include "trueform/spatial/ray_cast.hpp"

#include "utils/bridge_web.cpp"

#include <filesystem>
#include <string>
#include <string_view>

#include <emscripten/bind.h>
#include <emscripten/val.h>


class tf_bridge {
public:
  auto push_back(std::unique_ptr<MeshObject> mesh) -> void {
    map[mesh.get()] = actors.size();
    polys.push_back(&mesh->polyObject);
    frames.emplace_back();
    frames.back().fill(mesh->matrix.data());
    trees.emplace_back();
    trees.back().build(mesh->polyObject.polygons(), tf::config_tree(4, 4));
    face_memberships.emplace_back();
    face_memberships.back().build(mesh->polyObject.polygons());
    manifold_edge_links.emplace_back();
    manifold_edge_links.back().build(mesh->polyObject.faces(),
                                     face_memberships.back());
    actors.push_back(std::move(mesh));
  }

  auto get_actors() const -> const auto & { return actors; }

  auto ray_hit(tf::ray<float, 3> ray)
      -> std::pair<MeshObject *, tf::point<float, 3>> {
    tf::tree_ray_info<int, tf::ray_cast_info<float>> result;
    tf::ray_config<float> config{};
    MeshObject *picked = nullptr;

    for (const auto &[frame, poly, actor, tree] : tf::zip(frames, polys, actors, trees)) {
      auto form = tf::make_form(frame, tree, poly->polygons());
      auto res = tf::ray_cast(ray, form, config);
      if (res) {
        result = res;
        config.max_t = result.info.t;
        picked = actor.get();
      }
    }
    return std::make_pair(picked, ray.origin + result.info.t * ray.direction);
  }

  auto update_frame(MeshObject *actor) -> void {
    auto id = map[actor];
    frames[id].fill(actors[id]->matrix.data());
    actors[id]->matrixUpdated = true;
  }

  auto compute_boolean() {
    auto form0 = tf::make_form(frames[0], trees[0], polys[0]->polygons()) //
                 | tf::tag(face_memberships[0])                         //
                 | tf::tag(manifold_edge_links[0]);

    auto form1 = tf::make_form(frames[1], trees[1],polys[1]->polygons()) //
                 | tf::tag(face_memberships[1])                         //
                 | tf::tag(manifold_edge_links[1]);
    return tf::make_boolean(form0, form1, tf::boolean_op::left_difference,
                            tf::return_curves);
  }

  auto get_actors() -> std::vector<std::unique_ptr<MeshObject>> & { return actors; }

private:
  std::map<MeshObject *, int> map;
  std::vector<tf::polygons_buffer<int, float, 3, 3> *> polys;
  std::vector<std::unique_ptr<MeshObject>> actors;
  std::vector<tf::frame<double, 3>> frames;
  std::vector<tf::tree<int, float, 3>> trees;
  std::vector<tf::face_membership<int>> face_memberships;
  std::vector<tf::manifold_edge_link<int, 3>> manifold_edge_links;
};

class cursor_interactor {
public:
  std::unique_ptr<MeshObject> result_mesh = std::make_unique<MeshObject>();
  std::unique_ptr<MeshObject> curve_mesh = std::make_unique<MeshObject>();
private:
  tf_bridge bridge;
  std::vector<float> times;
  int time_index = 0;
  // vtkTextActor *text;

  tf::plane<float, 3> moving_plane;
  tf::point<float, 3> last_point;
  tf::vector<float, 3> dx;
  MeshObject *selected_actor = nullptr;
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
    std::snprintf(buffer, sizeof(buffer), "Boolean time per frame: %.1f ms",
                  time);
    std::cout << buffer << std::endl;
    //text->SetInput(buffer);
  }

  auto compute_curves() {
    tf::tick();
    auto [res_mesh, labels, curves] = bridge.compute_boolean();
    add_time(tf::tock());
    (void)labels;

    result_mesh->setPolydata(std::move(res_mesh));
    curve_mesh->setCurvesObject(std::move(curves));
  }

  auto make_moving_plane(tf::point<float, 3> origin, std::array<float, 3> cameraPosition, std::array<float, 3> cameraFocalPoint) {
    auto normal = tf::make_unit_vector(tf::make_vector_view<3>(cameraFocalPoint.data()) - tf::make_vector_view<3>(cameraPosition.data()));
    moving_plane = tf::make_plane(normal, origin);
  }

  auto move_selected(MeshObject *selected_actor) {
    for (int i = 0; i < 3; ++i)
      selected_actor->matrix[i*4 + 3] += dx[i];
    bridge.update_frame(selected_actor);
  }

  auto randomize_rotations() {
    for (std::unique_ptr<MeshObject>& actor : bridge.get_actors()) {
      tf::vector<double, 3> at{actor->matrix[3], actor->matrix[7],
                               actor->matrix[11]};
      auto tr = tf::random_transformation(at);
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 4; ++j)
          actor->matrix[i*4 + j] = tr(i, j);
      bridge.update_frame(actor.get());
    }
  }

public:
  auto get_actors() -> std::vector<std::unique_ptr<MeshObject>> & {
    return bridge.get_actors();
  }

  auto push_back(std::unique_ptr<MeshObject> mesh) -> void {
    bridge.push_back(std::move(mesh));
  }

  auto OnLeftButtonDown() {
    if (selected_actor) {
      selected_mode = true;
      return true;
    } else {
      camera_mode = true;
    }
    return false;
  }

  auto OnLeftButtonUp() {
    if (selected_mode) {
      selected_mode = false;
      return true;
    } else if (camera_mode) {
      camera_mode = false;
    }
    return false;
  }

  auto OnMouseMove(std::array<float, 3> origin, std::array<float, 3> direction, std::array<float, 3> cameraPosition, std::array<float, 3> cameraFocalPoint) {
    tf::ray<float, 3> ray{origin, direction};
    if (!selected_mode && !camera_mode) {
      auto [actor, point] = bridge.ray_hit(ray);
      if (actor) {
        make_moving_plane(point, cameraPosition, cameraFocalPoint);
        last_point = point;
      }
      selected_actor = actor;
      return true;
    } else if (selected_mode) {
      auto next_point = tf::ray_hit(ray, moving_plane).point;
      dx = next_point - last_point;
      last_point = next_point;
      move_selected(selected_actor);
      compute_curves();
      return true;
    } else if (camera_mode) {
        return false;
    }
    return false;
  }

  auto OnKeyPress(std::string key) {
    std::cout << "OnKeyPressed key: " << key << std::endl;
    if (key == "n") {
      randomize_rotations();
      compute_curves();
      return true;
    } else {
      return false;
    }
  }
};


auto set_at(std::array<double, 16>& mat, tf::vector<float, 3> at) -> void {
  auto tr = tf::random_transformation(at);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 4; ++j)
      mat[i*4+j] = tr(i, j);
}

auto center_and_scale_p(tf::polygons_buffer<int, float, 3, 3>& poly) -> void {
  auto pts = poly.points_buffer();
  auto aabb = tf::aabb_from(tf::make_polygon(pts));
  auto center = aabb.center().as_vector();
  auto r = aabb.diagonal().length() / 2;
  tf::parallel_apply(pts.points().as_vector_view(), [&](auto pt) {
    pt -= center;
    pt *= 10 / r;
  });

}

std::unique_ptr<cursor_interactor> interactor{};

auto OnLeftButtonUp() {
    return interactor->OnLeftButtonUp();
}
auto OnLeftButtonDown() {
    return interactor->OnLeftButtonDown();
}
auto OnMouseMove(std::array<float, 3> origin, std::array<float, 3> direction, std::array<float, 3> cameraPosition, std::array<float, 3> cameraFocalPoint) {
    return interactor->OnMouseMove(origin, direction, cameraPosition, cameraFocalPoint);
}
auto OnKeyPress(std::string key) {
    return interactor->OnKeyPress(key);
}
auto GetMeshOnIdx(int i) -> MeshObject * {
    return interactor->get_actors()[i].get();
}
auto GetResultMesh() -> MeshObject * {
    return interactor->result_mesh.get();
}
auto GetCurveMesh() -> MeshObject * {
    return interactor->curve_mesh.get();
}

int run_main(std::string path) {

  std::cout << "Reading file: " << path << std::endl;
  auto poly = tf::read_stl<int>(path);
  std::cout << "run main 0: " << poly.size() << std::endl;
  auto poly2 = tf::read_stl<int>(path);
  if (!poly.size()) {
    std::cout << "Failed to read file" << std::endl;
    throw std::runtime_error("Failed to read file");
  }
  interactor = std::make_unique<cursor_interactor>();

  center_and_scale_p(poly);
  auto actor = std::make_unique<MeshObject>();
  actor->polyObject = std::move(poly);
  set_at(actor->matrix, {0 * 15.f, 0.f, 0.f});
  interactor->push_back(std::move(actor));

  center_and_scale_p(poly2);
  auto actor2 = std::make_unique<MeshObject>();
  actor2->polyObject = std::move(poly2);
  set_at(actor2->matrix, {1 * 15.f, 0.f, 0.f});
  interactor->push_back(std::move(actor2));


/*
  // Optional curve actor on left
  auto curve_poly = vtk_make_unique<vtkPolyData>();
  curve_poly->Initialize();
  auto cmapper = vtk_make_unique<vtkOpenGLPolyDataMapper>();
  auto cactor = vtk_make_unique<vtkOpenGLActor>();
  cactor->SetMapper(cmapper.get());
  cmapper->SetInputData(curve_poly.get());
  cactor->GetProperty()->SetColor(1, 0.1, 0.1);
  rendererL->AddActor(cactor.get());
*/
  return 0;
}


EMSCRIPTEN_BINDINGS(boolean) {
  emscripten::function("run_main", &run_main);
  emscripten::function("OnLeftButtonUp", &OnLeftButtonUp);
  emscripten::function("OnLeftButtonDown", &OnLeftButtonDown);
  emscripten::function("OnMouseMove", &OnMouseMove);
  emscripten::function("OnKeyPress", &OnKeyPress);
  emscripten::function("GetMeshOnIdx", &GetMeshOnIdx, emscripten::allow_raw_pointers());
  emscripten::function("GetResultMesh", &GetResultMesh, emscripten::allow_raw_pointers());
  emscripten::function("GetCurveMesh", &GetCurveMesh, emscripten::allow_raw_pointers());
}

EMSCRIPTEN_BINDINGS(ArrayFloat3) {
    emscripten::value_array<std::array<float, 3>>("ArrayFloat3")
        .element(emscripten::index<0>())
        .element(emscripten::index<1>())
        .element(emscripten::index<2>());
}
EMSCRIPTEN_BINDINGS(ArrayDouble16) {
    emscripten::value_array<std::array<double, 16>>("ArrayDouble16")
        .element(emscripten::index<0>())
        .element(emscripten::index<1>())
        .element(emscripten::index<2>())
        .element(emscripten::index<3>())
        .element(emscripten::index<4>())
        .element(emscripten::index<5>())
        .element(emscripten::index<6>())
        .element(emscripten::index<7>())
        .element(emscripten::index<8>())
        .element(emscripten::index<9>())
        .element(emscripten::index<10>())
        .element(emscripten::index<11>())
        .element(emscripten::index<12>())
        .element(emscripten::index<13>())
        .element(emscripten::index<14>())
        .element(emscripten::index<15>());
}


EMSCRIPTEN_BINDINGS(MeshObject) {
    emscripten::class_<MeshObject>("MeshObject")
        .smart_ptr<std::shared_ptr<MeshObject>>("MeshObject")
        .function("GetPoints", &MeshObject::GetPoints)
        .function("GetPolys", &MeshObject::GetPolys)
        .property("matrix", &MeshObject::matrix)
        .property("matrixUpdated", &MeshObject::matrixUpdated)
        .property("polydataUpdated", &MeshObject::polydataUpdated)
        // .function("GetLines", &PolyDataJSView::GetLinesEmscripten)
    ;
}