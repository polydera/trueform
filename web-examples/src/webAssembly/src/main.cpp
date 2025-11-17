#include <emscripten/bind.h>
#include <emscripten/val.h>
#include "boolean_web.h"
#include "collision_web.h"
#include "forms_intersections_web.h"
#include "isobands_web.h"
#include "main.h"
#include "positioning_web.h"
#include "scalar_field_intersections_web.h"
#include "utils/bridge_web.h"
#include "utils/cursor_interactor_interface.h"

namespace {

constexpr char kInteractorError[] =
    "Interactor is not initialized. Call one of the run_main* entry points "
    "before invoking interaction methods.";

auto &require_interactor() {
  if (!interactor) {
    throw std::runtime_error(kInteractorError);
  }
  return *interactor;
}

auto *require_mesh_at(std::size_t index) {
  auto &actors = require_interactor().get_actors();
  if (index >= actors.size()) {
    throw std::out_of_range("Actor index out of range");
  }
  return actors[index].get();
}

} // namespace

auto OnLeftButtonUpCustom(std::array<double, 3> focal_point,
                          emscripten::val lambda_set_focal, float dt) {
  auto &active = require_interactor();
  if (auto *positioning =
          dynamic_cast<cursor_interactor_positioning *>(&active)) {
    return positioning->OnLeftButtonUpCustom(focal_point, lambda_set_focal,
                                            dt);
  }
  return 2.0f;
}

auto OnLeftButtonUp() { return require_interactor().OnLeftButtonUp(); }

auto OnLeftButtonDown() { return require_interactor().OnLeftButtonDown(); }

auto OnMouseMove(std::array<float, 3> origin,
                 std::array<float, 3> direction,
                 std::array<float, 3> camera_position,
                 std::array<float, 3> camera_focal_point) {
  return require_interactor().OnMouseMove(origin, direction, camera_position,
                                          camera_focal_point);
}

auto OnKeyPress(const std::string &key) {
  return require_interactor().OnKeyPress(key);
}

auto OnMouseWheel(int delta, bool shift_key) {
  return require_interactor().OnMouseWheel(delta, shift_key);
}

auto get_number_of_meshes() -> std::size_t {
  return require_interactor().get_actors().size();
}

auto get_mesh_on_idx(int i) -> mesh_object * {
  if (i < 0) {
    throw std::out_of_range("Negative actor index");
  }
  return require_mesh_at(static_cast<std::size_t>(i));
}

auto get_result_mesh() -> mesh_object * {
  return require_interactor().result_mesh.get();
}

auto get_curve_mesh() -> mesh_object * {
  return require_interactor().curve_mesh.get();
}

auto get_average_time() { return require_interactor().m_time; }

auto get_average_pick_time() { return require_interactor().m_pick_time; }

auto get_number_of_polygons() -> std::size_t {
  return require_interactor().total_polygons;
}

EMSCRIPTEN_BINDINGS(boolean) {
  emscripten::function("get_number_of_meshes", &get_number_of_meshes);
  emscripten::function("get_mesh_on_idx", &get_mesh_on_idx,
                       emscripten::allow_raw_pointers());
  emscripten::function("get_average_time", &get_average_time);
  emscripten::function("get_average_pick_time", &get_average_pick_time);
  emscripten::function("get_number_of_polygons", &get_number_of_polygons);
  // Interactor
  emscripten::function("OnLeftButtonUpCustom", &OnLeftButtonUpCustom);
  emscripten::function("OnLeftButtonUp", &OnLeftButtonUp);
  emscripten::function("OnLeftButtonDown", &OnLeftButtonDown);
  emscripten::function("OnMouseMove", &OnMouseMove);
  emscripten::function("OnMouseWheel", &OnMouseWheel);
  emscripten::function("OnKeyPress", &OnKeyPress);
  // Boolean
  emscripten::function("run_main", &run_main);
  emscripten::function("run_main_positioning", &run_main_positioning);
  emscripten::function("get_result_mesh", &get_result_mesh,
                       emscripten::allow_raw_pointers());
  emscripten::function("get_curve_mesh", &get_curve_mesh,
                       emscripten::allow_raw_pointers());
  // Collisions
  emscripten::function("run_main_collisions", &run_main_collisions);
  // Forms Intersections
  emscripten::function("run_main_forms_intersections",
                       &run_main_forms_intersections);
  // Isobands
  emscripten::function("run_main_isobands", &run_main_isobands);
  // Scalar field intersections
  emscripten::function("run_main_scalar_field_intersections",
                       &run_main_scalar_field_intersections);
}

EMSCRIPTEN_BINDINGS(VectorString) {
  emscripten::register_vector<std::string>("VectorString");
}

EMSCRIPTEN_BINDINGS(ArrayFloat3) {
  emscripten::value_array<std::array<float, 3>>("ArrayFloat3")
      .element(emscripten::index<0>())
      .element(emscripten::index<1>())
      .element(emscripten::index<2>());
}

EMSCRIPTEN_BINDINGS(ArrayDouble3) {
  emscripten::value_array<std::array<double, 3>>("ArrayDouble3")
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

EMSCRIPTEN_BINDINGS(mesh_object) {
  emscripten::class_<mesh_object>("mesh_object")
      .smart_ptr<std::shared_ptr<mesh_object>>("mesh_object")
      .function("get_points", &mesh_object::get_points)
      .function("get_polys", &mesh_object::get_polys)
      .function("get_curve_points", &mesh_object::get_curve_points)
      .function("get_curve_ids", &mesh_object::get_curve_ids)
      .function("get_curve_offsets", &mesh_object::get_curve_offsets)
      .function("get_matrix", &mesh_object::get_matrix)
      .property("color", &mesh_object::color)
      .property("matrix_updated", &mesh_object::matrix_updated)
      .property("polydata_updated", &mesh_object::polydata_updated);
}
