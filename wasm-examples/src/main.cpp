#include <emscripten/bind.h>
#include <emscripten/val.h>
#include "boolean_web.h"
#include "collision_web.h"
#include "cross_section_web.h"
#include "isobands_web.h"
#include "main.h"
#include "positioning_web.h"
#include "shape_histogram_web.h"
#include "laplacian_smoothing_web.h"
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

auto &require_mesh_data_at(std::size_t index) {
  auto &mesh_store = require_interactor().get_mesh_data_store();
  if (index >= mesh_store.size()) {
    throw std::out_of_range("Mesh data index out of range");
  }
  return mesh_store[index];
}

auto &require_instance_at(std::size_t index) {
  auto &instances = require_interactor().get_instances();
  if (index >= instances.size()) {
    throw std::out_of_range("Instance index out of range");
  }
  return instances[index];
}

} // namespace

auto OnLeftButtonUpCustom(std::array<double, 3> focal_point,
                          emscripten::val lambda_set_focal, float dt) {
  auto &active = require_interactor();
  if (auto *positioning =
          dynamic_cast<cursor_interactor_positioning *>(&active)) {
    return positioning->OnLeftButtonUpCustom(focal_point, lambda_set_focal, dt);
  }
  return 2.0f;
}

auto OnLeftButtonUp() { return require_interactor().OnLeftButtonUp(); }

auto OnLeftButtonDown() { return require_interactor().OnLeftButtonDown(); }

auto OnMouseMove(std::array<float, 3> origin, std::array<float, 3> direction,
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

auto get_number_of_mesh_data() -> std::size_t {
  return require_interactor().get_mesh_data_store().size();
}

auto get_number_of_instances() -> std::size_t {
  return require_interactor().get_instances().size();
}

auto get_mesh_data_on_idx(int i) -> mesh_data * {
  if (i < 0) {
    throw std::out_of_range("Negative mesh data index");
  }
  return &require_mesh_data_at(static_cast<std::size_t>(i));
}

auto get_instance_on_idx(int i) -> instance * {
  if (i < 0) {
    throw std::out_of_range("Negative instance index");
  }
  return &require_instance_at(static_cast<std::size_t>(i));
}

auto get_result_mesh() -> result_mesh * { return &require_interactor().result; }

auto get_curve_mesh() -> result_mesh * { return &require_interactor().curves; }

auto get_average_time() { return require_interactor().m_time; }

auto get_average_pick_time() { return require_interactor().m_pick_time; }

auto get_number_of_polygons() -> std::size_t {
  return require_interactor().total_polygons;
}

// Shape histogram helpers
auto get_shape_histogram_interactor() -> cursor_interactor_shape_histogram * {
  return dynamic_cast<cursor_interactor_shape_histogram *>(&require_interactor());
}

auto shape_histogram_colors_updated() -> bool {
  auto *sh = get_shape_histogram_interactor();
  return sh ? sh->colors_updated() : false;
}

auto shape_histogram_get_vertex_colors() -> emscripten::val {
  auto *sh = get_shape_histogram_interactor();
  if (!sh)
    return emscripten::val::undefined();
  return sh->get_vertex_colors();
}

auto shape_histogram_get_histogram_bins() -> emscripten::val {
  auto *sh = get_shape_histogram_interactor();
  if (!sh)
    return emscripten::val::undefined();
  return sh->get_histogram_bins();
}

auto shape_histogram_set_radius(float r) -> void {
  auto *sh = get_shape_histogram_interactor();
  if (sh)
    sh->set_radius(r);
}

auto shape_histogram_get_aabb_diagonal() -> float {
  auto *sh = get_shape_histogram_interactor();
  return sh ? sh->get_aabb_diagonal() : 1.0f;
}

// Laplacian smoothing helpers
auto get_laplacian_smoothing_interactor()
    -> cursor_interactor_laplacian_smoothing * {
  return dynamic_cast<cursor_interactor_laplacian_smoothing *>(
      &require_interactor());
}

auto laplacian_smoothing_colors_updated() -> bool {
  auto *ls = get_laplacian_smoothing_interactor();
  return ls ? ls->colors_updated() : false;
}

auto laplacian_smoothing_points_updated() -> bool {
  auto *ls = get_laplacian_smoothing_interactor();
  return ls ? ls->points_updated() : false;
}

auto laplacian_smoothing_get_vertex_colors() -> emscripten::val {
  auto *ls = get_laplacian_smoothing_interactor();
  if (!ls)
    return emscripten::val::undefined();
  return ls->get_vertex_colors();
}

auto laplacian_smoothing_get_points() -> emscripten::val {
  auto *ls = get_laplacian_smoothing_interactor();
  if (!ls)
    return emscripten::val::undefined();
  return ls->get_points();
}

auto laplacian_smoothing_set_radius(float r) -> void {
  auto *ls = get_laplacian_smoothing_interactor();
  if (ls)
    ls->set_radius(r);
}

auto laplacian_smoothing_set_lambda(float l) -> void {
  auto *ls = get_laplacian_smoothing_interactor();
  if (ls)
    ls->set_lambda(l);
}

auto laplacian_smoothing_get_aabb_diagonal() -> float {
  auto *ls = get_laplacian_smoothing_interactor();
  return ls ? ls->get_aabb_diagonal() : 1.0f;
}

EMSCRIPTEN_BINDINGS(boolean) {
  emscripten::function("get_number_of_mesh_data", &get_number_of_mesh_data);
  emscripten::function("get_number_of_instances", &get_number_of_instances);
  emscripten::function("get_mesh_data_on_idx", &get_mesh_data_on_idx,
                       emscripten::allow_raw_pointers());
  emscripten::function("get_instance_on_idx", &get_instance_on_idx,
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
  // Isobands
  emscripten::function("run_main_isobands", &run_main_isobands);
  // Cross-section
  emscripten::function("run_main_cross_section", &run_main_cross_section);
  // Shape histogram
  emscripten::function("run_main_shape_histogram", &run_main_shape_histogram);
  emscripten::function("shape_histogram_colors_updated",
                       &shape_histogram_colors_updated);
  emscripten::function("shape_histogram_get_vertex_colors",
                       &shape_histogram_get_vertex_colors);
  emscripten::function("shape_histogram_get_histogram_bins",
                       &shape_histogram_get_histogram_bins);
  emscripten::function("shape_histogram_set_radius",
                       &shape_histogram_set_radius);
  emscripten::function("shape_histogram_get_aabb_diagonal",
                       &shape_histogram_get_aabb_diagonal);
  // Laplacian smoothing
  emscripten::function("run_main_laplacian_smoothing",
                       &run_main_laplacian_smoothing);
  emscripten::function("laplacian_smoothing_colors_updated",
                       &laplacian_smoothing_colors_updated);
  emscripten::function("laplacian_smoothing_points_updated",
                       &laplacian_smoothing_points_updated);
  emscripten::function("laplacian_smoothing_get_vertex_colors",
                       &laplacian_smoothing_get_vertex_colors);
  emscripten::function("laplacian_smoothing_get_points",
                       &laplacian_smoothing_get_points);
  emscripten::function("laplacian_smoothing_set_radius",
                       &laplacian_smoothing_set_radius);
  emscripten::function("laplacian_smoothing_set_lambda",
                       &laplacian_smoothing_set_lambda);
  emscripten::function("laplacian_smoothing_get_aabb_diagonal",
                       &laplacian_smoothing_get_aabb_diagonal);
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

EMSCRIPTEN_BINDINGS(mesh_data) {
  emscripten::class_<mesh_data>("mesh_data")
      .function("get_points", &mesh_data::get_points)
      .function("get_faces", &mesh_data::get_faces);
}

EMSCRIPTEN_BINDINGS(instance) {
  emscripten::class_<instance>("instance")
      .function("get_matrix", &instance::get_matrix)
      .function("set_color", &instance::set_color)
      .function("update_frame", &instance::update_frame)
      .property("mesh_data_id", &instance::mesh_data_id)
      .property("color", &instance::color)
      .property("matrix_updated", &instance::matrix_updated);
}

EMSCRIPTEN_BINDINGS(result_mesh) {
  emscripten::class_<result_mesh>("result_mesh")
      .function("get_points", &result_mesh::get_points)
      .function("get_faces", &result_mesh::get_faces)
      .function("get_curve_points", &result_mesh::get_curve_points)
      .function("get_curve_ids", &result_mesh::get_curve_ids)
      .function("get_curve_offsets", &result_mesh::get_curve_offsets)
      .property("updated", &result_mesh::updated);
}
