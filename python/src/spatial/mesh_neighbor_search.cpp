/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */

#include <nanobind/nanobind.h>

namespace tf::py {

// Forward declarations for mesh neighbor search bindings split across multiple
// files
auto register_mesh_neighbor_search_intfloat32d(nanobind::module_ &m) -> void;
auto register_mesh_neighbor_search_intfloat33d(nanobind::module_ &m) -> void;
auto register_mesh_neighbor_search_intfloat42d(nanobind::module_ &m) -> void;
auto register_mesh_neighbor_search_intfloat43d(nanobind::module_ &m) -> void;
auto register_mesh_neighbor_search_intdouble32d(nanobind::module_ &m) -> void;
auto register_mesh_neighbor_search_intdouble33d(nanobind::module_ &m) -> void;
auto register_mesh_neighbor_search_intdouble42d(nanobind::module_ &m) -> void;
auto register_mesh_neighbor_search_intdouble43d(nanobind::module_ &m) -> void;
auto register_mesh_neighbor_search_int64float32d(nanobind::module_ &m) -> void;
auto register_mesh_neighbor_search_int64float33d(nanobind::module_ &m) -> void;
auto register_mesh_neighbor_search_int64float42d(nanobind::module_ &m) -> void;
auto register_mesh_neighbor_search_int64float43d(nanobind::module_ &m) -> void;
auto register_mesh_neighbor_search_int64double32d(nanobind::module_ &m)
    -> void;
auto register_mesh_neighbor_search_int64double33d(nanobind::module_ &m)
    -> void;
auto register_mesh_neighbor_search_int64double42d(nanobind::module_ &m)
    -> void;
auto register_mesh_neighbor_search_int64double43d(nanobind::module_ &m)
    -> void;

auto register_mesh_neighbor_search(nanobind::module_ &m) -> void {
  // Register all mesh neighbor search bindings
  // Split across multiple files for parallel compilation
  register_mesh_neighbor_search_intfloat32d(m);
  register_mesh_neighbor_search_intfloat33d(m);
  register_mesh_neighbor_search_intfloat42d(m);
  register_mesh_neighbor_search_intfloat43d(m);
  register_mesh_neighbor_search_intdouble32d(m);
  register_mesh_neighbor_search_intdouble33d(m);
  register_mesh_neighbor_search_intdouble42d(m);
  register_mesh_neighbor_search_intdouble43d(m);
  register_mesh_neighbor_search_int64float32d(m);
  register_mesh_neighbor_search_int64float33d(m);
  register_mesh_neighbor_search_int64float42d(m);
  register_mesh_neighbor_search_int64float43d(m);
  register_mesh_neighbor_search_int64double32d(m);
  register_mesh_neighbor_search_int64double33d(m);
  register_mesh_neighbor_search_int64double42d(m);
  register_mesh_neighbor_search_int64double43d(m);
}

} // namespace tf::py
