/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <trueform/python/core/edge_mesh.hpp>
#include <trueform/python/core/mesh.hpp>
#include <trueform/python/spatial/form_form_neighbor_search.hpp>

namespace tf::py {

auto register_mesh_neighbor_search_edge_mesh(nanobind::module_ &m) -> void {

  // ============================================================================
  // Mesh neighbor_search EdgeMesh
  // Mesh: 2 index types × 2 ngons
  // EdgeMesh: 2 index types
  // Real types: float, double (must match)
  // Dims: 2D, 3D (must match)
  // Total: 2 × 2 × 2 × 2 × 2 = 32 functions (all 4 index combos)
  // ============================================================================

  // ==== float, 2D ====

  // int32 mesh, int32 edge_mesh, triangle, float, 2D
  m.def("neighbor_search_mesh_edge_mesh_intintfloat32d",
        [](mesh_wrapper<int, float, 3, 2> &mesh,
           edge_mesh_wrapper<int, float, 2> &edge_mesh,
           std::optional<float> radius) {
          return form_form_neighbor_search(mesh, edge_mesh, radius);
        },
        nanobind::arg("mesh"),
        nanobind::arg("edge_mesh"),
        nanobind::arg("radius").none() = nanobind::none());

  // int32 mesh, int64 edge_mesh, triangle, float, 2D
  m.def("neighbor_search_mesh_edge_mesh_intint64float32d",
        [](mesh_wrapper<int, float, 3, 2> &mesh,
           edge_mesh_wrapper<int64_t, float, 2> &edge_mesh,
           std::optional<float> radius) {
          return form_form_neighbor_search(mesh, edge_mesh, radius);
        },
        nanobind::arg("mesh"),
        nanobind::arg("edge_mesh"),
        nanobind::arg("radius").none() = nanobind::none());

  // int64 mesh, int64 edge_mesh, triangle, float, 2D
  m.def("neighbor_search_mesh_edge_mesh_int64int64float32d",
        [](mesh_wrapper<int64_t, float, 3, 2> &mesh,
           edge_mesh_wrapper<int64_t, float, 2> &edge_mesh,
           std::optional<float> radius) {
          return form_form_neighbor_search(mesh, edge_mesh, radius);
        },
        nanobind::arg("mesh"),
        nanobind::arg("edge_mesh"),
        nanobind::arg("radius").none() = nanobind::none());

  // int32 mesh, int32 edge_mesh, quad, float, 2D
  m.def("neighbor_search_mesh_edge_mesh_intintfloat42d",
        [](mesh_wrapper<int, float, 4, 2> &mesh,
           edge_mesh_wrapper<int, float, 2> &edge_mesh,
           std::optional<float> radius) {
          return form_form_neighbor_search(mesh, edge_mesh, radius);
        },
        nanobind::arg("mesh"),
        nanobind::arg("edge_mesh"),
        nanobind::arg("radius").none() = nanobind::none());

  // int32 mesh, int64 edge_mesh, quad, float, 2D
  m.def("neighbor_search_mesh_edge_mesh_intint64float42d",
        [](mesh_wrapper<int, float, 4, 2> &mesh,
           edge_mesh_wrapper<int64_t, float, 2> &edge_mesh,
           std::optional<float> radius) {
          return form_form_neighbor_search(mesh, edge_mesh, radius);
        },
        nanobind::arg("mesh"),
        nanobind::arg("edge_mesh"),
        nanobind::arg("radius").none() = nanobind::none());

  // int64 mesh, int64 edge_mesh, quad, float, 2D
  m.def("neighbor_search_mesh_edge_mesh_int64int64float42d",
        [](mesh_wrapper<int64_t, float, 4, 2> &mesh,
           edge_mesh_wrapper<int64_t, float, 2> &edge_mesh,
           std::optional<float> radius) {
          return form_form_neighbor_search(mesh, edge_mesh, radius);
        },
        nanobind::arg("mesh"),
        nanobind::arg("edge_mesh"),
        nanobind::arg("radius").none() = nanobind::none());

  // ==== float, 3D ====

  // int32 mesh, int32 edge_mesh, triangle, float, 3D
  m.def("neighbor_search_mesh_edge_mesh_intintfloat33d",
        [](mesh_wrapper<int, float, 3, 3> &mesh,
           edge_mesh_wrapper<int, float, 3> &edge_mesh,
           std::optional<float> radius) {
          return form_form_neighbor_search(mesh, edge_mesh, radius);
        },
        nanobind::arg("mesh"),
        nanobind::arg("edge_mesh"),
        nanobind::arg("radius").none() = nanobind::none());

  // int32 mesh, int64 edge_mesh, triangle, float, 3D
  m.def("neighbor_search_mesh_edge_mesh_intint64float33d",
        [](mesh_wrapper<int, float, 3, 3> &mesh,
           edge_mesh_wrapper<int64_t, float, 3> &edge_mesh,
           std::optional<float> radius) {
          return form_form_neighbor_search(mesh, edge_mesh, radius);
        },
        nanobind::arg("mesh"),
        nanobind::arg("edge_mesh"),
        nanobind::arg("radius").none() = nanobind::none());

  // int64 mesh, int64 edge_mesh, triangle, float, 3D
  m.def("neighbor_search_mesh_edge_mesh_int64int64float33d",
        [](mesh_wrapper<int64_t, float, 3, 3> &mesh,
           edge_mesh_wrapper<int64_t, float, 3> &edge_mesh,
           std::optional<float> radius) {
          return form_form_neighbor_search(mesh, edge_mesh, radius);
        },
        nanobind::arg("mesh"),
        nanobind::arg("edge_mesh"),
        nanobind::arg("radius").none() = nanobind::none());

  // int32 mesh, int32 edge_mesh, quad, float, 3D
  m.def("neighbor_search_mesh_edge_mesh_intintfloat43d",
        [](mesh_wrapper<int, float, 4, 3> &mesh,
           edge_mesh_wrapper<int, float, 3> &edge_mesh,
           std::optional<float> radius) {
          return form_form_neighbor_search(mesh, edge_mesh, radius);
        },
        nanobind::arg("mesh"),
        nanobind::arg("edge_mesh"),
        nanobind::arg("radius").none() = nanobind::none());

  // int32 mesh, int64 edge_mesh, quad, float, 3D
  m.def("neighbor_search_mesh_edge_mesh_intint64float43d",
        [](mesh_wrapper<int, float, 4, 3> &mesh,
           edge_mesh_wrapper<int64_t, float, 3> &edge_mesh,
           std::optional<float> radius) {
          return form_form_neighbor_search(mesh, edge_mesh, radius);
        },
        nanobind::arg("mesh"),
        nanobind::arg("edge_mesh"),
        nanobind::arg("radius").none() = nanobind::none());

  // int64 mesh, int64 edge_mesh, quad, float, 3D
  m.def("neighbor_search_mesh_edge_mesh_int64int64float43d",
        [](mesh_wrapper<int64_t, float, 4, 3> &mesh,
           edge_mesh_wrapper<int64_t, float, 3> &edge_mesh,
           std::optional<float> radius) {
          return form_form_neighbor_search(mesh, edge_mesh, radius);
        },
        nanobind::arg("mesh"),
        nanobind::arg("edge_mesh"),
        nanobind::arg("radius").none() = nanobind::none());

  // ==== double, 2D ====

  // int32 mesh, int32 edge_mesh, triangle, double, 2D
  m.def("neighbor_search_mesh_edge_mesh_intintdouble32d",
        [](mesh_wrapper<int, double, 3, 2> &mesh,
           edge_mesh_wrapper<int, double, 2> &edge_mesh,
           std::optional<double> radius) {
          return form_form_neighbor_search(mesh, edge_mesh, radius);
        },
        nanobind::arg("mesh"),
        nanobind::arg("edge_mesh"),
        nanobind::arg("radius").none() = nanobind::none());

  // int32 mesh, int64 edge_mesh, triangle, double, 2D
  m.def("neighbor_search_mesh_edge_mesh_intint64double32d",
        [](mesh_wrapper<int, double, 3, 2> &mesh,
           edge_mesh_wrapper<int64_t, double, 2> &edge_mesh,
           std::optional<double> radius) {
          return form_form_neighbor_search(mesh, edge_mesh, radius);
        },
        nanobind::arg("mesh"),
        nanobind::arg("edge_mesh"),
        nanobind::arg("radius").none() = nanobind::none());

  // int64 mesh, int64 edge_mesh, triangle, double, 2D
  m.def("neighbor_search_mesh_edge_mesh_int64int64double32d",
        [](mesh_wrapper<int64_t, double, 3, 2> &mesh,
           edge_mesh_wrapper<int64_t, double, 2> &edge_mesh,
           std::optional<double> radius) {
          return form_form_neighbor_search(mesh, edge_mesh, radius);
        },
        nanobind::arg("mesh"),
        nanobind::arg("edge_mesh"),
        nanobind::arg("radius").none() = nanobind::none());

  // int32 mesh, int32 edge_mesh, quad, double, 2D
  m.def("neighbor_search_mesh_edge_mesh_intintdouble42d",
        [](mesh_wrapper<int, double, 4, 2> &mesh,
           edge_mesh_wrapper<int, double, 2> &edge_mesh,
           std::optional<double> radius) {
          return form_form_neighbor_search(mesh, edge_mesh, radius);
        },
        nanobind::arg("mesh"),
        nanobind::arg("edge_mesh"),
        nanobind::arg("radius").none() = nanobind::none());

  // int32 mesh, int64 edge_mesh, quad, double, 2D
  m.def("neighbor_search_mesh_edge_mesh_intint64double42d",
        [](mesh_wrapper<int, double, 4, 2> &mesh,
           edge_mesh_wrapper<int64_t, double, 2> &edge_mesh,
           std::optional<double> radius) {
          return form_form_neighbor_search(mesh, edge_mesh, radius);
        },
        nanobind::arg("mesh"),
        nanobind::arg("edge_mesh"),
        nanobind::arg("radius").none() = nanobind::none());

  // int64 mesh, int64 edge_mesh, quad, double, 2D
  m.def("neighbor_search_mesh_edge_mesh_int64int64double42d",
        [](mesh_wrapper<int64_t, double, 4, 2> &mesh,
           edge_mesh_wrapper<int64_t, double, 2> &edge_mesh,
           std::optional<double> radius) {
          return form_form_neighbor_search(mesh, edge_mesh, radius);
        },
        nanobind::arg("mesh"),
        nanobind::arg("edge_mesh"),
        nanobind::arg("radius").none() = nanobind::none());

  // ==== double, 3D ====

  // int32 mesh, int32 edge_mesh, triangle, double, 3D
  m.def("neighbor_search_mesh_edge_mesh_intintdouble33d",
        [](mesh_wrapper<int, double, 3, 3> &mesh,
           edge_mesh_wrapper<int, double, 3> &edge_mesh,
           std::optional<double> radius) {
          return form_form_neighbor_search(mesh, edge_mesh, radius);
        },
        nanobind::arg("mesh"),
        nanobind::arg("edge_mesh"),
        nanobind::arg("radius").none() = nanobind::none());

  // int32 mesh, int64 edge_mesh, triangle, double, 3D
  m.def("neighbor_search_mesh_edge_mesh_intint64double33d",
        [](mesh_wrapper<int, double, 3, 3> &mesh,
           edge_mesh_wrapper<int64_t, double, 3> &edge_mesh,
           std::optional<double> radius) {
          return form_form_neighbor_search(mesh, edge_mesh, radius);
        },
        nanobind::arg("mesh"),
        nanobind::arg("edge_mesh"),
        nanobind::arg("radius").none() = nanobind::none());

  // int64 mesh, int64 edge_mesh, triangle, double, 3D
  m.def("neighbor_search_mesh_edge_mesh_int64int64double33d",
        [](mesh_wrapper<int64_t, double, 3, 3> &mesh,
           edge_mesh_wrapper<int64_t, double, 3> &edge_mesh,
           std::optional<double> radius) {
          return form_form_neighbor_search(mesh, edge_mesh, radius);
        },
        nanobind::arg("mesh"),
        nanobind::arg("edge_mesh"),
        nanobind::arg("radius").none() = nanobind::none());

  // int32 mesh, int32 edge_mesh, quad, double, 3D
  m.def("neighbor_search_mesh_edge_mesh_intintdouble43d",
        [](mesh_wrapper<int, double, 4, 3> &mesh,
           edge_mesh_wrapper<int, double, 3> &edge_mesh,
           std::optional<double> radius) {
          return form_form_neighbor_search(mesh, edge_mesh, radius);
        },
        nanobind::arg("mesh"),
        nanobind::arg("edge_mesh"),
        nanobind::arg("radius").none() = nanobind::none());

  // int32 mesh, int64 edge_mesh, quad, double, 3D
  m.def("neighbor_search_mesh_edge_mesh_intint64double43d",
        [](mesh_wrapper<int, double, 4, 3> &mesh,
           edge_mesh_wrapper<int64_t, double, 3> &edge_mesh,
           std::optional<double> radius) {
          return form_form_neighbor_search(mesh, edge_mesh, radius);
        },
        nanobind::arg("mesh"),
        nanobind::arg("edge_mesh"),
        nanobind::arg("radius").none() = nanobind::none());

  // int64 mesh, int64 edge_mesh, quad, double, 3D
  m.def("neighbor_search_mesh_edge_mesh_int64int64double43d",
        [](mesh_wrapper<int64_t, double, 4, 3> &mesh,
           edge_mesh_wrapper<int64_t, double, 3> &edge_mesh,
           std::optional<double> radius) {
          return form_form_neighbor_search(mesh, edge_mesh, radius);
        },
        nanobind::arg("mesh"),
        nanobind::arg("edge_mesh"),
        nanobind::arg("radius").none() = nanobind::none());

  // int64 mesh, int32 edge_mesh, triangle, float, 2D
  m.def("neighbor_search_mesh_edge_mesh_int64intfloat32d",
        [](mesh_wrapper<int64_t, float, 3, 2> &mesh,
           edge_mesh_wrapper<int, float, 2> &edge_mesh,
           std::optional<float> radius) {
          return form_form_neighbor_search(mesh, edge_mesh, radius);
        },
        nanobind::arg("mesh"),
        nanobind::arg("edge_mesh"),
        nanobind::arg("radius").none() = nanobind::none());

  // int64 mesh, int32 edge_mesh, quad, float, 2D
  m.def("neighbor_search_mesh_edge_mesh_int64intfloat42d",
        [](mesh_wrapper<int64_t, float, 4, 2> &mesh,
           edge_mesh_wrapper<int, float, 2> &edge_mesh,
           std::optional<float> radius) {
          return form_form_neighbor_search(mesh, edge_mesh, radius);
        },
        nanobind::arg("mesh"),
        nanobind::arg("edge_mesh"),
        nanobind::arg("radius").none() = nanobind::none());

  // int64 mesh, int32 edge_mesh, triangle, float, 3D
  m.def("neighbor_search_mesh_edge_mesh_int64intfloat33d",
        [](mesh_wrapper<int64_t, float, 3, 3> &mesh,
           edge_mesh_wrapper<int, float, 3> &edge_mesh,
           std::optional<float> radius) {
          return form_form_neighbor_search(mesh, edge_mesh, radius);
        },
        nanobind::arg("mesh"),
        nanobind::arg("edge_mesh"),
        nanobind::arg("radius").none() = nanobind::none());

  // int64 mesh, int32 edge_mesh, quad, float, 3D
  m.def("neighbor_search_mesh_edge_mesh_int64intfloat43d",
        [](mesh_wrapper<int64_t, float, 4, 3> &mesh,
           edge_mesh_wrapper<int, float, 3> &edge_mesh,
           std::optional<float> radius) {
          return form_form_neighbor_search(mesh, edge_mesh, radius);
        },
        nanobind::arg("mesh"),
        nanobind::arg("edge_mesh"),
        nanobind::arg("radius").none() = nanobind::none());

  // int64 mesh, int32 edge_mesh, triangle, double, 2D
  m.def("neighbor_search_mesh_edge_mesh_int64intdouble32d",
        [](mesh_wrapper<int64_t, double, 3, 2> &mesh,
           edge_mesh_wrapper<int, double, 2> &edge_mesh,
           std::optional<double> radius) {
          return form_form_neighbor_search(mesh, edge_mesh, radius);
        },
        nanobind::arg("mesh"),
        nanobind::arg("edge_mesh"),
        nanobind::arg("radius").none() = nanobind::none());

  // int64 mesh, int32 edge_mesh, quad, double, 2D
  m.def("neighbor_search_mesh_edge_mesh_int64intdouble42d",
        [](mesh_wrapper<int64_t, double, 4, 2> &mesh,
           edge_mesh_wrapper<int, double, 2> &edge_mesh,
           std::optional<double> radius) {
          return form_form_neighbor_search(mesh, edge_mesh, radius);
        },
        nanobind::arg("mesh"),
        nanobind::arg("edge_mesh"),
        nanobind::arg("radius").none() = nanobind::none());

  // int64 mesh, int32 edge_mesh, triangle, double, 3D
  m.def("neighbor_search_mesh_edge_mesh_int64intdouble33d",
        [](mesh_wrapper<int64_t, double, 3, 3> &mesh,
           edge_mesh_wrapper<int, double, 3> &edge_mesh,
           std::optional<double> radius) {
          return form_form_neighbor_search(mesh, edge_mesh, radius);
        },
        nanobind::arg("mesh"),
        nanobind::arg("edge_mesh"),
        nanobind::arg("radius").none() = nanobind::none());

  // int64 mesh, int32 edge_mesh, quad, double, 3D
  m.def("neighbor_search_mesh_edge_mesh_int64intdouble43d",
        [](mesh_wrapper<int64_t, double, 4, 3> &mesh,
           edge_mesh_wrapper<int, double, 3> &edge_mesh,
           std::optional<double> radius) {
          return form_form_neighbor_search(mesh, edge_mesh, radius);
        },
        nanobind::arg("mesh"),
        nanobind::arg("edge_mesh"),
        nanobind::arg("radius").none() = nanobind::none());
}

} // namespace tf::py
