/*
* Copyright (c) 2025 XLAB
* All rights reserved.
*
* This file is part of trueform (trueform.polydera.com)
*
* Licensed for noncommercial use under the PolyForm Noncommercial
* License 1.0.0.
* Commercial licensing available via info@polydera.com.
*
* Author: Žiga Sajovic
*/
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <trueform/python/spatial/mesh.hpp>
#include <trueform/python/spatial/form_form_neighbor_search.hpp>

namespace tf::py {

auto register_mesh_neighbor_search_mesh_double2d(nanobind::module_ &m) -> void {

  // ==== double, 2D ====

  // int32 × int32
  m.def("neighbor_search_mesh_mesh_intint33double2d",
        [](mesh_wrapper<int, double, 3, 2> &mesh0,
           mesh_wrapper<int, double, 3, 2> &mesh1,
           std::optional<double> radius) {
          return form_form_neighbor_search(mesh0, mesh1, radius);
        },
        nanobind::arg("mesh0"),
        nanobind::arg("mesh1"),
        nanobind::arg("radius").none() = nanobind::none());

  m.def("neighbor_search_mesh_mesh_intint3dyndouble2d",
        [](mesh_wrapper<int, double, 3, 2> &mesh0,
           mesh_wrapper<int, double, tf::dynamic_size, 2> &mesh1,
           std::optional<double> radius) {
          return form_form_neighbor_search(mesh0, mesh1, radius);
        },
        nanobind::arg("mesh0"),
        nanobind::arg("mesh1"),
        nanobind::arg("radius").none() = nanobind::none());

  m.def("neighbor_search_mesh_mesh_intintdyn3double2d",
        [](mesh_wrapper<int, double, tf::dynamic_size, 2> &mesh0,
           mesh_wrapper<int, double, 3, 2> &mesh1,
           std::optional<double> radius) {
          return form_form_neighbor_search(mesh0, mesh1, radius);
        },
        nanobind::arg("mesh0"),
        nanobind::arg("mesh1"),
        nanobind::arg("radius").none() = nanobind::none());

  m.def("neighbor_search_mesh_mesh_intintdyndyndouble2d",
        [](mesh_wrapper<int, double, tf::dynamic_size, 2> &mesh0,
           mesh_wrapper<int, double, tf::dynamic_size, 2> &mesh1,
           std::optional<double> radius) {
          return form_form_neighbor_search(mesh0, mesh1, radius);
        },
        nanobind::arg("mesh0"),
        nanobind::arg("mesh1"),
        nanobind::arg("radius").none() = nanobind::none());

  // int32 × int64
  m.def("neighbor_search_mesh_mesh_intint6433double2d",
        [](mesh_wrapper<int, double, 3, 2> &mesh0,
           mesh_wrapper<int64_t, double, 3, 2> &mesh1,
           std::optional<double> radius) {
          return form_form_neighbor_search(mesh0, mesh1, radius);
        },
        nanobind::arg("mesh0"),
        nanobind::arg("mesh1"),
        nanobind::arg("radius").none() = nanobind::none());

  m.def("neighbor_search_mesh_mesh_intint643dyndouble2d",
        [](mesh_wrapper<int, double, 3, 2> &mesh0,
           mesh_wrapper<int64_t, double, tf::dynamic_size, 2> &mesh1,
           std::optional<double> radius) {
          return form_form_neighbor_search(mesh0, mesh1, radius);
        },
        nanobind::arg("mesh0"),
        nanobind::arg("mesh1"),
        nanobind::arg("radius").none() = nanobind::none());

  m.def("neighbor_search_mesh_mesh_intint64dyn3double2d",
        [](mesh_wrapper<int, double, tf::dynamic_size, 2> &mesh0,
           mesh_wrapper<int64_t, double, 3, 2> &mesh1,
           std::optional<double> radius) {
          return form_form_neighbor_search(mesh0, mesh1, radius);
        },
        nanobind::arg("mesh0"),
        nanobind::arg("mesh1"),
        nanobind::arg("radius").none() = nanobind::none());

  m.def("neighbor_search_mesh_mesh_intint64dyndyndouble2d",
        [](mesh_wrapper<int, double, tf::dynamic_size, 2> &mesh0,
           mesh_wrapper<int64_t, double, tf::dynamic_size, 2> &mesh1,
           std::optional<double> radius) {
          return form_form_neighbor_search(mesh0, mesh1, radius);
        },
        nanobind::arg("mesh0"),
        nanobind::arg("mesh1"),
        nanobind::arg("radius").none() = nanobind::none());

  // int64 × int64
  m.def("neighbor_search_mesh_mesh_int64int6433double2d",
        [](mesh_wrapper<int64_t, double, 3, 2> &mesh0,
           mesh_wrapper<int64_t, double, 3, 2> &mesh1,
           std::optional<double> radius) {
          return form_form_neighbor_search(mesh0, mesh1, radius);
        },
        nanobind::arg("mesh0"),
        nanobind::arg("mesh1"),
        nanobind::arg("radius").none() = nanobind::none());

  m.def("neighbor_search_mesh_mesh_int64int643dyndouble2d",
        [](mesh_wrapper<int64_t, double, 3, 2> &mesh0,
           mesh_wrapper<int64_t, double, tf::dynamic_size, 2> &mesh1,
           std::optional<double> radius) {
          return form_form_neighbor_search(mesh0, mesh1, radius);
        },
        nanobind::arg("mesh0"),
        nanobind::arg("mesh1"),
        nanobind::arg("radius").none() = nanobind::none());

  m.def("neighbor_search_mesh_mesh_int64int64dyn3double2d",
        [](mesh_wrapper<int64_t, double, tf::dynamic_size, 2> &mesh0,
           mesh_wrapper<int64_t, double, 3, 2> &mesh1,
           std::optional<double> radius) {
          return form_form_neighbor_search(mesh0, mesh1, radius);
        },
        nanobind::arg("mesh0"),
        nanobind::arg("mesh1"),
        nanobind::arg("radius").none() = nanobind::none());

  m.def("neighbor_search_mesh_mesh_int64int64dyndyndouble2d",
        [](mesh_wrapper<int64_t, double, tf::dynamic_size, 2> &mesh0,
           mesh_wrapper<int64_t, double, tf::dynamic_size, 2> &mesh1,
           std::optional<double> radius) {
          return form_form_neighbor_search(mesh0, mesh1, radius);
        },
        nanobind::arg("mesh0"),
        nanobind::arg("mesh1"),
        nanobind::arg("radius").none() = nanobind::none());

  // Note: int64 × int32 is handled by Python canonical ordering
}

} // namespace tf::py
