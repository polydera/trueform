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

#include "trueform/python/csg/outer_shell_impl.hpp"

namespace tf::py {

auto register_outer_shell_int643double3d(nanobind::module_ &m) -> void {
  m.def("outer_shell_int643double3d",
        [](mesh_wrapper<std::int64_t, double, 3, 3> &mesh, int domain_flags) {
          return outer_shell(mesh, domain_flags);
        },
        nanobind::arg("mesh"), nanobind::arg("domain_flags") = 0);

  m.def("outer_shell_int64dyndouble3d",
        [](mesh_wrapper<std::int64_t, double, dynamic_size, 3> &mesh,
           int domain_flags) {
          return outer_shell(mesh, domain_flags);
        },
        nanobind::arg("mesh"), nanobind::arg("domain_flags") = 0);
}

} // namespace tf::py
