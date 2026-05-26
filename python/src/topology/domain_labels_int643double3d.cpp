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
#include "trueform/python/topology/domain_labels_impl.hpp"

namespace tf::py {

auto register_topology_domain_labels_int643double3d(nanobind::module_ &m)
    -> void {
  using namespace nanobind;
  m.def(
      "domain_labels_int643double3d",
      [](mesh_wrapper<std::int64_t, double, 3, 3> &mesh, int config) {
        return impl::domain_labels_impl<std::int64_t, 3, double, 3>(mesh,
                                                                    config);
      },
      arg("mesh"), arg("config"));
}

} // namespace tf::py
