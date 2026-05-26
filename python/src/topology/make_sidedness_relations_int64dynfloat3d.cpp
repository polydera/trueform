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
#include "trueform/python/topology/make_sidedness_relations_impl.hpp"

namespace tf::py {

auto register_topology_make_sidedness_relations_int64dynfloat3d(
    nanobind::module_ &m) -> void {
  using namespace nanobind;
  m.def(
      "sidedness_relations_int64dynfloat3d",
      [](mesh_wrapper<std::int64_t, float, dynamic_size, 3> &mesh,
         ndarray<numpy, const std::int64_t, shape<-1>> tag_labels) {
        return impl::make_sidedness_relations_impl<std::int64_t, dynamic_size,
                                                   float, 3>(mesh, tag_labels);
      },
      arg("mesh"), arg("tag_labels"));
}

} // namespace tf::py
