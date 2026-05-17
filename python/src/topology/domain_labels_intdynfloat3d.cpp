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
#include <trueform/python/topology/domain_labels_impl.hpp>

namespace tf::py {

auto register_topology_domain_labels_intdynfloat3d(nanobind::module_ &m)
    -> void {
  using namespace nanobind;
  m.def("domain_labels_intdynfloat3d",
        [](const offset_blocked_array_wrapper<int, int> &indices,
           ndarray<numpy, const float, shape<-1, 3>> points, int config) {
          return impl::domain_labels_impl_dynamic<int, float, 3>(indices, points,
                                                                 config);
        },
        arg("indices"), arg("points"), arg("config"));
}

} // namespace tf::py
