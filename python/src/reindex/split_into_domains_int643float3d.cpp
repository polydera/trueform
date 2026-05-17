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
#include <trueform/python/reindex/split_into_domains_impl.hpp>

namespace tf::py {

auto register_reindex_split_into_domains_int643float3d(nanobind::module_ &m)
    -> void {
  using namespace nanobind;
  m.def("split_into_domains_int643float3d",
        [](ndarray<numpy, const std::int64_t, shape<-1, 3>> indices,
           ndarray<numpy, const float, shape<-1, 3>> points,
           ndarray<numpy, const std::int64_t, shape<-1, 2>> domain_labels,
           std::int64_t n_domains) {
          return impl::split_into_domains_impl<std::int64_t, 3, float, 3>(
              indices, points, domain_labels, n_domains);
        },
        arg("indices"), arg("points"), arg("domain_labels"),
        arg("n_domains"));
}

} // namespace tf::py
