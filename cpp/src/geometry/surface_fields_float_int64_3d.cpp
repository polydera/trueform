/*
 * Copyright (c) 2026 XLAB
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
#include "surface_fields_impl.hpp"

#include <cstdint>

namespace tf::cpp {

#define TF_CPP_INSTANTIATE_GEODESIC_DISTANCES(Ngon)                            \
  template auto geodesic_distances<std::int64_t, float, 3, Ngon>(              \
      const mesh<std::int64_t, float, 3, Ngon> &,                              \
      const nd_array<std::int64_t> &, double) -> nd_array<float>

TF_CPP_MATRIX_FOR_EACH_NGON(TF_CPP_INSTANTIATE_GEODESIC_DISTANCES)

#undef TF_CPP_INSTANTIATE_GEODESIC_DISTANCES

template auto dirichlet_field<std::int64_t, float, 3>(
    const mesh<std::int64_t, float, 3, 3> &, const nd_array<double> &,
    const nd_array<std::int64_t> &, const nd_array<double> &,
    tf::dirichlet_order) -> dirichlet_field_result<float>;
template auto dirichlet_field<std::int64_t, float, 3>(
    const mesh<std::int64_t, float, 3, 3> &, const nd_array<std::int64_t> &,
    const nd_array<double> &, tf::dirichlet_order)
    -> dirichlet_field_result<float>;

} // namespace tf::cpp
