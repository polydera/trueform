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
#include "./signed_distance_impl.hpp"

#include <cstddef>
#include <cstdint>

namespace tf::cpp {

/// A query, named by its real alone.
template <typename QueryReal>
using signed_distance_query_t = primitive<QueryReal, 3>;
template <std::size_t Ngon> using mesh_t = mesh<std::int64_t, double, 3, Ngon>;

#define TF_CPP_INSTANTIATE_SIGNED_DISTANCE(QueryReal, Ngon)                    \
  template auto signed_distance(const mesh_t<Ngon> &,                          \
                                const signed_distance_query_t<QueryReal> &)    \
      -> distance_result<double>

TF_CPP_MATRIX_FOR_EACH_REAL_NGON(TF_CPP_INSTANTIATE_SIGNED_DISTANCE)

#undef TF_CPP_INSTANTIATE_SIGNED_DISTANCE

} // namespace tf::cpp
