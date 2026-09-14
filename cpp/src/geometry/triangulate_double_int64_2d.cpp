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
#include "triangulate_impl.hpp"

#include <cstdint>

namespace tf::cpp {

#define TF_CPP_INSTANTIATE_TRIANGULATE_AT_ARITY(Ngon)                          \
  template auto triangulate<std::int64_t, double, 2, Ngon>(                    \
      const mesh<std::int64_t, double, 2, Ngon> &)                             \
      -> tf::polygons_buffer<std::int64_t, double, 2, 3>

#define TF_CPP_INSTANTIATE_TRIANGULATE(Real, Index, Dims)                      \
  template auto triangulate<Index, Real, Dims>(                                \
      const offset_blocked_buffer<Index, Index> &, const nd_array<Real> &)     \
      -> tf::polygons_buffer<Index, Real, Dims, 3>;                            \
  template auto triangulate<Index, Real, Dims>(const nd_array<Index> &,        \
                                               const nd_array<Real> &)         \
      -> tf::polygons_buffer<Index, Real, Dims, 3>;                            \
  template auto triangulate<Index, Real, Dims>(const nd_array<Real> &)         \
      -> tf::polygons_buffer<Index, Real, Dims, 3>

TF_CPP_MATRIX_FOR_EACH_NGON(TF_CPP_INSTANTIATE_TRIANGULATE_AT_ARITY)
TF_CPP_INSTANTIATE_TRIANGULATE(double, std::int64_t, 2);

#undef TF_CPP_INSTANTIATE_TRIANGULATE
#undef TF_CPP_INSTANTIATE_TRIANGULATE_AT_ARITY

} // namespace tf::cpp
