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
#include "isobands_impl.hpp"

#include <cstdint>

namespace tf::cpp {

#define TF_CPP_INSTANTIATE_ISOBANDS(Ngon)                                      \
  template auto isobands<std::int32_t, double, Ngon>(                          \
      const mesh<std::int32_t, double, 3, Ngon> &, const nd_array<double> &,   \
      const nd_array<double> &)                                                \
      -> isobands_result<std::int32_t, double, Ngon>;                          \
  template auto isobands_with_curves<std::int32_t, double, Ngon>(              \
      const mesh<std::int32_t, double, 3, Ngon> &, const nd_array<double> &,   \
      const nd_array<double> &)                                                \
      -> isobands_with_curves_result<std::int32_t, double, Ngon>;              \
  template auto isobands_selected<std::int32_t, double, Ngon>(                 \
      const mesh<std::int32_t, double, 3, Ngon> &, const nd_array<double> &,   \
      const nd_array<double> &, const nd_array<std::int32_t> &)                \
      -> isobands_result<std::int32_t, double, Ngon>;                          \
  template auto isobands_with_curves_selected<std::int32_t, double, Ngon>(     \
      const mesh<std::int32_t, double, 3, Ngon> &, const nd_array<double> &,   \
      const nd_array<double> &, const nd_array<std::int32_t> &)                \
      -> isobands_with_curves_result<std::int32_t, double, Ngon>

TF_CPP_MATRIX_FOR_EACH_NGON(TF_CPP_INSTANTIATE_ISOBANDS)

#undef TF_CPP_INSTANTIATE_ISOBANDS

} // namespace tf::cpp
