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
#include "has_self_intersections_impl.hpp"

#include <cstdint>

namespace tf::cpp {

#define TF_CPP_INSTANTIATE_HAS_SELF_INTERSECTIONS(Ngon)                        \
  template auto has_self_intersections<std::int32_t, double, Ngon>(            \
      const mesh<std::int32_t, double, 3, Ngon> &) -> bool

TF_CPP_MATRIX_FOR_EACH_NGON(TF_CPP_INSTANTIATE_HAS_SELF_INTERSECTIONS)

#undef TF_CPP_INSTANTIATE_HAS_SELF_INTERSECTIONS

} // namespace tf::cpp
