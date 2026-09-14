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
#include "intersection_curves_impl.hpp"

#include <cstdint>

namespace tf::cpp {

#define TF_CPP_INSTANTIATE_INTERSECTION_CURVES_PAIR(Ngon0, Ngon1)              \
  template auto                                                                \
  intersection_curves<std::int32_t, float, std::int64_t, Ngon0, Ngon1>(        \
      const mesh<std::int32_t, float, 3, Ngon0> &,                             \
      const mesh<std::int64_t, float, 3, Ngon1> &, tf::intersect_config)       \
      -> tf::curves_buffer<common_index_t<std::int32_t, std::int64_t>, float,  \
                           3>

TF_CPP_MATRIX_FOR_EACH_NGON_PAIR(TF_CPP_INSTANTIATE_INTERSECTION_CURVES_PAIR)

#undef TF_CPP_INSTANTIATE_INTERSECTION_CURVES_PAIR

} // namespace tf::cpp
