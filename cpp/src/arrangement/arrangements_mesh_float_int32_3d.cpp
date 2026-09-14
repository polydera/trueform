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
#include "arrangements_impl.hpp"

#include <cstdint>
#include <vector>

namespace tf::cpp {

#define TF_CPP_INSTANTIATE_RANGE_ARRANGEMENTS(Ngon)                            \
  template auto mesh_arrangements<std::int32_t, float, Ngon>(                  \
      const std::vector<mesh<std::int32_t, float, 3, Ngon>> &,                 \
      tf::arrangement_config)                                                  \
      -> mesh_arrangement_result<std::int32_t, float, Ngon>;                   \
  template auto mesh_arrangements_with_curves<std::int32_t, float, Ngon>(      \
      const std::vector<mesh<std::int32_t, float, 3, Ngon>> &,                 \
      tf::arrangement_config)                                                  \
      -> mesh_arrangement_with_curves_result<std::int32_t, float, Ngon>

TF_CPP_MATRIX_FOR_EACH_NGON(TF_CPP_INSTANTIATE_RANGE_ARRANGEMENTS)

#undef TF_CPP_INSTANTIATE_RANGE_ARRANGEMENTS

} // namespace tf::cpp
