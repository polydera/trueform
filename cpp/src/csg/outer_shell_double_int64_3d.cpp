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
#include "outer_shell_impl.hpp"

#include <cstdint>

namespace tf::cpp {

#define TF_CPP_INSTANTIATE_OUTER_SHELL(Ngon)                                   \
  template auto outer_shell<std::int64_t, double, Ngon>(                       \
      const mesh<std::int64_t, double, 3, Ngon> &, tf::intersect_config)       \
      -> tf::polygons_buffer<std::int64_t, double, 3, Ngon>

TF_CPP_MATRIX_FOR_EACH_NGON(TF_CPP_INSTANTIATE_OUTER_SHELL)

#undef TF_CPP_INSTANTIATE_OUTER_SHELL

} // namespace tf::cpp
