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
#include "boolean_impl.hpp"

#include <cstdint>
#include <vector>

namespace tf::cpp {

#define TF_CPP_INSTANTIATE_BOOLEAN(Ngon0, Ngon1)                               \
  template auto                                                                \
  make_boolean<std::int64_t, double, std::int32_t, Ngon0, Ngon1>(              \
      const mesh<std::int64_t, double, 3, Ngon0> &,                            \
      const mesh<std::int32_t, double, 3, Ngon1> &, tf::boolean_op,            \
      std::vector<std::int32_t>, tf::arrangement_config)                       \
      -> boolean_result<common_index_t<std::int64_t, std::int32_t>,        \
                            double,                                            \
                            detail::concatenated_arity_v<Ngon0, Ngon1>>;       \
  template auto                                                                \
  make_boolean_with_curves<std::int64_t, double, std::int32_t, Ngon0, Ngon1>(  \
      const mesh<std::int64_t, double, 3, Ngon0> &,                            \
      const mesh<std::int32_t, double, 3, Ngon1> &, tf::boolean_op,            \
      std::vector<std::int32_t>, tf::arrangement_config)                       \
      -> boolean_with_curves_result<                                       \
          common_index_t<std::int64_t, std::int32_t>, double,                  \
          detail::concatenated_arity_v<Ngon0, Ngon1>>

TF_CPP_MATRIX_FOR_EACH_NGON_PAIR(TF_CPP_INSTANTIATE_BOOLEAN)

#undef TF_CPP_INSTANTIATE_BOOLEAN

} // namespace tf::cpp
