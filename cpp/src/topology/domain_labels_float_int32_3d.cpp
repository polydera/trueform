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
#include "domain_labels_impl.hpp"

#include <cstdint>

namespace tf::cpp {

#define TF_CPP_INSTANTIATE_DOMAIN_LABELS(Ngon)                                 \
  template auto make_domain_labels<std::int32_t, float, 3, Ngon>(              \
      const mesh<std::int32_t, float, 3, Ngon> &, tf::domain_config)           \
      -> domain_labels_result<std::int32_t>

TF_CPP_MATRIX_FOR_EACH_NGON(TF_CPP_INSTANTIATE_DOMAIN_LABELS)

#undef TF_CPP_INSTANTIATE_DOMAIN_LABELS

} // namespace tf::cpp
