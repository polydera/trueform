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
#pragma once

#include "trueform/cpp/core/index_type.hpp"

#include <type_traits>

namespace tf::cpp::detail {

template <typename... IndexTs> struct common_index {
  static_assert(sizeof...(IndexTs) != 0,
                "common_index_t requires at least one index type");
  static_assert((is_supported_index_v<IndexTs> && ...),
                "common_index_t requires supported index types");

  using type = std::common_type_t<IndexTs...>;
  static_assert(is_supported_index_v<type>,
                "common_index_t did not produce a supported index type");
};

} // namespace tf::cpp::detail
