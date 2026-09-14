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

#include "trueform/core/index_map.hpp"
#include "trueform/cpp/core/index_type.hpp"
#include "trueform/cpp/core/matrix.hpp"
#include "trueform/cpp/core/nd_array.hpp"

#include <type_traits>

namespace tf::cpp {

template <typename Index = default_index_t> struct index_map {
  static_assert(is_supported_index_v<Index> &&
                    std::is_same_v<Index, std::remove_cv_t<Index>>,
                "index_map requires an unqualified supported index type");

  nd_array<Index> f;
  nd_array<Index> kept_ids;

  static auto from_index_map_buffer(tf::index_map_buffer<Index> &&buffer)
      -> index_map;

  auto deep_copy() const -> index_map;
  auto is_valid() const -> bool;
};

#define TF_CPP_EXTERN_INDEX_MAP(Index) extern template struct index_map<Index>

TF_CPP_MATRIX_FOR_EACH_INDEX(TF_CPP_EXTERN_INDEX_MAP)

#undef TF_CPP_EXTERN_INDEX_MAP

} // namespace tf::cpp
