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

#include "trueform/core/polygons_buffer.hpp"
#include "trueform/cpp/core/matrix.hpp"
#include "trueform/cpp/core/mesh.hpp"
#include "trueform/cpp/core/nd_array.hpp"
#include "trueform/cpp/topology/domain_labels.hpp"

#include <cstddef>
#include <vector>

namespace tf::cpp {

/// @brief Owning domain split preserving carrier and label index width.
template <typename Index, typename Real, std::size_t Dims = 3,
          std::size_t Ngon = 3>
struct split_domains_result {
  std::vector<tf::polygons_buffer<Index, Real, Dims, Ngon>> components;
  nd_array<Index> labels;
};

/// @brief Split a typed fixed/dynamic mesh using labels of the same index type.
template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto split_into_domains(const mesh<Index, Real, Dims, Ngon> &value,
                        const domain_labels_result<Index> &labels)
    -> split_domains_result<Index, Real, Dims, Ngon>;

#define TF_CPP_REINDEX_EXTERN_SPLIT_DOMAINS(Index, Real, Dims, Ngon)           \
  extern template auto split_into_domains<Index, Real, Dims, Ngon>(            \
      const mesh<Index, Real, Dims, Ngon> &,                                   \
      const domain_labels_result<Index> &)                                     \
      -> split_domains_result<Index, Real, Dims, Ngon>

TF_CPP_MATRIX_FOR_EACH_INDEX_REAL_DIMS_NGON(TF_CPP_REINDEX_EXTERN_SPLIT_DOMAINS)

#undef TF_CPP_REINDEX_EXTERN_SPLIT_DOMAINS

} // namespace tf::cpp
