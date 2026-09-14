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
#include "trueform/cpp/core/matrix.hpp"
#include "trueform/cpp/core/mesh.hpp"
#include "trueform/cpp/core/nd_array.hpp"
#include "trueform/topology/domain_config.hpp"

#include <cstddef>
#include <type_traits>

namespace tf::cpp {

/// Owning per-face, per-side labels for volumetric mesh domains.
///
/// Valid domain labels occupy [0, number_of_domains()). sentinel_label() is
/// one past that range and marks excluded sides. Empty input has zero faces,
/// zero domains, sentinel 0, and outer shell -1.
template <typename Index = default_index_t> class domain_labels_result {
  static_assert(is_supported_index_v<Index> &&
                    std::is_same_v<Index, std::remove_cv_t<Index>>,
                "domain_labels_result requires an unqualified supported "
                "index type");

  nd_array<Index> _labels;
  Index _number_of_domains = 0;
  Index _outer_shell_label = -1;

public:
  domain_labels_result();
  domain_labels_result(nd_array<Index> labels, Index number_of_domains,
                       Index outer_shell_label);

  auto labels() const -> nd_array<Index>;
  auto number_of_faces() const -> Index;
  auto number_of_domains() const -> Index;
  auto valid_label_begin() const -> Index;
  auto valid_label_end() const -> Index;
  auto sentinel_label() const -> Index;
  auto outer_shell_label() const -> Index;
  auto has_outer_shell_domain() const -> bool;
  auto empty() const -> bool;
  auto label(Index face, Index side) const -> Index;
};

#define TF_CPP_EXTERN_DOMAIN_LABELS_RESULT(Index)                              \
  extern template class domain_labels_result<Index>

TF_CPP_MATRIX_FOR_EACH_INDEX(TF_CPP_EXTERN_DOMAIN_LABELS_RESULT)

#undef TF_CPP_EXTERN_DOMAIN_LABELS_RESULT

/// Compute typed labels for a 3D mesh.
template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon,
          std::enable_if_t<Dims == 3, int> = 0>
auto make_domain_labels(const mesh<Index, Real, Dims, Ngon> &value,
                        tf::domain_config config = tf::domain_config::none)
    -> domain_labels_result<Index>;

#define TF_CPP_EXTERN_DOMAIN_LABELS(Index, Real, Ngon)                         \
  extern template auto make_domain_labels<Index, Real, 3, Ngon>(               \
      const mesh<Index, Real, 3, Ngon> &, tf::domain_config)                   \
      -> domain_labels_result<Index>

TF_CPP_MATRIX_FOR_EACH_INDEX_REAL_NGON(TF_CPP_EXTERN_DOMAIN_LABELS)

#undef TF_CPP_EXTERN_DOMAIN_LABELS

} // namespace tf::cpp
