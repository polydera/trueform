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
#include "../../core/range.hpp"
#include <utility>

namespace tf::topology {

/// @ingroup topology_components
/// @brief Storage-agnostic policy for connected component labels.
///
/// Stores `labels` of type `Range`, which may be an owning container
/// such as `tf::buffer<LabelType>`, or a non-owning view such as
/// `tf::range<const LabelType*, tf::dynamic_size>` — both satisfy the
/// same labels-container interface, so the policy is parameterized
/// over either.
///
/// @tparam LabelType The integer type for component labels.
/// @tparam Range The underlying labels container (owning or view).
template <typename LabelType, typename Range>
struct connected_component_labels_policy {
  using label_type = LabelType;
  using range_type = Range;

  Range labels;
  LabelType n_components;

  connected_component_labels_policy() = default;

  connected_component_labels_policy(Range r, LabelType n)
      : labels{std::move(r)}, n_components{n} {}
};

/// @ingroup topology_components
/// @brief Wrap an external labels range as a view policy.
template <typename Range, typename LabelType>
auto make_connected_component_labels_policy(Range &&r,
                                            LabelType n_components) {
  auto labels = tf::make_range(static_cast<Range &&>(r));
  using policy_t =
      connected_component_labels_policy<LabelType,
                                        std::decay_t<decltype(labels)>>;
  return policy_t{labels, n_components};
}

} // namespace tf::topology
