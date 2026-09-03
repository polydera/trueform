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
#include "../core/buffer.hpp"
#include "./connected_component_labels_like.hpp"

namespace tf {

/// @ingroup topology_components
/// @brief Owning container for per-element connected component labels.
///
/// Carries `labels` (one label per element of the producer's carrier —
/// faces for the edge rules, vertices for the vertex rule) and
/// `n_components` (total label count).
///
/// Build via one of @ref tf::make_manifold_edge_connected_component_labels,
/// @ref tf::make_edge_connected_component_labels, or
/// @ref tf::make_vertex_connected_component_labels.
///
/// @tparam LabelType The integer type for component labels.
template <typename LabelType>
class connected_component_labels
    : public connected_component_labels_like<
          tf::topology::connected_component_labels_policy<
              LabelType, tf::buffer<LabelType>>> {
  using base_t = connected_component_labels_like<
      tf::topology::connected_component_labels_policy<LabelType,
                                                      tf::buffer<LabelType>>>;

public:
  using base_t::base_t;
};

} // namespace tf
