/*
 * Copyright (c) 2026 XLAB
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
#include "../../exact/resolve_int_type.hpp"
#include "../arrangement_config.hpp"
#include "../arrangement_graph.hpp"
#include <utility>

namespace tf::arrangement::dispatch {

/// The one construction point of an arrangement: the operands' lattice
/// view is stated here, from the settled policy, where the union box, the
/// completed face membership and the tolerance are all in hand, and the
/// graph is handed the view it will read for the rest of its life.
template <typename Int, typename Policy>
auto make_graph(Policy policy, tf::arrangement_config config) {
  using resolved_int_type =
      tf::exact::resolve_int_type<Int, typename Policy::input_real_type>;
  auto lattice = policy.template make_lattice<resolved_int_type>(
      config.intersect.tolerance);
  return tf::arrangement_graph<Policy, Int>(std::move(policy),
                                            std::move(lattice), config);
}

} // namespace tf::arrangement::dispatch
