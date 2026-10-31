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
#include "../identity/near_identity_request.hpp"
#include "../policy/arrangement_reseat_policy.hpp"
#include "../reseat/carve_world.hpp"
#include "./make_input_quantum.hpp"
#include <utility>

namespace tf::arrangement::dispatch {

/// The one construction point of an arrangement: the operands' lattice
/// view is stated here, from the settled policy, where the union box, the
/// completed face membership and the tolerance are all in hand, and the
/// graph is handed the view it will read for the rest of its life. Once
/// the world is built, a carrier that does not hold the position the gate
/// fused its identity onto is carved (@ref
/// tf::arrangement::reseat::carve_world) before the plane arrangement is
/// made. The input's quantum (@ref make_input_quantum) is stated here, where
/// the converter it is read through is first in hand, and the graph carries
/// it to every tier that reads it.
template <typename Int, typename Policy>
auto make_graph(Policy policy, tf::arrangement_config config,
                tf::arrangement::near_identity_request request) {
  using resolved_int_type =
      tf::exact::resolve_int_type<Int, typename Policy::input_real_type>;
  using reseat_policy_type =
      tf::arrangement::arrangement_reseat_policy<Policy, resolved_int_type>;
  using graph_type = tf::arrangement_graph<reseat_policy_type, Int>;
  using index_type = typename reseat_policy_type::index_type;
  reseat_policy_type reseat_policy(std::move(policy));
  auto lattice = reseat_policy.template make_lattice<resolved_int_type>(
      config.intersect.tolerance);
  const auto quantum =
      request.quanta != 0 || reseat_policy_type::reseats
          ? make_input_quantum(lattice.converter(),
                               reseat_policy.make_apply_to_form(),
                               reseat_policy.n_tags())
          : resolved_int_type(0);
  if constexpr (!reseat_policy_type::reseats) {
    return graph_type(std::move(reseat_policy), std::move(lattice), config,
                      quantum, request);
  } else {
    const auto converter = lattice.converter();
    const auto n_tags = reseat_policy.n_tags();
    return graph_type(
        std::move(reseat_policy), std::move(lattice), config, quantum, request,
        [&](const auto &world, const auto &apply_to_form, const auto &reader,
            auto &graph) {
          tf::arrangement::reseat::carve_world<index_type, resolved_int_type>(
              world, apply_to_form, reader, converter, n_tags, config.intersect,
              graph);
        });
  }
}

} // namespace tf::arrangement::dispatch
