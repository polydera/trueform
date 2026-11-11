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
#include "../../exact/pt_converter.hpp"
#include "../../spatial/policy/tree.hpp"
#include "../../topology/policy/manifold_edge_link.hpp"
#include "../arrangement_config.hpp"
#include "./make_graph.hpp"
#include <type_traits>
#include <utility>

namespace tf::arrangement {

/// @brief The arrangement of `operands`, completed by
///        @ref tf::arrangement::complete_operands, on the lattice `converter`
///        states.
///
/// The body of @ref tf::make_arrangement_graph, which completes its
/// operands and states the house converter over them; a caller holding the
/// converter its own lattice is read on states it here, and every original
/// vertex is read through it. The body completes nothing and builds nothing
/// but the graph; the converter's integer is the lattice's.
template <typename Operands, typename ConverterInt, typename RealT>
auto make_arrangement_graph(
    Operands operands,
    const tf::exact::pt_converter<ConverterInt, RealT, 3> &converter,
    tf::arrangement_config config = {}) {
  operands.make_apply_to_form()(0, [](const auto &form) {
    using form_t = std::decay_t<decltype(form)>;
    static_assert(tf::has_tree_policy<form_t> &&
                      tf::has_manifold_edge_link_policy<form_t>,
                  "the body arranges completed operands");
  });
  return tf::arrangement::dispatch::make_graph<ConverterInt>(
      std::move(operands), converter, config, {});
}

} // namespace tf::arrangement
