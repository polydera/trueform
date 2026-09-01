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

#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../intersect/graph/plane_identity_names.hpp"
#include "./plane_recovery_birth.hpp"
#include "./plane_recovery_name.hpp"
#include "./plane_recovery_statement.hpp"

#include <cstddef>
#include <cstdint>

namespace tf::arrangement {

/// Lift accepted refinement records into PA's one identity/topology currency.
/// Every row states one exact current-piece birth candidate; equal names close
/// and elect once at the ordinary barrier, then the resulting identity rides
/// ordinary respan.
///
/// An edge is an edge: each split's source IS its tier — the table that
/// indexes the group it names — so the statement carries it verbatim and a
/// refinement split on a wave-born piece cuts exactly as one on an original
/// edge does.
///
/// THE FRAME LAW: the name and its row state one parameter, and its frame is
/// the name's own — `feature[0] -> feature[1]` — so an inverted split states
/// `whole - parameter` and the position the birth blends is the position the
/// producer chose. A consumer that binds its parameter to the edge instead
/// passes false.
template <typename Index, typename Param, typename Splits>
auto make_plane_refinement_evidence(
    const Splits &splits, Index name_kind, Param whole,
    bool parameters_in_name_frame,
    tf::buffer<plane_recovery_statement<Index, Param>> &statements,
    tf::buffer<plane_recovery_proposal<Index, Param>> &topology) -> void {
  const auto statement_base = statements.size();
  const auto topology_base = topology.size();
  statements.reallocate(statement_base + splits.size());
  topology.reallocate(topology_base + splits.size());
  tf::parallel_for_each(
      tf::make_sequence_range(splits.size()),
      [&](std::size_t row) {
        const auto &split = splits[row];
        const tf::intersect::graph::plane_name<Index> feature{
            name_kind, split.feature[0], split.feature[1], Index(-1),
            Index(-1)};
        const auto parameter =
            parameters_in_name_frame && split.inverted != std::uint8_t(0)
                ? Param(whole - split.parameter)
                : Param(split.parameter);
        const auto name =
            make_plane_recovery_name<Index, Param>(feature, parameter);
        statements[statement_base + row] = {
            name, split.group, split.plane, Index(split.source)};
        topology[topology_base + row] = {
            name, split.group, split.plane, Index(split.source), parameter,
            std::uint8_t(1)};
      },
      tf::checked);
}

} // namespace tf::arrangement
