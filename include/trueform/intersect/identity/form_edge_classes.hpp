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
#include "../../core/algorithm/compute_offsets.hpp"
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../exact/edge_parameter.hpp"
#include "../../exact/meta.hpp"
#include "../classify/intersection_payload.hpp"
#include "../records/tagged_intersection.hpp"
#include "./identity_records.hpp"
#include "tbb/parallel_sort.h"

#include <algorithm>
#include <cstddef>
#include <iterator>

namespace tf::intersect {

/// Group the kind-E candidates by canonical carrier and split each
/// carrier's candidates into exact-parameter classes.
///
/// The carrier is the whole identity: two candidates on one carrier with
/// equal exact parameter are one point, whatever second generator found
/// them — that is on-edge junction closure by construction.
///
/// A vertex lying on an edge is a candidate too: its identity stays the
/// vertex, but it subdivides the carrier and must take its place in the
/// carrier's order.
///
/// Every parameter was computed where the incidence was decided, by the
/// kernel that decided it, and travels on the record's id; a candidate
/// only names which of its record's fractions is its own and which way
/// round it runs. Nothing here reads a coordinate, and nothing recovers
/// a fact the classifier already proved.
template <typename Index, typename Int>
auto form_edge_classes(
    const tf::buffer<tf::intersect::tagged_intersection<Index>> &records,
    const tf::buffer<tf::exact::edge_fractions<Int, Index>> &parameters,
    Index no_endpoint, identity_scratch<Index, Int> &scratch,
    edge_classes<Index, Int> &out) -> void {
  using T2 = typename tf::exact::meta<Int>::T2;

  auto &candidates = scratch.candidates;
  auto &candidate_class = scratch.candidate_class;
  out.carriers.clear();
  out.offsets.clear();
  out.endpoint.clear();
  out.parameter.clear();
  out.offsets.push_back(Index(0));
  if (candidates.size() == 0)
    return;

  // Only records with an edge target become candidates, and those are
  // exactly the records the kernels gave a fraction — never the
  // shared-vertex deliveries, whose ids are sentinels past the table.
  auto parameter_of =
      [&](const edge_candidate<Index> &cand) -> tf::exact::edge_parameter<Int> {
    const auto &fractions =
        parameters[std::size_t(records[std::size_t(cand.row)].id)];
    const auto &t = fractions.t[std::size_t(cand.parameter)];
    return cand.reversed ? tf::exact::reversed_parameter(t) : t;
  };

  if (candidates.size() < 4096)
    std::sort(candidates.begin(), candidates.end());
  else
    tbb::parallel_sort(candidates.begin(), candidates.end());

  auto &carrier_offsets = scratch.carrier_offsets;
  carrier_offsets.clear();
  carrier_offsets.reserve(candidates.size() + 1);
  tf::compute_offsets(
      candidates, std::back_inserter(carrier_offsets), Index(0),
      [](const auto &a, const auto &b) { return a.carrier == b.carrier; });
  const auto n_carriers = carrier_offsets.size() - 1;

  auto &slot_endpoint = scratch.slot_endpoint;
  auto &counts = scratch.counts;
  auto &slot_parameter = scratch.slot_parameter;
  slot_endpoint.allocate(candidates.size());
  slot_parameter.allocate(candidates.size());
  counts.allocate(n_carriers);
  candidate_class.allocate(candidates.size());
  out.carriers.allocate(n_carriers);

  struct local_t {
    tf::buffer<tf::exact::edge_parameter<Int>> params;
    tf::buffer<Index> order;
  };

  tf::parallel_for_each(
      tf::make_sequence_range(Index(n_carriers)),
      [&](Index g, local_t &local) {
        const auto lo = std::size_t(carrier_offsets[std::size_t(g)]);
        const auto hi = std::size_t(carrier_offsets[std::size_t(g) + 1]);
        const auto carrier = candidates[lo].carrier;
        out.carriers[std::size_t(g)] = carrier;

        const auto m = hi - lo;
        local.params.allocate(m);
        local.order.allocate(m);
        for (std::size_t k = 0; k < m; ++k) {
          local.params[k] = parameter_of(candidates[lo + k]);
          local.order[k] = Index(k);
        }
        std::sort(local.order.begin(), local.order.end(),
                  [&local](Index a, Index b) {
                    return tf::exact::compare_parameter(
                               local.params[std::size_t(a)],
                               local.params[std::size_t(b)]) < 0;
                  });

        Index c = 0;
        for (std::size_t k = 0; k < m; ++k) {
          const auto j = std::size_t(local.order[k]);
          const auto &t = local.params[j];
          if (k == 0 ||
              tf::exact::compare_parameter(
                  t, local.params[std::size_t(local.order[k - 1])]) != 0) {
            if (k != 0)
              ++c;
            const auto slot = lo + std::size_t(c);
            slot_parameter[slot] = t;
            slot_endpoint[slot] = t.num == T2(0)   ? carrier.u
                                  : t.num == t.den ? carrier.v
                                                   : no_endpoint;
          }
          candidate_class[lo + j] = c;
        }
        counts[std::size_t(g)] = c + 1;
      },
      local_t{});

  out.offsets.reallocate(n_carriers + 1);
  for (std::size_t g = 0; g < n_carriers; ++g)
    out.offsets[g + 1] = out.offsets[g] + counts[g];
  const auto n_classes = std::size_t(out.offsets[n_carriers]);

  out.endpoint.allocate(n_classes);
  out.parameter.allocate(n_classes);
  tf::parallel_for_each(
      tf::make_sequence_range(Index(n_carriers)),
      [&](Index g) {
        const auto lo = std::size_t(carrier_offsets[std::size_t(g)]);
        const auto hi = std::size_t(carrier_offsets[std::size_t(g) + 1]);
        const auto base = out.offsets[std::size_t(g)];
        for (Index c = 0; c < counts[std::size_t(g)]; ++c) {
          const auto slot = lo + std::size_t(c);
          const auto dst = std::size_t(base + c);
          out.endpoint[dst] = slot_endpoint[slot];
          out.parameter[dst] = slot_parameter[slot];
        }
        for (auto i = lo; i < hi; ++i)
          candidate_class[i] += base;
      },
      tf::checked);
}

} // namespace tf::intersect
