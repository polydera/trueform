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
#include "../../arrangement/planes/plane_flat_weld.hpp"
#include "../../core/buffer.hpp"
#include "../../core/config/verify.hpp"
#include "../../core/offset_block_buffer.hpp"
#include "../../core/point.hpp"
#include "../../core/range.hpp"
#include "../../intersect/graph/elect_identity_components.hpp"
#include <algorithm>
#include <array>
#include <cstddef>

namespace tf::csg::graph {

/// The census classes as the tiers state them. A member is a flat id of the
/// exposed stream: below `n_flat` an original, below `n_flat` plus the plane
/// arrangement's `n_base_created()` a created identity of the local tier,
/// past it one the plane arrangement minted.
///
/// A class's candidates in the local tier's currency — its created members
/// and its originals — are one coincidence of THE ELECTION (@ref
/// tf::intersect::graph::elect_identity_component); the local tier cannot
/// speak a plane-tier identity, so a plane-tier member is no candidate of
/// it. Mint `k` is the local identity `n_created_points() + k`, at the
/// smallest original's position, and stands in `survivors` past the
/// exposed extent, at `n_flat + created_points().size() + k`. A class
/// without a local member survives as its smallest plane-tier member.
/// - `survivors`: per class, the identity it speaks;
/// - `la_welds`: `{1, source, target}` — every created member's fold onto
///   its survivor, a lawful weld of the standing world;
/// - `carve_relations`: `{0, flat, target}` — the originals' retirements,
///   the carve's statement alone;
/// - `minted`: per mint, its position;
/// - `pa_welds`: per plane-tier member of a class whose survivor stands, its
///   weld onto the survivor, carrying the plane of a triangle that witnessed
///   the member — the evidence a discarded round re-enters through;
/// - `carve_pa_welds`: the same for a class whose survivor is minted. The
///   survivor is named by its smallest original, which the closed merges
///   chase to the mint only once the carve retires it, so these welds are
///   the carve's statement too; the ticket currency never leaves the two row
///   tables.
/// The two row tables speak local identities.
template <typename Index, typename Int> struct census_statement {
  tf::buffer<Index> survivors;
  tf::buffer<std::array<Index, 3>> la_welds;
  tf::buffer<std::array<Index, 3>> carve_relations;
  tf::buffer<tf::point<Int, 3>> minted;
  tf::buffer<tf::arrangement::plane_flat_weld<Index>> pa_welds;
  tf::buffer<tf::arrangement::plane_flat_weld<Index>> carve_pa_welds;
};

/// The classes are few, so one serial pass over them states everything.
/// `pairs` carry each member's witness plane (@ref collect_census_id_pairs).
template <typename CsgGraph>
auto make_census_statement(
    const CsgGraph &g,
    const tf::offset_block_buffer<typename CsgGraph::index_type,
                                  typename CsgGraph::index_type> &classes,
    const tf::buffer<std::array<typename CsgGraph::index_type, 4>> &pairs)
    -> census_statement<typename CsgGraph::index_type,
                        typename CsgGraph::resolved_int_type> {
  using Index = typename CsgGraph::index_type;
  using Int = typename CsgGraph::resolved_int_type;
  const auto &ag = g.arrangement();
  const auto n_flat = ag.arrangement().n_flat_points();
  const auto n_local = ag.world().n_created_points(ag.identities());
  if constexpr (tf::core::verification)
    tf::core::verify(n_local == ag.arrangement().n_base_created(),
                     "no local statement follows the plane value standing "
                     "on it");
  const auto local_end = n_flat + ag.arrangement().n_base_created();
  const auto mint_base = n_flat + Index(ag.created_points().size());
  const auto reader = ag.lattice().reader(ag.apply_to_form());
  tf::buffer<std::array<Index, 2>> witnesses;
  for (const auto &pair : pairs)
    for (std::size_t k = 0; k < 2; ++k)
      if (pair[k] >= local_end)
        witnesses.push_back({pair[k], pair[2 + k]});
  std::sort(witnesses.begin(), witnesses.end());
  witnesses.erase_till_end(
      std::unique(witnesses.begin(), witnesses.end(),
                  [](const auto &a, const auto &b) { return a[0] == b[0]; }));
  const auto witness_of = [&](Index id) -> Index {
    const auto at = std::lower_bound(
        witnesses.begin(), witnesses.end(), id,
        [](const std::array<Index, 2> &w, Index v) { return w[0] < v; });
    if constexpr (tf::core::verification)
      tf::core::verify(at != witnesses.end() && (*at)[0] == id,
                       "every plane-tier member has a witness");
    return (*at)[1];
  };
  census_statement<Index, Int> out;
  out.survivors.allocate(classes.size());
  tf::buffer<tf::intersect::graph::identity_key<Index>> candidates;
  tf::buffer<std::array<Index, 3>> rows;
  const auto mint =
      [&](const tf::intersect::graph::identity_key<Index> &winner) {
        out.minted.push_back(reader.point_of_flat(winner[1]));
        return n_local + Index(out.minted.size()) - Index(1);
      };
  for (std::size_t c = 0; c < classes.size(); ++c) {
    const auto members = classes[c];
    if constexpr (tf::core::verification)
      tf::core::verify(members.size() >= 2 &&
                           std::is_sorted(members.begin(), members.end()) &&
                           std::adjacent_find(members.begin(), members.end()) ==
                               members.end(),
                       "a census class holds two or more ascending members");
    const auto created =
        std::lower_bound(members.begin(), members.end(), n_flat);
    const auto planar = std::lower_bound(created, members.end(), local_end);
    const bool holds_original = created != members.begin();
    auto &survivor = out.survivors[c];
    bool mints = false;
    if (created != planar || holds_original) {
      candidates.clear();
      for (auto at = created; at != planar; ++at)
        candidates.push_back(tf::intersect::graph::standing_identity_key<Index>(
            Index(*at - n_flat)));
      for (auto at = members.begin(); at != created; ++at)
        candidates.push_back(
            tf::intersect::graph::original_identity_key<Index>(Index(*at)));
      rows.clear();
      const auto target = tf::intersect::graph::elect_identity_component<Index>(
          tf::make_range(candidates), mint, rows);
      mints = target >= n_local;
      survivor = mints ? mint_base + (target - n_local) : n_flat + target;
      // an original's retirement is the carve's statement; a created
      // member's fold is a lawful weld of the standing world either way
      for (const auto &row : rows)
        (row[0] == Index(0) ? out.carve_relations : out.la_welds)
            .push_back(row);
    } else
      survivor = *planar;
    // a minted survivor's plane-tier weld names the smallest original
    // instead: the mint ticket is the carve's to rebase, while the original
    // chases to the mint through the closed merges it retires into, so the
    // weld rides the carve that states the retirement
    const auto weld_survivor = mints ? *members.begin() : survivor;
    auto &weld_rows = mints ? out.carve_pa_welds : out.pa_welds;
    for (auto at = planar; at != members.end(); ++at)
      if (*at != survivor)
        weld_rows.push_back({weld_survivor, *at, witness_of(*at)});
  }
  return out;
}

} // namespace tf::csg::graph
