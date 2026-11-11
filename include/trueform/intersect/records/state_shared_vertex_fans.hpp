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
#include "../../topology/topo_type.hpp"
#include "../../topology/vertex_id_in_face.hpp"
#include "./shared_vertex_claim.hpp"
#include "./tagged_intersection.hpp"
#include "tbb/parallel_sort.h"

#include <algorithm>
#include <cstddef>
#include <utility>

namespace tf::intersect {

/// THE SHARED VERTEX'S FAN IS ONE STATEMENT PER CLAIM.
///
/// A self pair that shares a vertex states that vertex at every face of
/// the vertex's fan, so every incident loop substitutes the corner with
/// the same created point and cut and uncut faces agree on the vertex's
/// identity; a copy landing in a non-contact group is vertex-only and
/// stays silent. What the fan states is a function of the claim alone —
/// not of the partner the pair named, not of the record that found it —
/// so the pairs that make the same claim state one fact, and it is
/// stated here, over the claims the duplicator left. The pair's own
/// contact at the vertex is the duplicator's own, and is not restated.
///
/// The claim is the carrier: the claims are ordered and made unique, the
/// records each one states are counted and prefixed, and every claim
/// writes its fan into its own slice. The records go to the end of
/// `records`, before the generator dedup reads it.
template <typename Index, typename ApplyToForm, typename Sentinel>
auto state_shared_vertex_fans(tf::buffer<shared_vertex_claim<Index>> &claims,
                              const ApplyToForm &apply_to_form,
                              const Sentinel &sentinel_of,
                              tf::buffer<tagged_intersection<Index>> &records)
    -> void {
  if (claims.size() == 0)
    return;
  // A small batch is sorted below the cost of entering the parallel
  // machinery.
  if (claims.size() < 4096)
    std::sort(claims.begin(), claims.end());
  else
    tbb::parallel_sort(claims.begin(), claims.end());
  claims.erase_till_end(std::unique(claims.begin(), claims.end()));

  const auto n = claims.size();
  tf::buffer<std::size_t> at;
  at.allocate(n + 1);
  at[0] = 0;
  tf::parallel_for_each(
      tf::make_sequence_range(n),
      [&](std::size_t k) {
        const auto &claim = claims[k];
        // the membership holds one entry per corner, so a face that holds
        // the vertex twice stands in the fan twice; the count skips what
        // the write skips
        std::size_t fan = 0;
        apply_to_form(int(claim.tag), [&](const auto &form) {
          for (auto face : form.face_membership()[claim.vertex])
            fan += Index(face) != claim.object;
        });
        at[k + 1] = 2 * fan;
      },
      tf::checked);
  for (std::size_t k = 1; k <= n; ++k)
    at[k] += at[k - 1];

  const auto base = records.size();
  records.reallocate(base + at[n]);
  tf::parallel_for_each(
      tf::make_sequence_range(n),
      [&](std::size_t k) {
        const auto &claim = claims[k];
        const auto vid = claim.vertex;
        auto *out = records.begin() + std::ptrdiff_t(base + at[k]);
        apply_to_form(int(claim.tag), [&](const auto &form) {
          auto &&faces = form.faces();
          tagged_intersection<Index> v;
          v.tag = claim.tag;
          v.tag_other = claim.tag;
          v.object_other = claim.object;
          v.target_other = {
              tf::vertex_id_in_face<Index>(vid, faces[claim.object]),
              tf::topo_type::vertex};
          v.id = sentinel_of(claim.tag) + vid;
          for (auto face : form.face_membership()[vid]) {
            if (Index(face) == claim.object)
              continue;
            v.object = Index(face);
            v.target = {tf::vertex_id_in_face<Index>(vid, faces[Index(face)]),
                        tf::topo_type::vertex};
            *out++ = v;
            auto w = v;
            std::swap(w.object, w.object_other);
            std::swap(w.target, w.target_other);
            *out++ = w;
          }
        });
      },
      tf::checked);
}

} // namespace tf::intersect
