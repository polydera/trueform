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
#include "../../core/algorithm/block_reduce.hpp"
#include "../../core/algorithm/parallel_fill.hpp"
#include "../../core/buffer.hpp"
#include "../../core/range.hpp"
#include "../../core/views/blocked_range.hpp"
#include "./component_labels.hpp"
#include "../face_regions.hpp"
#include "./arrangement_descriptor.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <tuple>

namespace tf::cut {

/// @ingroup cut
/// @brief Per-domain operand-inclusion bitvectors.
///
/// Flat `uint32_t` buffer with one block of `words_per_domain` words
/// per domain. Access per-domain bitvectors via @ref make_range, which
/// returns a dynamic-block-size view: `range[d]` yields the
/// `words_per_domain` word span for domain `d`.
struct domain_inclusions {
  tf::buffer<std::uint32_t> bits;
  std::size_t words_per_domain = 0;

  /// @brief View as a blocked range over `bits`, one block per domain.
  auto make_range() const {
    return tf::make_blocked_range(tf::make_range(bits), words_per_domain);
  }
};

/// @ingroup cut
/// @brief Per-domain operand-inclusion bitvectors for an implicit
///        N-form arrangement.
///
/// Walks the wedge representatives in the @ref tf::cut::arrangement_descriptor.
/// Each representative's fan is K loops meeting at a non-manifold edge;
/// each fan loop's component (via @ref tf::loop_connectivity::loop_labels)
/// has its interior-side domain in `descriptor.domain_of_side[2c + 1]`,
/// and that domain accumulates the loop's form-tag bit.
///
/// Coplanar duplicates: each fan loop may be a survivor of a coplanar
/// pack. `_coplanar_pairs` (sorted by survivor id) yields the dead pack
/// members in a single binary search; their tag bits are set on the
/// side selected by the `reversed` flag (same direction → side 1,
/// reversed → side 0).
///
/// Returns a flat buffer of `uint32_t` words sized
/// `n_domains * words_per_domain` where
/// `words_per_domain = ceil(n_tags / 32)`. Bit
/// `inclusion[d * words_per_domain + i/32] & (1 << (i % 32))` is set
/// iff domain `d` is inside form `i`.
///
/// Scope of correctness: this is the **local-seed pass**. Components
/// that have no loop in any wedge representative's fan are not visited;
/// their tag bit remains 0 for the domains they border. That answer is
/// correct under open-mesh semantics. For closed-manifold operands
/// with multi-bundle arrangements (e.g. nested shells without surface
/// contact), the ray-cast complement step covers the residue — handled
/// by the nesting pass, deferred to Task #28.
template <typename Index, typename Index1>
auto compute_domain_inclusions(
    const tf::cut::component_labels<Index> &ag,
    const tf::face_regions<Index, Index1> &fr,
    const tf::cut::arrangement_descriptor<Index> &desc)
    -> domain_inclusions {
  const Index n_tags = ag.n_tags();
  const Index n_domains = desc.n_domains;

  domain_inclusions out;
  out.words_per_domain = static_cast<std::size_t>((n_tags + 31) / 32);
  out.bits.allocate(static_cast<std::size_t>(n_domains) *
                    out.words_per_domain);
  tf::parallel_fill(out.bits, std::uint32_t(0));

  if (n_domains == 0 || n_tags == 0)
    return out;

  auto descs = fr.descriptors();
  auto loop_labels = ag.loop_labels();
  auto domain_of_side = desc.domain_of_side;
  auto coplanar_pairs = ag.coplanar_pairs();
  const std::size_t words_per_domain = out.words_per_domain;

  // Parallel: each thread accumulates (flat_word_index, bit) emissions
  // locally. Aggregator OR-s them into the global bits buffer
  // sequentially — no atomics needed.
  struct local_t {
    tf::buffer<std::array<std::size_t, 2>> emissions; // (flat_idx, bits)
  };
  tf::blocked_reduce(
      desc.reps, std::tie(out.bits), local_t{},
      [&](const auto &chunk, local_t &local) {
        auto emit = [&](Index domain, Index tag) {
          const std::size_t flat_idx =
              static_cast<std::size_t>(domain) * words_per_domain +
              static_cast<std::size_t>(tag) / 32u;
          const std::size_t bit = std::size_t(1)
                                  << (static_cast<unsigned>(tag) % 32u);
          local.emissions.push_back({flat_idx, bit});
        };
        for (Index rep : chunk) {
          auto fan = desc.fans.faces[rep];
          for (Index k = 0; k < static_cast<Index>(fan.size()); ++k) {
            const Index loop_id = fan[k];
            const Index c = loop_labels[loop_id];

            emit(domain_of_side[2 * c + 1], descs[loop_id].tag);

            const std::array<Index, 3> key{
                loop_id, std::numeric_limits<Index>::min(), Index(0)};
            auto lo = std::lower_bound(
                coplanar_pairs.begin(), coplanar_pairs.end(), key,
                [](const std::array<Index, 3> &a,
                   const std::array<Index, 3> &b) { return a[0] < b[0]; });
            for (auto it = lo;
                 it != coplanar_pairs.end() && (*it)[0] == loop_id;
                 ++it) {
              const Index dead = (*it)[1];
              const Index reversed = (*it)[2];
              const Index interior_side = (reversed == 0) ? 1 : 0;
              emit(domain_of_side[2 * c + interior_side], descs[dead].tag);
            }
          }
        }
      },
      [](const local_t &local,
         std::tuple<tf::buffer<std::uint32_t> &> result) {
        auto &bits = std::get<0>(result);
        for (auto &e : local.emissions)
          bits[e[0]] |= static_cast<std::uint32_t>(e[1]);
      });

  return out;
}

} // namespace tf::cut
