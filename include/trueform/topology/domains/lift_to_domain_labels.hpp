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
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../domain_labels.hpp"

namespace tf::topology::domains {

/// @ingroup topology_components
/// @brief Compact any sentinel-flagged garbage class out of
/// `domain_of_side` and fan it out to per-face per-side domain labels.
///
/// The per-face write is the final, large output of `make_domain_labels`.
/// It runs after every source of merges (radial-vote, boundary,
/// optional nesting) has settled into the small `domain_of_side`
/// buffer, so the per-face buffer is written exactly once with final
/// ids.
///
/// @param domain_of_side Per-fragment-side domain id buffer of size
///        `2 * n_components`, produced by
///        @ref tf::topology::domains::build_domain_of_side and
///        optionally remapped by nesting.
/// @param n_domains Number of distinct domains in `domain_of_side`.
/// @param fragment_labels Per-face MEL fragment labels.
/// @param n_faces Number of faces in the source mesh.
/// @param excluded_domain Optional domain id to be excluded from the
///        output. When `>= 0`, that domain is parked at the sentinel
///        `label == n_domains - 1` (one past the new compacted range)
///        and all other domain ids are compacted into
///        `[0, n_domains - 1)`. Default `-1` skips compaction.
/// @param outer_shell_domain Index of the outer-shell (unbounded
///        universe) domain, already remapped. Used only when
///        `excluded_domain < 0` (the outer shell is kept), where it
///        becomes `outer_shell_label`; ignored otherwise, since the
///        excluded path reports the sentinel instead. Default `-1`.
/// @return @ref tf::domain_labels with `n_faces` 2-blocks of per-side
///         domain ids.
template <typename Index, typename FragLabels>
auto lift_to_domain_labels(tf::buffer<Index> &domain_of_side, Index n_domains,
                           const FragLabels &fragment_labels, Index n_faces,
                           Index excluded_domain = Index(-1),
                           Index outer_shell_domain = Index(-1))
    -> tf::domain_labels<Index> {
  if (excluded_domain >= 0) {
    Index garbage_id = excluded_domain;
    Index sentinel = n_domains - 1;
    // The outer-shell domain rides the same compaction as every side:
    // folded to the sentinel when it *is* the excluded domain (i.e.
    // exclude_outer_shell removed the universe), otherwise shifted down
    // past the removed slot. When some *other* domain is excluded (a fin
    // garbage class under ignore_open_fragments), the universe stays a
    // real domain and we report its shifted index.
    if (outer_shell_domain == garbage_id)
      outer_shell_domain = sentinel;
    else if (outer_shell_domain > garbage_id)
      outer_shell_domain -= 1;
    tf::parallel_for_each(
        domain_of_side,
        [&](Index &d) {
          if (d > garbage_id)
            d -= 1;
          else if (d == garbage_id)
            d = sentinel;
        },
        tf::checked);
    n_domains -= 1;
  }

  tf::domain_labels<Index> out;
  out.labels.allocate(n_faces);
  out.n_domains = n_domains;
  // exclude_outer_shell: the universe was the excluded domain, so this is
  // the sentinel `n_domains` (one past the range). Otherwise it is the
  // index of the outer-shell domain in `[0, n_domains)`, or `-1` if none.
  out.outer_shell_label = outer_shell_domain;
  auto &flat = out.labels.data_buffer();
  tf::parallel_for_each(tf::make_sequence_range(n_faces), [&](Index f) {
    Index frag = fragment_labels.labels[f];
    flat[2 * f + 0] = domain_of_side[2 * frag + 0];
    flat[2 * f + 1] = domain_of_side[2 * frag + 1];
  });
  return out;
}

} // namespace tf::topology::domains
