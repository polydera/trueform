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
#include "../../core/buffer.hpp"
#include "../../core/views/blocked_range.hpp"
#include "../../cut/arrangements/arrangement_descriptor.hpp"
#include "../../cut/arrangements/compute_domain_inclusions.hpp"
#include "../../topology/domain_config.hpp"
#include <algorithm>
#include <array>
#include <cstdint>

namespace tf::csg::graph {

/// @ingroup csg
/// @brief Coarsened domain partition input for @ref compute_domain_partition.
///
/// `domain_of_side` is the (possibly open-merge-coarsened) per-side domain
/// id, size `2 * n_components`. `keep` is the per-coarse-domain keep flag,
/// size `n_coarse`. `coarse_of_fine[d]` maps each fine descriptor domain to
/// its coarse id (identity when `ignore_open_fragments` is off).
template <typename Index> struct domain_membership {
  tf::buffer<Index> domain_of_side; // size 2*n_components, coarse ids
  tf::buffer<bool> keep;            // size n_coarse
  tf::buffer<Index> coarse_of_fine; // size n_domains (fine)
  Index n_coarse = 0;
  tf::buffer<std::uint32_t> rep_bits; // n_coarse * words_per_domain
  std::size_t words_per_domain = 0;
  Index n_components = 0;
};

/// @brief Coarsen the descriptor's fine domains under
///        @ref tf::domain_config and compute the per-coarse keep mask.
///
/// Path-halving union-find over `[0, n_domains)`. `nesting_merges`
/// (contact-free nested shells) are always united; the open components
/// flagged by `open_mask` union their two sides only under
/// `ignore_open_fragments` (no-op for already self-merged non-sheet
/// opens, a real merge for sheet opens). A coarse domain's representative
/// bits are those of an all-zero fine constituent if one exists (the
/// universe / outer shell), else its single bounded constituent;
/// `keep[coarse] = E(rep_bits) && !(exclude_outer_shell && is_outer)`.
template <typename Index, typename OpenMask, typename Expr>
auto compute_domain_membership(
    const tf::cut::arrangement_descriptor<Index> &desc,
    const tf::cut::domain_inclusions &inc, const OpenMask &open_mask,
    const tf::buffer<std::array<Index, 2>> &nesting_merges,
    tf::domain_config config, Expr E) -> domain_membership<Index> {
  domain_membership<Index> out;
  const Index n_domains = desc.n_domains;
  out.n_components = static_cast<Index>(desc.bundle_of_component.size());

  tf::buffer<Index> parent;
  parent.allocate(static_cast<std::size_t>(n_domains));
  for (Index d = 0; d < n_domains; ++d)
    parent[d] = d;

  auto find = [&](Index x) -> Index {
    while (parent[x] != x) {
      parent[x] = parent[parent[x]];
      x = parent[x];
    }
    return x;
  };
  auto unite = [&](Index a, Index b) {
    Index ra = find(a), rb = find(b);
    if (ra != rb)
      parent[std::min(ra, rb)] = std::max(ra, rb), parent[std::max(ra, rb)] =
                                                        std::min(ra, rb);
  };

  // Nesting merges repair the false split between contact-free nested
  // shells; always applied (a correct-partition fix, not a policy).
  for (const auto &m : nesting_merges)
    unite(m[0], m[1]);

  if (config & tf::domain_config::ignore_open_fragments) {
    for (Index c = 0; c < out.n_components; ++c)
      if (open_mask[c])
        unite(desc.domain_of_side[2 * c + 0], desc.domain_of_side[2 * c + 1]);
  }

  out.coarse_of_fine.allocate(static_cast<std::size_t>(n_domains));
  tf::buffer<Index> dense_of_root;
  dense_of_root.allocate(static_cast<std::size_t>(n_domains));
  for (Index d = 0; d < n_domains; ++d)
    dense_of_root[d] = Index(-1);
  Index n_coarse = 0;
  for (Index d = 0; d < n_domains; ++d) {
    const Index r = find(d);
    if (dense_of_root[r] < 0)
      dense_of_root[r] = n_coarse++;
    out.coarse_of_fine[d] = dense_of_root[r];
  }
  out.n_coarse = n_coarse;

  out.domain_of_side.allocate(static_cast<std::size_t>(2 * out.n_components));
  for (Index i = 0; i < 2 * out.n_components; ++i) {
    const Index d = desc.domain_of_side[i];
    out.domain_of_side[i] =
        (d >= 0 && d < n_domains) ? out.coarse_of_fine[d] : Index(-1);
  }

  auto blocks = inc.make_range();
  const std::size_t words = inc.words_per_domain;

  auto &rep_bits = out.rep_bits;
  rep_bits.allocate(static_cast<std::size_t>(n_coarse) * words);
  out.words_per_domain = words;
  for (auto &w : rep_bits)
    w = 0;
  tf::buffer<bool> is_outer;
  is_outer.allocate(static_cast<std::size_t>(n_coarse));
  for (Index k = 0; k < n_coarse; ++k)
    is_outer[k] = false;
  tf::buffer<bool> rep_is_outer;
  rep_is_outer.allocate(static_cast<std::size_t>(n_coarse));
  for (Index k = 0; k < n_coarse; ++k)
    rep_is_outer[k] = false;

  for (Index d = 0; d < n_domains; ++d) {
    const Index k = out.coarse_of_fine[d];
    std::uint32_t any = 0;
    for (auto w : blocks[d])
      any |= w;
    const bool zero = (any == 0);
    if (zero)
      is_outer[k] = true;
    // Adopt an all-zero constituent's (zero) bits as the representative;
    // otherwise adopt the first bounded constituent if no zero seen yet.
    if (zero && !rep_is_outer[k]) {
      rep_is_outer[k] = true;
      std::size_t base = static_cast<std::size_t>(k) * words;
      for (std::size_t w = 0; w < words; ++w)
        rep_bits[base + w] = 0;
    } else if (!zero && !rep_is_outer[k]) {
      std::size_t base = static_cast<std::size_t>(k) * words;
      std::size_t src = static_cast<std::size_t>(d) * words;
      for (std::size_t w = 0; w < words; ++w)
        rep_bits[base + w] = inc.bits[src + w];
    }
  }

  auto rep_blocks = tf::make_blocked_range(tf::make_range(rep_bits), words);
  out.keep.allocate(static_cast<std::size_t>(n_coarse));
  for (Index k = 0; k < n_coarse; ++k) {
    const bool excluded =
        (config & tf::domain_config::exclude_outer_shell) && is_outer[k];
    out.keep[k] = E(rep_blocks[k]) && !excluded;
  }
  return out;
}

} // namespace tf::csg::graph
