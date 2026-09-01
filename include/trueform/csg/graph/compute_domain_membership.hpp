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
#include "../../topology/domain_config.hpp"
#include "./arrangement_descriptor.hpp"
#include "./domain_inclusions.hpp"
#include "./union_find.hpp"
#include <array>
#include <cstddef>
#include <cstdint>

namespace tf::csg::graph {

/// @ingroup csg_graph_internals
/// @brief Coarsened domain partition input for @ref compute_domain_partition.
///
/// `domain_of_side` is the (possibly open-merge-coarsened) per-side domain
/// id, size `2 * n_components`. `keep` is the per-coarse-domain keep flag,
/// size `n_coarse`. `coarse_of_fine[d]` maps each fine descriptor domain to
/// its coarse id (identity when `ignore_open_fragments` is off). `rep` holds
/// each coarse domain's representative bits, one block per coarse domain.
template <typename Index> struct domain_membership {
  tf::buffer<Index> domain_of_side; // size 2*n_components, coarse ids
  tf::buffer<bool> keep;            // size n_coarse
  tf::buffer<Index> coarse_of_fine; // size n_domains (fine)
  Index n_coarse = 0;
  tf::csg::graph::domain_inclusions rep;
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
/// `keep[coarse] = E(rep) && !(exclude_outer_shell && is_outer)`.
///
/// `universe_fine` >= 0 selects the STRUCTURAL read (one-form graphs):
/// the inclusion bits of a self arrangement are winding parity, which
/// misreads double-covered pockets as outside. With the volume-argmin
/// universe passed in, `is_outer` is that domain alone and every other
/// coarse domain carries the honest "enclosed by form 0" bit.
template <typename Index, typename OpenMask, typename Expr>
auto compute_domain_membership(
    const tf::csg::graph::arrangement_descriptor<Index> &desc,
    const tf::csg::graph::domain_inclusions &inc, const OpenMask &open_mask,
    const tf::buffer<std::array<Index, 2>> &nesting_merges,
    tf::domain_config config, Expr E, Index universe_fine = Index(-1),
    const tf::buffer<char> &is_sheet_tag = {},
    const tf::buffer<std::array<Index, 3>> &sheet_folds = {})
    -> domain_membership<Index> {
  domain_membership<Index> out;
  const Index n_domains = desc.n_domains;
  out.n_components = static_cast<Index>(desc.bundle_of_component.size());

  tf::csg::graph::union_find<Index> uf;
  uf.reset(static_cast<std::size_t>(n_domains));

  // Nesting merges repair the false split between contact-free nested
  // shells; always applied (a correct-partition fix, not a policy).
  for (const auto &m : nesting_merges)
    uf.unite(m[0], m[1]);

  // Fusing an open component's sides erases the partition that gave
  // "behind" its meaning for every sheet in that wall: the component's
  // own tag and the folds (a dropped duplicate has no component of its
  // own to fuse). Their bits clear from the fused representative below.
  tf::buffer<std::array<Index, 2>> sheet_fuses; // (fine domain, sheet tag)
  if (config & tf::domain_config::ignore_open_fragments) {
    tf::buffer<bool> fused;
    fused.allocate(static_cast<std::size_t>(out.n_components));
    for (Index c = 0; c < out.n_components; ++c) {
      fused[c] = bool(open_mask[c]);
      if (!fused[c])
        continue;
      uf.unite(desc.domain_of_side[2 * c + 0],
               desc.domain_of_side[2 * c + 1]);
      const Index t = desc.tag_of_component[c];
      if (t >= 0 && t < Index(is_sheet_tag.size()) && is_sheet_tag[t])
        sheet_fuses.push_back({desc.domain_of_side[2 * c + 0], t});
    }
    for (const auto &f : sheet_folds)
      if (fused[f[0]])
        sheet_fuses.push_back({desc.domain_of_side[2 * f[0] + 0], f[1]});
  }

  out.coarse_of_fine.allocate(static_cast<std::size_t>(n_domains));
  tf::buffer<Index> dense_of_root;
  dense_of_root.allocate(static_cast<std::size_t>(n_domains));
  for (Index d = 0; d < n_domains; ++d)
    dense_of_root[d] = Index(-1);
  Index n_coarse = 0;
  for (Index d = 0; d < n_domains; ++d) {
    const Index r = uf.find(d);
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

  const std::size_t words = inc.words_per_domain;

  auto &rep = out.rep;
  rep.words_per_domain = words;
  rep.bits.allocate(static_cast<std::size_t>(n_coarse) * words);
  for (auto &w : rep.bits)
    w = 0;
  tf::buffer<bool> is_outer;
  is_outer.allocate(static_cast<std::size_t>(n_coarse));
  for (Index k = 0; k < n_coarse; ++k)
    is_outer[k] = false;

  if (universe_fine >= 0) {
    const Index uk = out.coarse_of_fine[universe_fine];
    for (Index k = 0; k < n_coarse; ++k) {
      is_outer[k] = k == uk;
      if (k != uk)
        rep.set(static_cast<std::size_t>(k), 0);
    }
  } else {
    // Representative = OR of constituent bits, minus the bits of sheets
    // whose open fuse formed the class. Classes are bits-homogeneous
    // outside fused sheet columns (nesting merges unite copies of one
    // physical region; open fuses only flip the fused sheet's own bit),
    // so this reproduces the constituent bits on volume-only input and
    // the universe reads all-zero again once its fused sheet halves
    // stop meaning "behind".
    for (Index d = 0; d < n_domains; ++d) {
      const std::size_t base =
          static_cast<std::size_t>(out.coarse_of_fine[d]) * words;
      const std::size_t src = static_cast<std::size_t>(d) * words;
      for (std::size_t w = 0; w < words; ++w)
        rep.bits[base + w] |= inc.bits[src + w];
    }
    for (const auto &f : sheet_fuses)
      rep.clear(static_cast<std::size_t>(out.coarse_of_fine[f[0]]),
                static_cast<std::size_t>(f[1]));
    for (Index k = 0; k < n_coarse; ++k) {
      std::uint32_t any = 0;
      const std::size_t base = static_cast<std::size_t>(k) * words;
      for (std::size_t w = 0; w < words; ++w)
        any |= rep.bits[base + w];
      is_outer[k] = (any == 0);
    }
  }

  auto rep_blocks = rep.make_range();
  out.keep.allocate(static_cast<std::size_t>(n_coarse));
  for (Index k = 0; k < n_coarse; ++k) {
    const bool excluded =
        (config & tf::domain_config::exclude_outer_shell) && is_outer[k];
    out.keep[k] = E(rep_blocks[k]) && !excluded;
  }
  return out;
}

} // namespace tf::csg::graph
