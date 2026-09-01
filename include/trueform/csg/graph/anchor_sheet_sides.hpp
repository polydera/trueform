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
#include "../../core/algorithm/parallel_fill.hpp"
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../core/views/sequence_range.hpp"
#include "./arrangement_descriptor.hpp"
#include "./domain_inclusions.hpp"
#include <array>
#include <cstddef>
#include <cstdint>

namespace tf::csg::graph {

/// @ingroup csg_graph_internals
/// @brief Anchor sheet-form bits to the sheets' orientation.
///
/// The inclusion flood keeps a sheet's bit consistent across its
/// fragments (crossing one flips it), but which side carries the set
/// bit is an accident of the flood tree. Normalise per connected
/// region of the domain graph and per sheet tag: the representative
/// fragment's side-1 domain (the interior side by stored winding —
/// the back of the sheet's normal) must carry the set bit; if it does
/// not, flip that tag's bit across the region. Regions without
/// fragments of the tag (disconnected bundles) keep their seeded
/// value.
///
/// A sheet coplanar-folded into another component's wall has no
/// component of its own to anchor from — and its seeded value comes
/// from a winding query evaluated exactly ON the shared wall, which is
/// degenerate. `sheet_folds` supplies those anchors: each entry
/// `(component, sheet tag, reversed)` anchors the tag to the carrying
/// component's behind side, mirrored when the folded winding opposes
/// the survivor's. Own-tag fragments anchor first; a fold only decides
/// a (region, sheet) no live fragment already decided.
template <typename Index>
auto anchor_sheet_sides(
    tf::csg::graph::domain_inclusions &inc,
    const arrangement_descriptor<Index> &desc,
    const tf::buffer<char> &is_sheet_tag,
    const tf::buffer<std::array<Index, 3>> &sheet_folds = {}) -> void {
  if (is_sheet_tag.size() == 0)
    return;
  const Index n_components =
      static_cast<Index>(desc.bundle_of_component.size());
  const Index n_domains = desc.n_domains;
  const std::size_t words = inc.words_per_domain;
  if (n_components == Index(0) || n_domains == Index(0))
    return;

  // Connected regions of the domain graph (components are the edges).
  tf::buffer<Index> region;
  region.allocate(static_cast<std::size_t>(n_domains));
  for (Index d = Index(0); d < n_domains; ++d)
    region[d] = d;
  auto find = [&](Index x) {
    while (region[x] != x)
      x = region[x] = region[region[x]];
    return x;
  };
  for (Index c = Index(0); c < n_components; ++c) {
    const Index r0 = find(desc.domain_of_side[2 * c]);
    const Index r1 = find(desc.domain_of_side[2 * c + 1]);
    if (r0 != r1)
      region[r1] = r0;
  }

  // Per (region, sheet): the smallest separating fragment decides.
  // `decided` is dense over (region, sheet ordinal) — sheets, not tags,
  // so it does not scale with the operand count. `flip` carries the same
  // rows as `inc`, holding the tags to xor across each region root.
  tf::buffer<Index> sheet_ordinal;
  sheet_ordinal.allocate(is_sheet_tag.size());
  Index n_sheets = Index(0);
  for (std::size_t t = 0; t < is_sheet_tag.size(); ++t)
    sheet_ordinal[t] = is_sheet_tag[t] ? n_sheets++ : Index(-1);

  tf::csg::graph::domain_inclusions flip;
  flip.words_per_domain = words;
  flip.bits.allocate(static_cast<std::size_t>(n_domains) * words);
  tf::parallel_fill(flip.bits, std::uint32_t(0));
  tf::buffer<char> decided;
  decided.allocate(static_cast<std::size_t>(n_domains) *
                   static_cast<std::size_t>(n_sheets));
  tf::parallel_fill(decided, char(0));

  // Anchor sheet `t` through component `c`: `reversed` mirrors the
  // behind side when the folded winding opposes the carrying wall's.
  auto anchor = [&](Index c, Index t, Index reversed) {
    if (t < Index(0) || t >= Index(is_sheet_tag.size()) || !is_sheet_tag[t])
      return;
    const Index d0 = desc.domain_of_side[2 * c];
    const Index d1 = desc.domain_of_side[2 * c + 1];
    if (d0 == d1)
      return;
    const Index r = find(d0);
    const std::size_t slot =
        static_cast<std::size_t>(r) * static_cast<std::size_t>(n_sheets) +
        static_cast<std::size_t>(sheet_ordinal[t]);
    if (decided[slot])
      return;
    decided[slot] = char(1);
    const Index behind = reversed ? d0 : d1;
    if (inc.test(static_cast<std::size_t>(behind), static_cast<std::size_t>(t)))
      return;
    flip.set(static_cast<std::size_t>(r), static_cast<std::size_t>(t));
  };

  for (Index c = Index(0); c < n_components; ++c)
    anchor(c, desc.tag_of_component[c], Index(0));
  for (const auto &f : sheet_folds)
    anchor(f[0], f[1], f[2]);

  // Flatten roots first so the apply pass reads `region` without
  // mutating it.
  for (Index d = Index(0); d < n_domains; ++d)
    region[d] = find(d);

  tf::parallel_for_each(tf::make_sequence_range(n_domains), [&](Index d) {
    const Index r = region[d];
    for (std::size_t w = 0; w < words; ++w)
      inc.bits[static_cast<std::size_t>(d) * words + w] ^=
          flip.bits[static_cast<std::size_t>(r) * words + w];
  });
}

} // namespace tf::csg::graph
