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
#include "./compute_domain_inclusions.hpp"
#include <cstdint>

namespace tf::cut {

/// @ingroup cut
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
template <typename Index>
auto anchor_sheet_sides(tf::cut::domain_inclusions &inc,
                        const arrangement_descriptor<Index> &desc,
                        const tf::buffer<char> &is_sheet_tag) -> void {
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
  // so it does not scale with the operand count. flip_mask[region-root]
  // accumulates the per-tag xor words.
  tf::buffer<Index> sheet_ordinal;
  sheet_ordinal.allocate(is_sheet_tag.size());
  Index n_sheets = Index(0);
  for (std::size_t t = 0; t < is_sheet_tag.size(); ++t)
    sheet_ordinal[t] = is_sheet_tag[t] ? n_sheets++ : Index(-1);

  tf::buffer<std::uint32_t> flip_mask;
  flip_mask.allocate(static_cast<std::size_t>(n_domains) * words);
  tf::parallel_fill(flip_mask, std::uint32_t(0));
  tf::buffer<char> decided;
  decided.allocate(static_cast<std::size_t>(n_domains) *
                   static_cast<std::size_t>(n_sheets));
  tf::parallel_fill(decided, char(0));

  for (Index c = Index(0); c < n_components; ++c) {
    const Index t = desc.tag_of_component[c];
    if (t < Index(0) || t >= Index(is_sheet_tag.size()) || !is_sheet_tag[t])
      continue;
    const Index d0 = desc.domain_of_side[2 * c];
    const Index d1 = desc.domain_of_side[2 * c + 1];
    if (d0 == d1)
      continue;
    const Index r = find(d0);
    const std::size_t slot =
        static_cast<std::size_t>(r) * static_cast<std::size_t>(n_sheets) +
        static_cast<std::size_t>(sheet_ordinal[t]);
    if (decided[slot])
      continue;
    decided[slot] = char(1);
    const std::size_t word = static_cast<std::size_t>(t) / 32u;
    const std::uint32_t mask = std::uint32_t(1)
                               << (static_cast<unsigned>(t) % 32u);
    const bool interior_set =
        (inc.bits[static_cast<std::size_t>(d1) * words + word] & mask) != 0u;
    if (!interior_set)
      flip_mask[static_cast<std::size_t>(r) * words + word] |= mask;
  }

  // Flatten roots first so the apply pass reads `region` without
  // mutating it.
  for (Index d = Index(0); d < n_domains; ++d)
    region[d] = find(d);

  tf::parallel_for_each(tf::make_sequence_range(n_domains), [&](Index d) {
    const Index r = region[d];
    for (std::size_t w = 0; w < words; ++w)
      inc.bits[static_cast<std::size_t>(d) * words + w] ^=
          flip_mask[static_cast<std::size_t>(r) * words + w];
  });
}

} // namespace tf::cut
