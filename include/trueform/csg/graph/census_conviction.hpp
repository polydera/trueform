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
#include "../../core/algorithm/parallel_copy.hpp"
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/algorithm/reduce.hpp"
#include "../../core/algorithm/sequenced_generate.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/range.hpp"
#include "../../core/views/mapped_range.hpp"
#include "../../core/views/sequence_range.hpp"
#include "./find_census_crossings.hpp"
#include "./make_live_near_identities.hpp"
#include "./make_self_sided_rings.hpp"
#include "./ring_domain_parity_is_odd.hpp"
#include <array>
#include <cstddef>

namespace tf::csg::graph {

/// The census's verdict on a classified graph, three facts that must all
/// be zero:
/// - `failed`: carriers the arrangement left without a product;
/// - `odd`: radial rings whose domains do not pair;
/// - `self_sided`: closed volume components whose two sides are one
///   domain, unless every stack layer crossing them nets to zero — the
///   coincident walls that cancel are not a contradiction.
/// What the sites of a fusion are read from (@ref collect_census_id_pairs):
/// `odd_rings`, the fans whose rings do not pair, ascending; `convicted`, the
/// self-sided components; `near_identities`, the near table as its members
/// speak now (@ref make_live_near_identities); `rounding`, the rounding
/// crossings it witnesses (@ref find_census_crossings), `crossed` their
/// count, and `reaching`, those reaching a convicted component, ascending;
/// and `sided_rings`, for each convicted component no rounding crossing
/// reaches, the rings stating its self-sidedness (@ref
/// make_self_sided_rings). The sites convict nothing, and a green verdict
/// reads none of them.
template <typename Index> struct census_conviction {
  std::size_t failed = 0;
  std::size_t odd = 0;
  std::size_t self_sided = 0;
  std::size_t crossed = 0;
  tf::buffer<char> convicted;
  tf::buffer<Index> odd_rings;
  tf::buffer<std::array<Index, 2>> sided_rings;
  tf::buffer<std::array<Index, 2>> near_identities;
  tf::buffer<std::array<Index, 4>> rounding;
  tf::buffer<Index> reaching;

  auto green() const -> bool { return failed + odd + self_sided == 0; }
};

/// The green path is two parallel passes, over the components and over the
/// rings: the sites are read only off a convicted graph.
template <typename CsgGraph>
auto make_census_conviction(const CsgGraph &g)
    -> census_conviction<typename CsgGraph::index_type> {
  using Index = typename CsgGraph::index_type;
  census_conviction<Index> out;
  out.failed = std::size_t(g.failed().size());
  const auto &d = g.descriptor();
  const auto &fans = d.fans;
  const auto eor = g.arrangement().exposed_of_row();
  const auto tl = g.labels().triangle_labels();
  const auto open = g.labels().open_component_mask();
  const auto &crossings = g.crossings();
  const auto &is_sheet = g.is_sheet();
  const auto n_components = std::size_t(d.tag_of_component.size());
  out.convicted.allocate(n_components);
  tf::parallel_for_each(
      tf::make_sequence_range(n_components),
      [&](std::size_t c) {
        out.convicted[c] = char(0);
        if (d.domain_of_side[2 * c] != d.domain_of_side[2 * c + 1])
          return;
        const auto tag = std::size_t(d.tag_of_component[c]);
        if (open[c] || (tag < is_sheet.size() && is_sheet[tag]))
          return;
        const auto row = crossings.row_of(Index(c));
        if (row != Index(-1)) {
          bool net_zero = true;
          for (const auto &layer : crossings.layers[std::size_t(row)])
            net_zero = net_zero && layer[1] == 0;
          if (net_zero)
            return;
        }
        out.convicted[c] = char(1);
      },
      tf::checked);
  const auto plus = [](std::size_t x, std::size_t y) { return x + y; };
  out.self_sided = tf::reduce(
      tf::make_mapped_range(tf::make_range(out.convicted),
                            [](char c) { return std::size_t(c != 0); }),
      plus, std::size_t(0), tf::checked);
  tf::sequenced_generate(
      tf::make_sequence_range(fans.pieces.size()), out.odd_rings,
      [&](std::size_t f, tf::buffer<Index> &rings) {
        if (d.valid[f] &&
            ring_domain_parity_is_odd(tl, fans, eor, d.domain_of_side, f))
          rings.push_back(Index(f));
      },
      tf::checked);
  out.odd = out.odd_rings.size();
  if (out.green())
    return out;

  make_live_near_identities(g.arrangement(), out.near_identities);
  find_census_crossings(g.arrangement(), out.near_identities, out.rounding);
  out.crossed = out.rounding.size();
  if (out.self_sided == 0)
    return out;
  // a crossing is the sharper evidence: a component it reaches is sited
  // there, and only the rest read the rings that state them
  tf::buffer<char> unreached;
  unreached.allocate(n_components);
  tf::parallel_copy(out.convicted, unreached);
  std::size_t n_unreached = out.self_sided;
  const auto n_exposed = Index(g.arrangement().global().exposed_tris().size());
  const auto descriptors = g.arrangement().global().exposed_descriptors();
  const auto component_of = [&](Index t) -> Index {
    if (t < n_exposed)
      return Index(tl[std::size_t(t)]);
    const auto &descriptor = descriptors[std::size_t(t - n_exposed)];
    return Index(g.labels().polygon_labels(
        Index(descriptor.tag))[std::size_t(descriptor.object)]);
  };
  for (std::size_t k = 0; k < out.rounding.size(); ++k) {
    bool reaches = false;
    for (std::size_t s = 0; s < 2; ++s) {
      const auto c = component_of(out.rounding[k][s]);
      if (c < Index(0) || !out.convicted[std::size_t(c)])
        continue;
      reaches = true;
      if (unreached[std::size_t(c)]) {
        unreached[std::size_t(c)] = char(0);
        --n_unreached;
      }
    }
    if (reaches)
      out.reaching.push_back(Index(k));
  }
  if (n_unreached != 0)
    out.sided_rings = make_self_sided_rings(g, unreached);
  return out;
}

} // namespace tf::csg::graph
