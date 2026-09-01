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
#include "../../core/edges.hpp"
#include "../../core/range.hpp"
#include "./append_constrained_delaunay_full_span_alias.hpp"
#include "./order_constrained_delaunay_full_span_aliases.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

namespace tf::topology::cdt {

template <typename Owner, typename EdgesPolicy, typename SplittableIterator,
          std::size_t SplittableN>
auto make_constrained_delaunay_refinement_constraints(
    Owner &owner, const tf::edges<EdgesPolicy> &edges,
    const tf::range<SplittableIterator, SplittableN> &is_splittable) -> void {
  using Index = typename Owner::index_type;

  // One speculative sub-edge per input constraint keeps the ordinary path
  // in input space. Only an actual alias block rebuilds physical segments.
  const auto &im = owner._cdt.index_map().f();
  auto &aliases = owner._constraint_aliases;
  aliases.reserve(std::size_t(edges.size()));
  owner._segments.reserve(std::size_t(edges.size()));
  const auto hook = [&](Index x, Index y) {
    if (owner._con_nbr[std::size_t(x)][0] == Owner::none)
      owner._con_nbr[std::size_t(x)][0] = y;
    else
      owner._con_nbr[std::size_t(x)][1] = y;
  };
  for (Index j = 0; j < owner._n_input_edges; ++j) {
    const Index u = im[std::size_t(edges[j][0])];
    const Index v = im[std::size_t(edges[j][1])];
    if (u != v) {
      append_constrained_delaunay_full_span_alias(u, v, j, aliases);
      hook(u, v);
      hook(v, u);
    }
    owner._segments.push_back(
        {j, u, 0,
         is_splittable[std::size_t(j)]
             ? std::uint8_t(0)
             : std::uint8_t(Owner::max_split_depth)});
  }
  order_constrained_delaunay_full_span_aliases(
      aliases, owner._constraint_alias_blocks);

  const bool broadcast = owner._constraint_alias_blocks.size() != 0;
  if (!broadcast)
    return;

  owner._segments.clear();
  std::fill(owner._con_nbr.begin(), owner._con_nbr.end(),
            std::array<Index, 2>{Owner::none, Owner::none});
  owner._constraint_split_parent.allocate(
      std::size_t(owner._n_input_edges));
  std::fill(owner._constraint_split_parent.begin(),
            owner._constraint_split_parent.end(), Owner::none);
  owner._constraint_split_reversed.allocate(
      std::size_t(owner._n_input_edges));
  std::fill(owner._constraint_split_reversed.begin(),
            owner._constraint_split_reversed.end(), char(0));

  std::size_t write = 0;
  std::size_t block = 0;
  for (std::size_t begin = 0; begin < aliases.size();) {
    std::size_t end = begin + 1;
    if (block < owner._constraint_alias_blocks.size() &&
        std::size_t(owner._constraint_alias_blocks[block][0]) == begin) {
      end = std::size_t(owner._constraint_alias_blocks[block][1]);
      ++block;
    }
    const auto representative = aliases[begin];
    const auto representative_lo =
        representative.reversed ? representative.second : representative.first;
    bool split = false;
    for (auto row = begin; row < end; ++row)
      split = split || bool(is_splittable[std::size_t(aliases[row].input)]);
    owner._segments.push_back(
        {representative.input, representative_lo, 0,
         split ? std::uint8_t(0)
               : std::uint8_t(Owner::max_split_depth)});
    hook(representative.first, representative.second);
    hook(representative.second, representative.first);
    for (auto row = begin; row < end; ++row) {
      const auto &alias = aliases[row];
      owner._constraint_split_parent[std::size_t(alias.input)] =
          representative.input;
      owner._constraint_split_reversed[std::size_t(alias.input)] =
          char(alias.reversed != representative.reversed);
    }
    aliases[write++] = representative;
    begin = end;
  }
  aliases.erase_till_end(aliases.begin() + std::ptrdiff_t(write));
}

} // namespace tf::topology::cdt
