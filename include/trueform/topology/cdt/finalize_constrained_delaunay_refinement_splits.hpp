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
#include "../cdt_constraint_split.hpp"
#include "./size_adaptive.hpp"
#include <cstddef>
#include <cstdint>
#include <utility>

namespace tf::topology::cdt {

template <typename Owner>
auto finalize_constrained_delaunay_refinement_splits(Owner &owner) -> void {
  using Index = typename Owner::index_type;

  tf::topology::cdt::sort_auto(
      owner._splits.begin(), owner._splits.end(),
      [](const tf::cdt_constraint_split<Index> &a,
         const tf::cdt_constraint_split<Index> &b) {
        if (a.edge != b.edge)
          return a.edge < b.edge;
        return (std::uint32_t(a.numerator)
                << (Owner::max_split_depth + 1 - a.depth)) <
               (std::uint32_t(b.numerator)
                << (Owner::max_split_depth + 1 - b.depth));
      });
  const bool broadcast = owner._constraint_alias_blocks.size() != 0;
  auto &offsets = broadcast ? owner._representative_split_offsets
                            : owner._split_offsets;
  offsets.allocate(std::size_t(owner._n_input_edges) + 1);
  std::size_t write = 0;
  for (Index edge = 0; edge <= owner._n_input_edges; ++edge) {
    offsets[std::size_t(edge)] = Index(write);
    while (write < owner._splits.size() && edge < owner._n_input_edges &&
           owner._splits[write].edge == edge)
      ++write;
  }
  if (!broadcast)
    return;

  std::size_t total = 0;
  for (Index input = 0; input < owner._n_input_edges; ++input) {
    const auto representative =
        owner._constraint_split_parent[std::size_t(input)];
    if (representative != Owner::none)
      total += std::size_t(
          owner._representative_split_offsets[std::size_t(representative) + 1] -
          owner._representative_split_offsets[std::size_t(representative)]);
  }
  owner._expanded_splits.allocate(total);
  owner._split_offsets.allocate(std::size_t(owner._n_input_edges) + 1);
  std::size_t out = 0;
  for (Index input = 0; input < owner._n_input_edges; ++input) {
    owner._split_offsets[std::size_t(input)] = Index(out);
    const auto representative =
        owner._constraint_split_parent[std::size_t(input)];
    if (representative == Owner::none)
      continue;
    const auto begin = std::size_t(
        owner._representative_split_offsets[std::size_t(representative)]);
    const auto end = std::size_t(
        owner._representative_split_offsets[std::size_t(representative) + 1]);
    if (!owner._constraint_split_reversed[std::size_t(input)]) {
      for (auto row = begin; row < end; ++row) {
        auto split = owner._splits[row];
        split.edge = input;
        owner._expanded_splits[out++] = split;
      }
    } else {
      for (auto row = end; row > begin; --row) {
        auto split = owner._splits[row - 1];
        split.edge = input;
        split.numerator = std::uint8_t(
            (std::uint32_t(1) << split.depth) - split.numerator);
        owner._expanded_splits[out++] = split;
      }
    }
  }
  owner._split_offsets[std::size_t(owner._n_input_edges)] = Index(out);
  std::swap(owner._splits, owner._expanded_splits);
}

} // namespace tf::topology::cdt
