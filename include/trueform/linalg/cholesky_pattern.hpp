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
#include "../core/buffer.hpp"
#include <cstddef>

namespace tf {

/// @ingroup linalg
/// @brief What a sparse Cholesky factor's structure is, read once from a
/// @ref tf::symmetric_lower pattern and kept for every value set it carries.
///
/// `adjacency` is the matrix's own graph, each row's neighbours at
/// `adjacency_offsets` with `adjacency_entries` naming the input entry each
/// stands for (@ref tf::linalg::make_symmetric_adjacency): the order is read
/// from it, and a value set's true residual is gathered through it.
///
/// The elimination order is the approximate minimum degree order composed
/// with a postorder of its elimination tree, so every subtree is a contiguous
/// run of columns. `lower` is the matrix under that order with column `k`
/// holding its diagonal and then its rows `i > k`, and `lower_slot` is the
/// ticket from an input entry to its slot there.
///
/// The factor's columns are cut into supernodes, contiguous runs that
/// `supernode_offsets` partitions. Supernode `s` holds the rows
/// `rows[row_offsets[s]..row_offsets[s + 1])`, ascending, its own columns
/// first, and its block of the factor stands column-major from
/// `value_offsets[s]`, one column per own column and one row per row, the
/// triangle above its diagonal zero. `supernode_parent` is the tree the
/// supernodes form, -1 at a root, itself postordered. `schedule` lists the
/// supernodes the way the numeric factor takes them: the runs of the
/// subtrees that factor apart, partitioned by `block_offsets`, and then the
/// frontier above them, ascending from @ref frontier_first. The schedule is
/// not a numerical choice: every partition factors to the same bits.
///
/// @tparam Index The index type rows and entries are numbered in.
template <typename Index> struct cholesky_pattern {
  tf::buffer<Index> adjacency_offsets;
  tf::buffer<Index> adjacency;
  tf::buffer<Index> adjacency_entries;
  /// The eliminated row at each position: new to old.
  tf::buffer<Index> order;
  /// The position each row is eliminated at: old to new.
  tf::buffer<Index> rank;
  tf::buffer<Index> lower_offsets;
  tf::buffer<Index> lower_rows;
  tf::buffer<Index> lower_slot;
  tf::buffer<Index> supernode_offsets;
  tf::buffer<Index> supernode_parent;
  tf::buffer<Index> row_offsets;
  tf::buffer<Index> rows;
  tf::buffer<Index> value_offsets;
  tf::buffer<Index> schedule;
  tf::buffer<Index> block_offsets;

  auto size() const -> std::size_t { return order.size(); }
  auto number_of_supernodes() const -> std::size_t {
    return supernode_offsets.size() - 1;
  }
  auto number_of_blocks() const -> std::size_t {
    return block_offsets.size() - 1;
  }
  /// Where the frontier starts in the schedule.
  auto frontier_first() const -> std::size_t {
    return std::size_t(block_offsets.back());
  }
};

} // namespace tf
