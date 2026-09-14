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

#include "trueform/core/algorithm/parallel_contains.hpp"
#include "trueform/core/algorithm/parallel_copy.hpp"
#include "trueform/core/buffer.hpp"
#include "trueform/core/checked.hpp"
#include "trueform/core/range.hpp"
#include "trueform/core/small_vector.hpp"
#include "trueform/core/views/slide_range.hpp"
#include "trueform/cpp/core/nd_array.hpp"
#include "trueform/cpp/core/offset_blocked_buffer.hpp"
#include "trueform/topology/manifold_edge_peer.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

namespace tf::cpp::carrier {

/// The shapes a carrier's door accepts, and the two ways a cache hands its own
/// storage back: copied, when what it hands out is an identity it validated, or
/// borrowed, when the storage is the caller's own value to keep alive.

template <std::size_t Columns, typename T>
auto require_columns(const nd_array<T> &array, const char *name) -> void {
  if (!array.is_valid() || array.ndim() != 2 ||
      array.shape_at(1) != static_cast<int>(Columns))
    throw std::invalid_argument(std::string(name) + " must have shape [N, " +
                                std::to_string(Columns) + "]");
}

/// @brief What a caller that has neither count states in its place.
inline constexpr int any_count = -1;

/// @brief The shape a per-element cache must have to be this carrier's.
///
/// The one producer of that fact for every carrier that seals a cache and
/// every entry handed one. An offset-blocked buffer states its own CSR at
/// construction but hands its offsets back in shared, writable storage, so the
/// blocks a caller presents here are read again rather than assumed.
///
/// The caller adds what only it knows: how many blocks it expects, what a
/// value may name, and how small the smallest block may be. It reads back the
/// block count, which is what the offsets state when the caller has no
/// expectation of its own.
template <typename Index>
auto require_offset_blocks(const offset_blocked_buffer<Index, Index> &blocks,
                           int block_count, int value_count, const char *name,
                           Index minimum_block_size = Index{0}) -> std::size_t {
  if (!blocks.is_valid())
    throw std::invalid_argument(std::string(name) + " must be valid");
  const auto offsets = blocks.offsets();
  const auto values = blocks.data();
  if (block_count != any_count &&
      offsets.length() != static_cast<std::size_t>(block_count) + 1)
    throw std::invalid_argument(std::string(name) +
                                " must state one block per element");
  if (offsets.length() == 0 || offsets[0] != Index{0})
    throw std::invalid_argument(std::string(name) +
                                " offsets must start at zero");
  if (offsets[offsets.length() - 1] != static_cast<Index>(values.length()))
    throw std::invalid_argument(std::string(name) +
                                " offsets must end at the data length");
  const auto too_small = tf::parallel_contains(
      tf::make_slide_range<2>(offsets.make_range()),
      [minimum_block_size](const auto &block) {
        // read in this order: a block's size is what its own ends state, and
        // ends that do not stand on zero and rise have no size to compare
        return block[0] < Index{0} || block[1] < block[0] ||
               block[1] - block[0] < minimum_block_size;
      },
      tf::checked);
  if (too_small)
    throw std::invalid_argument(
        minimum_block_size > Index{0}
            ? std::string(name) + " blocks must hold at least " +
                  std::to_string(minimum_block_size) + " values"
            : std::string(name) + " offsets must be nondecreasing");
  if (value_count != any_count) {
    const auto limit = static_cast<Index>(value_count);
    const auto refused = tf::parallel_contains(
        values.make_range(),
        [limit](Index value) { return value < Index{0} || value >= limit; },
        tf::checked);
    if (refused)
      throw std::out_of_range(std::string(name) + " index out of range");
  }
  return offsets.length() - 1;
}

/// @brief What a manifold edge link's values may name.
///
/// A peer is a face of this reading, or one of the three things the link says
/// in place of a face: the edge is on the boundary, it is non-manifold, or it
/// is the representative of its non-manifold class. Every other integer names a
/// face the reading does not have, and a read through it walks memory that is
/// not this mesh's.
template <typename Index, typename Values>
auto require_face_peers(const Values &values, int face_count, const char *name)
    -> void {
  const auto limit = static_cast<Index>(face_count);
  const auto refused = tf::parallel_contains(
      values,
      [limit](Index value) {
        return value <
                   tf::manifold_edge_peer<Index>::non_manifold_representative ||
               value >= limit;
      },
      tf::checked);
  if (refused)
    throw std::out_of_range(std::string(name) + " names a face out of range");
}

template <typename T> auto copied(const tf::buffer<T> &source) -> nd_array<T> {
  tf::buffer<T> out;
  out.allocate(source.size());
  tf::parallel_copy(source, out);
  return nd_array<T>::from_buffer(std::move(out));
}

template <typename T>
auto borrowed(const tf::buffer<T> &source, tf::small_vector<int, 3> shape)
    -> nd_array<T> {
  return nd_array<T>::from_borrowed({}, const_cast<T *>(source.data()),
                                    source.size(), std::move(shape));
}

} // namespace tf::cpp::carrier
