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

#include "carriers.hpp"
#include "nd_array.hpp"

#include "trueform/core/buffer.hpp"
#include "trueform/core/polygons_buffer.hpp"
#include "trueform/core/static_size.hpp"

#include <cstddef>
#include <utility>

namespace tf::cpp::test {

/// @brief The mixed-arity mesh beside a triangle one of the same axes.
///
/// A fixture that states both layouts states two types, because the layout is
/// the geometry's own.
template <typename Owned>
using mixed_mesh_of =
    owned_mesh<typename Owned::index_type, typename Owned::real_type,
               Owned::dims, tf::dynamic_size>;

/// @brief The same mesh's axes at a stated arity.
template <typename Owned, std::size_t Ngon>
using mesh_at = owned_mesh<typename Owned::index_type,
                           typename Owned::real_type, Owned::dims, Ngon>;

/// @brief The packed face indices of a mesh, whatever its layout states them
/// with: the storage is flat at both arities.
template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto face_indices_of(const tf::polygons_buffer<Index, Real, Dims, Ngon> &value) {
  const auto &indices = value.faces_buffer().data_buffer();
  return copied_nd_array(indices, {static_cast<int>(indices.size())});
}

/// @brief The offsets a mesh's faces are blocked by: a triangle mesh states
/// i * 3 by construction, and a mixed one states its own.
template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto face_offsets_of(const tf::polygons_buffer<Index, Real, Dims, Ngon> &value) {
  if constexpr (Ngon == 3) {
    tf::buffer<Index> offsets;
    offsets.allocate(value.faces_buffer().size() + 1);
    for (std::size_t face = 0; face != offsets.size(); ++face)
      offsets[face] = static_cast<Index>(face * 3);
    const auto count = static_cast<int>(offsets.size());
    return tf::cpp::nd_array<Index>::from_buffer(std::move(offsets), {count});
  } else {
    const auto &offsets = value.faces_buffer().offsets_buffer();
    return copied_nd_array(offsets, {static_cast<int>(offsets.size())});
  }
}

} // namespace tf::cpp::test
