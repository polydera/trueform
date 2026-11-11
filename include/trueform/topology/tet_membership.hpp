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
#include "../core/offset_block_buffer.hpp"
#include "../core/tetrahedra.hpp"
#include "../core/tetras.hpp"
#include "./structures/compute_face_membership.hpp"
#include "./tet_membership_like.hpp"
#include <cstddef>

namespace tf {

/// @ingroup topology_connectivity
/// @brief Maps each vertex to the tetrahedra that contain it.
///
/// The face membership one dimension up: per vertex, the tets naming it as a
/// corner, in descending tet order.
///
/// @tparam Index The integer type for vertex and tet indices.
template <typename Index>
class tet_membership
    : public tet_membership_like<offset_block_buffer<Index, Index>> {
  using base_t = tet_membership_like<offset_block_buffer<Index, Index>>;

public:
  tet_membership() = default;

  /// @brief Construct and build from a tetrahedra range.
  template <typename Policy>
  tet_membership(const tf::tetrahedra<Policy> &tetrahedra) {
    build(tetrahedra);
  }

  /// @brief Build from the tets' corner blocks over `n_unique_ids` vertices.
  template <typename Policy>
  auto build(const tf::tetras<Policy> &tetras, std::size_t n_unique_ids)
      -> void {
    base_t::offsets_buffer().allocate(n_unique_ids + 1);
    base_t::data_buffer().allocate(4 * tetras.size());
    topology::compute_face_membership(tetras, base_t::offsets_buffer(),
                                      base_t::data_buffer());
  }

  /// @brief Build from a tetrahedra range over its own points.
  template <typename Policy>
  auto build(const tf::tetrahedra<Policy> &tetrahedra) -> void {
    return build(tetrahedra.tetras(), tetrahedra.points().size());
  }
};

} // namespace tf
