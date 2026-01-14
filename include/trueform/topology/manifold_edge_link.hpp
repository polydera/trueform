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
#include "../core/algorithm/parallel_transform.hpp"
#include "../core/blocked_buffer.hpp"
#include "../core/faces.hpp"
#include "../core/offset_block_buffer.hpp"
#include "../core/views/drop.hpp"
#include "../core/views/slide_range.hpp"
#include "./manifold_edge_link_like.hpp"
#include "./structures/compute_manifold_edge_link.hpp"

namespace tf {

/// @ingroup topology_connectivity
/// @brief Stores edge connectivity for manifold meshes with fixed face size.
///
/// For each directed edge in each face, stores a @ref tf::manifold_edge_peer
/// indicating the peer face that shares that edge. This is the foundation
/// for boundary detection, face orientation, and mesh traversal.
///
/// Use the fixed-size specialization (`N != dynamic_size`) for meshes with
/// constant polygon size (e.g., triangles), or the dynamic specialization
/// for variable-size polygons.
///
/// @tparam Index The integer type for face indices.
/// @tparam N The number of edges per face (use @ref tf::dynamic_size for variable).
template <typename Index, std::size_t N>
class manifold_edge_link : public manifold_edge_link_like<
                               blocked_buffer<manifold_edge_peer<Index>, N>> {
  using base_t =
      manifold_edge_link_like<blocked_buffer<manifold_edge_peer<Index>, N>>;

public:
  manifold_edge_link() = default;

  /// @brief Construct and build from faces and face membership.
  /// @tparam Policy The faces policy type.
  /// @tparam Policy1 The face membership policy type.
  /// @param faces The faces range.
  /// @param blink The face membership structure.
  template <typename Policy, typename Policy1>
  manifold_edge_link(const tf::faces<Policy> &faces,
                     const tf::face_membership_like<Policy1> &blink) {
    build(faces, blink);
  }

  /// @brief Build edge connectivity from face blocks and face membership.
  /// @tparam Range The face blocks range type.
  /// @tparam Policy1 The face membership policy type.
  /// @param blocks The face blocks range.
  /// @param blink The face membership structure.
  template <typename Range, typename Policy1>
  auto build(const Range &blocks,
             const tf::face_membership_like<Policy1> &blink) -> void {
    base_t::data_buffer().allocate(blocks.size() * N);
    topology::compute_manifold_edge_link(blocks, blink, *this);
  }
};

/// @ingroup topology_connectivity
/// @brief Stores edge connectivity for meshes with variable face sizes.
///
/// Specialization for variable-size polygons. Uses @ref tf::offset_block_buffer
/// to store varying numbers of edges per face.
///
/// @tparam Index The integer type for face indices.
template <typename Index>
class manifold_edge_link<Index, tf::dynamic_size>
    : public manifold_edge_link_like<
          offset_block_buffer<Index, manifold_edge_peer<Index>>> {
  using base_t = manifold_edge_link_like<
      offset_block_buffer<Index, manifold_edge_peer<Index>>>;

public:
  manifold_edge_link() = default;

  /// @brief Construct and build from faces and face membership.
  /// @tparam Policy The faces policy type.
  /// @tparam Policy1 The face membership policy type.
  /// @param faces The faces range.
  /// @param blink The face membership structure.
  template <typename Policy, typename Policy1>
  manifold_edge_link(const tf::faces<Policy> &faces,
                     const tf::face_membership_like<Policy1> &blink) {
    build(faces, blink);
  }

  /// @brief Build edge connectivity from face blocks and face membership.
  /// @tparam Range The face blocks range type.
  /// @tparam Policy1 The face membership policy type.
  /// @param blocks The face blocks range.
  /// @param blink The face membership structure.
  template <typename Range, typename Policy1>
  auto build(const Range &blocks,
             const tf::face_membership_like<Policy1> &blink) -> void {
    if (!blocks.size())
      return;
    base_t::offsets_buffer().allocate(blocks.size() + 1);
    base_t::offsets_buffer()[0] = 0;
    tf::parallel_transform(
        blocks, tf::drop(base_t::offsets_buffer(), 1),
        [](const auto &block) { return block.size(); }, tf::checked);
    for (auto &&[a, b] : tf::make_slide_range<2>(base_t::offsets_buffer()))
      b += a;
    base_t::data_buffer().allocate(base_t::offsets_buffer().back());
    topology::compute_manifold_edge_link(blocks, blink, *this);
  }
};

} // namespace tf
