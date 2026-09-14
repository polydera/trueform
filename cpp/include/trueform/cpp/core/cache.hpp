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

#include "trueform/core/static_size.hpp"
#include "trueform/cpp/core/detail/face_blocks_array.hpp"
#include "trueform/cpp/core/index_type.hpp"
#include "trueform/cpp/core/matrix.hpp"
#include "trueform/cpp/core/mesh_geometry.hpp"
#include "trueform/cpp/core/nd_array.hpp"
#include "trueform/cpp/core/offset_blocked_buffer.hpp"
#include "trueform/spatial/aabb_tree.hpp"
#include "trueform/spatial/winding_moments.hpp"
#include "trueform/topology/half_edges.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>

namespace tf::cpp {

/// @brief What is known about a mesh's shape, remembered.
///
/// A cache computes nothing the geometry does not determine; it remembers. It
/// exists because the bookkeeping is tedious, not because the library owns
/// anything: it holds no geometry, every ask takes the reading it is to answer
/// for, and dropping the whole of it loses only time.
///
/// COHERENCE. The caller states what changed — `points_changed()` after
/// coordinates moved, `faces_changed()` after connectivity was rewired — and
/// the cache carries those two generations. Every structure stamps the pair it
/// was built for, so staleness is precise: moving a point stales the tree and
/// leaves the face membership exactly where it was. The library cannot observe
/// memory a caller owns, so an unstated mutation is undetectable, exactly as it
/// is in the templated tier. A mesh SNAPSHOTS the pair when it is assembled;
/// under `!NDEBUG` a read through a mesh older than these generations says so.
///
/// THREAD SAFETY, one sentence: a cache shared across threads — concurrent
/// async jobs included — is filled before sharing or synchronized by the
/// caller; reads of filled state are free and unlimited. Filling is lazy and
/// therefore mutates what is shared, which is the whole reason this class is
/// the one place the rule is stated; it is the same contract `tree.build()` and
/// every trueform structure carries. A fill may fill its DEPENDENCIES, so
/// warming means asking, on one thread, the top-level structures the shared
/// work will touch — the `build_*` verbs are that ask, and each builds what it
/// stands on.
///
/// THE UNIT OF SHARING is the geometry and this cache; a MESH VALUE is one
/// thread's reading of them. A mesh retains each structure the first time it is
/// asked for, so two threads reading one mesh value write its slots even over a
/// fully warm cache — a race this class cannot see, because no cache write
/// happens. Share the arrays and the cache, and assemble a mesh per thread: an
/// assembly is a handful of pointers and two stamps, which is why it is the
/// cheap half.
///
/// IT IS A VALUE, AND A COPY SHARES NOTHING: the flat structures are deep-
/// copied with the stamps they were built for, the tree, its winding moments
/// and the half edges are dropped — they hold views into their own storage, so
/// a copy builds them when it is asked — and every build count starts at zero,
/// because the copy built none of what it carries.
///
/// ITS ADDRESS IS WHAT A MESH HOLDS. A carrier borrows this cache by reference,
/// so a cache does not MOVE while a mesh assembled over it is alive: reserve
/// before filling a container of caches, hold them by handle, or assemble
/// again after the move. It is the same law the geometry obeys, stated for the
/// half of the assembly that is not an array.
///
/// The `*_build_count` accessors are a SUPPORTED OBSERVABILITY SURFACE: a
/// caller that cares whether a cache was rebuilt can ask, which is how the
/// freshness contract is stated and tested. They cost one increment per build.
template <typename Index, typename Real, std::size_t Dims = 3,
          std::size_t Ngon = 3>
class cache {
  static_assert(is_supported_index_v<Index> &&
                    std::is_same_v<Index, std::remove_cv_t<Index>>,
                "cache Index must be an unqualified supported index type");
  static_assert(Dims == 2 || Dims == 3, "cache Dims must be 2 or 3");
  static_assert(matrix_carries_v<Index, Real, Dims>,
                "this (index, real, dims) is not in the built C++ facade "
                "matrix; see TF_CPP_INDICES / TF_CPP_REALS / TF_CPP_DIMS");
  static_assert(Ngon == 3 || Ngon == tf::dynamic_size,
                "cache Ngon must be 3 or tf::dynamic_size");
  static_assert(matrix_has_ngon_v<Ngon>,
                "this face layout is not in the built C++ facade matrix; "
                "see TF_CPP_LAYOUTS");

  using geometry_type = mesh_geometry<Index, Real, Dims, Ngon>;

public:
  /// @brief A per-face-side fact in the shape this arity states it: `[F, 3]`
  /// for triangles, face-aligned offset blocks for a mixed mesh.
  ///
  /// It is the type the free `manifold_edge_link` entry produces, so what the
  /// library computed is what its door takes back.
  using face_blocks_type = detail::face_blocks_array_t<Index, Ngon>;

private:
  offset_blocked_buffer<Index, Index> _face_membership;
  face_blocks_type _manifold_edge_link;
  offset_blocked_buffer<Index, Index> _face_link;
  offset_blocked_buffer<Index, Index> _vertex_link;
  nd_array<Real> _face_normals;
  nd_array<Real> _point_normals;
  std::shared_ptr<tf::half_edges<Index>> _half_edges;
  std::shared_ptr<tf::aabb_tree<Index, Real, Dims>> _tree;
  std::shared_ptr<tf::winding_moments<Real>> _winding_moments;

  std::uint64_t _faces_generation = 1;
  std::uint64_t _points_generation = 1;

  std::uint64_t _indices_faces_stamp = 0;
  int _indices_point_count = -1;
  std::uint64_t _tree_faces_stamp = 0;
  std::uint64_t _tree_points_stamp = 0;
  /// A moment row mirrors a node row, so the moments are stamped for the TREE
  /// BUILD they were read off rather than for the generations that produced it.
  std::uint64_t _winding_moments_tree_build = 0;
  std::uint64_t _half_edges_faces_stamp = 0;
  int _half_edges_point_count = -1;
  std::uint64_t _face_membership_faces_stamp = 0;
  int _face_membership_point_count = -1;
  std::uint64_t _manifold_edge_link_faces_stamp = 0;
  std::uint64_t _face_link_faces_stamp = 0;
  std::uint64_t _vertex_link_faces_stamp = 0;
  int _vertex_link_point_count = -1;
  std::uint64_t _face_normals_faces_stamp = 0;
  std::uint64_t _face_normals_points_stamp = 0;
  std::uint64_t _point_normals_faces_stamp = 0;
  std::uint64_t _point_normals_points_stamp = 0;

  std::uint64_t _tree_build_count = 0;
  std::uint64_t _winding_moments_build_count = 0;
  std::uint64_t _half_edges_build_count = 0;
  std::uint64_t _face_membership_build_count = 0;
  std::uint64_t _manifold_edge_link_build_count = 0;
  std::uint64_t _face_link_build_count = 0;
  std::uint64_t _vertex_link_build_count = 0;
  std::uint64_t _face_normals_build_count = 0;
  std::uint64_t _point_normals_build_count = 0;

  mutable std::atomic<bool> _filling{false};

public:
  cache() = default;
  cache(const cache &other);
  cache(cache &&other) noexcept;
  auto operator=(const cache &other) -> cache &;
  auto operator=(cache &&other) noexcept -> cache &;

  /// @brief The caller states the change; the cache answers for it.
  ///
  /// The generations are the channel, and whether a caller uses it is the
  /// caller's own business: there is no observer, no callback, and nothing here
  /// detects a mutation of memory this cache cannot see.
  auto faces_changed() -> void;
  auto points_changed() -> void;
  auto faces_generation() const -> std::uint64_t;
  auto points_generation() const -> std::uint64_t;

  /// @brief Every structure is handed back RETAINED.
  ///
  /// A rebuild replaces the whole structure, so a reader that keeps views into
  /// its arrays keeps the structure itself: a handle shares the storage this
  /// one stands on and outlives whatever another reading asks for next.
  auto tree_handle(const geometry_type &geometry)
      -> std::shared_ptr<tf::aabb_tree<Index, Real, Dims>>;
  /// @brief The winding expansion of the tree's nodes, built over the tree it
  /// mirrors and stale the moment that tree is built again.
  auto winding_moments_handle(const geometry_type &geometry)
      -> std::shared_ptr<tf::winding_moments<Real>>;
  auto half_edges_handle(const geometry_type &geometry)
      -> std::shared_ptr<tf::half_edges<Index>>;
  auto face_membership_handle(const geometry_type &geometry)
      -> offset_blocked_buffer<Index, Index>;
  auto manifold_edge_link_handle(const geometry_type &geometry)
      -> face_blocks_type;
  auto face_link_handle(const geometry_type &geometry)
      -> offset_blocked_buffer<Index, Index>;
  auto vertex_link_handle(const geometry_type &geometry)
      -> offset_blocked_buffer<Index, Index>;
  /// @brief One unit normal per face, `[F, 3]`, and one per point, `[P, 3]`.
  ///
  /// A normal is WHERE the points stand as much as which points a face names,
  /// so a stated move stales both of them where it leaves the connectivity
  /// structures exactly where they were. They are read off the stored points;
  /// a mesh's frame is not applied.
  auto face_normals_handle(const geometry_type &geometry) -> nd_array<Real>;
  auto point_normals_handle(const geometry_type &geometry) -> nd_array<Real>;

  /// @brief Refuse what raw storage alone states wrongly.
  ///
  /// The offsets a mixed mesh states, the arity of a face, the alignment of the
  /// coordinates, and every index naming a point this reading has. It is the
  /// one door, remembered per reading, so asking again costs a stamp check and
  /// no body carries a scan of its own.
  auto require_indices(const geometry_type &geometry) -> void;

  /// @brief State a structure a caller already holds.
  ///
  /// It is stamped for the reading it is stated against, exactly as a built one
  /// is, and refused where its shape cannot describe that reading.
  auto set_half_edges(tf::half_edges<Index> &&half_edges,
                      const geometry_type &geometry) -> void;
  auto set_face_membership(offset_blocked_buffer<Index, Index> face_membership,
                           const geometry_type &geometry) -> void;
  auto set_manifold_edge_link(face_blocks_type manifold_edge_link,
                              const geometry_type &geometry) -> void;
  auto set_face_link(offset_blocked_buffer<Index, Index> face_link,
                     const geometry_type &geometry) -> void;
  auto set_vertex_link(offset_blocked_buffer<Index, Index> vertex_link,
                       const geometry_type &geometry) -> void;

  auto is_tree_built() const -> bool;
  auto is_tree_fresh(const geometry_type &geometry) const -> bool;
  auto is_winding_moments_built() const -> bool;
  auto is_winding_moments_fresh(const geometry_type &geometry) const -> bool;
  auto is_half_edges_built() const -> bool;
  auto is_half_edges_fresh(const geometry_type &geometry) const -> bool;
  auto is_face_membership_built() const -> bool;
  auto is_face_membership_fresh(const geometry_type &geometry) const -> bool;
  auto is_manifold_edge_link_built() const -> bool;
  auto is_manifold_edge_link_fresh(const geometry_type &geometry) const -> bool;
  auto is_face_link_built() const -> bool;
  auto is_face_link_fresh(const geometry_type &geometry) const -> bool;
  auto is_vertex_link_built() const -> bool;
  auto is_vertex_link_fresh(const geometry_type &geometry) const -> bool;
  auto is_face_normals_built() const -> bool;
  auto is_face_normals_fresh(const geometry_type &geometry) const -> bool;
  auto is_point_normals_built() const -> bool;
  auto is_point_normals_fresh(const geometry_type &geometry) const -> bool;

  auto tree_build_count() const -> std::uint64_t;
  auto winding_moments_build_count() const -> std::uint64_t;
  auto half_edges_build_count() const -> std::uint64_t;
  auto face_membership_build_count() const -> std::uint64_t;
  auto manifold_edge_link_build_count() const -> std::uint64_t;
  auto face_link_build_count() const -> std::uint64_t;
  auto vertex_link_build_count() const -> std::uint64_t;
  auto face_normals_build_count() const -> std::uint64_t;
  auto point_normals_build_count() const -> std::uint64_t;

private:
  auto copy_from(const cache &other) -> void;
  auto move_from(cache &&other) -> void;
  /// A mesh states the generations it was assembled over; a mesh that states
  /// others is reading through arrays this cache can no longer answer for.
  auto assert_is_current(const geometry_type &geometry) const -> void;
  auto ensure_indices(const geometry_type &geometry) -> void;
  auto ensure_tree(const geometry_type &geometry) -> void;
  auto ensure_winding_moments(const geometry_type &geometry) -> void;
  auto ensure_half_edges(const geometry_type &geometry) -> void;
  auto ensure_face_membership(const geometry_type &geometry)
      -> const offset_blocked_buffer<Index, Index> &;
  auto ensure_manifold_edge_link(const geometry_type &geometry) -> void;
  auto ensure_face_link(const geometry_type &geometry) -> void;
  auto ensure_vertex_link(const geometry_type &geometry) -> void;
  auto ensure_face_normals(const geometry_type &geometry) -> void;
  auto ensure_point_normals(const geometry_type &geometry) -> void;
};

#define TF_CPP_EXTERN_CACHE(Index, Real, Dims, Ngon)                           \
  extern template class cache<Index, Real, Dims, Ngon>

TF_CPP_MATRIX_FOR_EACH_INDEX_REAL_DIMS_NGON(TF_CPP_EXTERN_CACHE)

#undef TF_CPP_EXTERN_CACHE

} // namespace tf::cpp
