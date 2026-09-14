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

#include "trueform/core/policy/frame.hpp"
#include "trueform/core/polygons.hpp"
#include "trueform/core/static_size.hpp"
#include "trueform/core/transformation_view.hpp"
#include "trueform/core/unit_vectors.hpp"
#include "trueform/core/views/blocked_range.hpp"
#include "trueform/core/views/mapped_range.hpp"
#include "trueform/core/views/offset_block_range.hpp"
#include "trueform/cpp/core/cache.hpp"
#include "trueform/cpp/core/detail/reading_views.hpp"
#include "trueform/cpp/core/identity_transformation.hpp"
#include "trueform/cpp/core/index_type.hpp"
#include "trueform/cpp/core/matrix.hpp"
#include "trueform/cpp/core/mesh_geometry.hpp"
#include "trueform/cpp/core/nd_array.hpp"
#include "trueform/cpp/core/offset_blocked_buffer.hpp"
#include "trueform/spatial/aabb_tree.hpp"
#include "trueform/spatial/policy/tree.hpp"
#include "trueform/spatial/winding_moments.hpp"
#include "trueform/topology/face_link_like.hpp"
#include "trueform/topology/face_membership_like.hpp"
#include "trueform/topology/half_edges.hpp"
#include "trueform/topology/manifold_edge_link_like.hpp"
#include "trueform/topology/manifold_edge_peer.hpp"
#include "trueform/topology/policy/face_membership.hpp"
#include "trueform/topology/policy/manifold_edge_link.hpp"
#include "trueform/topology/vertex_link_like.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>

namespace tf::cpp {

/// @brief A mesh: geometry a caller holds, a cache that remembers it, and this
/// instance's frame.
///
/// THE ASSEMBLY is the entrance and the only one: faces and points as views
/// into memory the CALLER owns, a reference to a cache the CALLER holds, and
/// optionally the frame this instance is placed by. Ownership is never this
/// library's business — core's `polygons_buffer` is the owner when a caller
/// wants one, and VTK's arrays or a mapping are as good.
///
/// INSTANCING FALLS OUT OF IT: N meshes over one geometry and one cache, each
/// with its own frame. A cache is built in local coordinates and the frame is a
/// tag, so the instances share the tree.
///
/// It is a value. The arity is in the type, the frame is always there
/// (identity when none is given), and one type-erased slot retains whatever
/// must outlive the caller's own handle.
///
/// It YIELDS tagged forms; it never is one. A body writes
/// `mesh.polygons() | tag(tree) | tag(face_membership) | tag(...) |
/// tag(frame)` in that order, which is the order every operand in trueform is
/// tagged in, so one (index, real, dims, arity) has exactly one form type.
///
/// Every structure is asked of the cache FOR THIS READING, so they answer for
/// the geometry this mesh holds and not for whatever the caller holds by now —
/// and a body that never asks pays for nothing. Each is RETAINED the first time
/// it is asked for; see the slots below.
///
/// A mesh is ONE READING OF THE GEOMETRY, and nothing more, and it BORROWS it —
/// the arrays and the cache alike: both outlive the mesh, which is the law
/// every view in trueform obeys. It snapshots the cache's generations when it
/// is assembled, so a caller that states a change assembles again.
///
/// ONE READING IS ONE THREAD'S. The slots below are filled by the first ask, so
/// two threads reading one mesh VALUE write them concurrently even over a fully
/// warm cache — the cache is untouched there, so nothing detects it. What is
/// shared between threads is the geometry and the cache; each thread assembles
/// its own mesh over them, which is a handful of pointers and two stamps.
///
/// A FORM OUTLIVES NOTHING ITS MESH HOLDS. The form is a core type and keeps
/// no handle, so the mesh it was taken from stays alive for as long as it is
/// read — bind the mesh, then take the form.
template <typename Index, typename Real, std::size_t Dims = 3,
          std::size_t Ngon = 3>
class mesh {
  static_assert(is_supported_index_v<Index> &&
                    std::is_same_v<Index, std::remove_cv_t<Index>>,
                "mesh Index must be an unqualified supported index type");
  static_assert(Dims == 2 || Dims == 3, "mesh Dims must be 2 or 3");
  static_assert(matrix_carries_v<Index, Real, Dims>,
                "this (index, real, dims) is not in the built C++ facade "
                "matrix; see TF_CPP_INDICES / TF_CPP_REALS / TF_CPP_DIMS");
  static_assert(Ngon == 3 || Ngon == tf::dynamic_size,
                "mesh Ngon must be 3 or tf::dynamic_size");
  static_assert(matrix_has_ngon_v<Ngon>,
                "this face layout is not in the built C++ facade matrix; "
                "see TF_CPP_LAYOUTS");

public:
  using index_type = Index;
  using real_type = Real;
  using geometry_type = mesh_geometry<Index, Real, Dims, Ngon>;
  using faces_type = typename geometry_type::faces_type;
  using points_type = typename geometry_type::points_type;
  using frame_type = tf::transformation_view<const Real, Dims>;
  using cache_type = cpp::cache<Index, Real, Dims, Ngon>;

  using tree_type = tf::aabb_tree<Index, Real, Dims>;

  mesh() = delete;
  /// @brief The assembly.
  ///
  /// The views are the caller's, in either spelling: storage held const and
  /// storage the caller still writes through name the same reading, and this
  /// keeps the const one. A caller that moves a point writes through its own
  /// buffer and says `points_changed()`, so there is no second name for it.
  ///
  /// `keepalive` is empty when the caller's own storage is what this reads —
  /// the borrow law — and filled when the reading must outlive the caller's
  /// handle, which is how an executor or a binding is handed a mesh. It retains
  /// the geometry AND the cache, because it retains the owner of both.
  template <typename Faces, typename Points,
            std::enable_if_t<detail::reads_blocks_v<Faces, Index, Ngon> &&
                                 detail::reads_blocks_v<Points, Real, Dims>,
                             int> = 0>
  mesh(const Faces &faces, const Points &points, cache_type &cache,
       frame_type frame = identity_transformation_view<Real, Dims>(),
       std::shared_ptr<const void> keepalive = {})
      : _geometry(reading_of(faces, points, cache.faces_generation(),
                             cache.points_generation())),
        _frame(frame), _cache(&cache), _keepalive(std::move(keepalive)) {}

  auto faces() const -> faces_type { return _geometry.faces(); }
  auto points() const -> points_type { return _geometry.points(); }
  auto polygons() const {
    return tf::make_polygons(_geometry.faces(), _geometry.points());
  }
  auto frame() const -> frame_type { return _frame; }
  auto geometry() const -> const geometry_type & { return _geometry; }

  auto number_of_faces() const -> std::size_t {
    return _geometry.number_of_faces();
  }
  auto number_of_points() const -> std::size_t {
    return _geometry.number_of_points();
  }

  auto tree() const -> const tree_type & {
    if (!_tree)
      _tree = _cache->tree_handle(_geometry);
    return *_tree;
  }
  auto winding_moments() const -> const tf::winding_moments<Real> & {
    if (!_winding_moments)
      _winding_moments = _cache->winding_moments_handle(_geometry);
    return *_winding_moments;
  }
  auto half_edges() const -> const tf::half_edges<Index> & {
    if (!_half_edges)
      _half_edges = _cache->half_edges_handle(_geometry);
    return *_half_edges;
  }
  auto face_membership() const {
    if (!_face_membership.is_valid())
      _face_membership = _cache->face_membership_handle(_geometry);
    return tf::make_face_membership_like(
        std::as_const(_face_membership).make_range());
  }
  /// One peer per face side, in the shape that arity states it: three
  /// consecutive sides, or the blocks the structure carries. This is where the
  /// two states part and nowhere else.
  auto manifold_edge_link() const {
    struct dereference {
      auto operator()(Index index) const -> tf::manifold_edge_peer<Index> {
        return {index};
      }
    };
    if (!_manifold_edge_link.is_valid())
      _manifold_edge_link = _cache->manifold_edge_link_handle(_geometry);
    if constexpr (Ngon == 3) {
      auto mapped = tf::make_mapped_range(
          std::as_const(_manifold_edge_link).make_range(), dereference{});
      return tf::make_manifold_edge_link_like(
          tf::make_blocked_range<3>(mapped));
    } else {
      const auto offsets = _manifold_edge_link.offsets();
      const auto peers = _manifold_edge_link.data();
      auto mapped = tf::make_mapped_range(peers.make_range(), dereference{});
      return tf::make_manifold_edge_link_like(
          tf::make_offset_block_range(offsets.make_range(), mapped));
    }
  }
  auto face_link() const {
    if (!_face_link.is_valid())
      _face_link = _cache->face_link_handle(_geometry);
    return tf::make_face_link_like(std::as_const(_face_link).make_range());
  }
  /// @brief Refuse a face index these points do not have.
  ///
  /// The cache owns the fact and remembers it for this reading, so a second
  /// ask costs nothing and no body carries its own scan.
  ///
  /// WHO ASKS: an entry asks explicitly iff it reads the geometry without
  /// asking for a structure. Every structure ask is the door's too — a fill
  /// states it before it writes — so a body that takes a form has already
  /// passed it.
  auto require_indices() const -> void { _cache->require_indices(_geometry); }
  auto vertex_link() const {
    if (!_vertex_link.is_valid())
      _vertex_link = _cache->vertex_link_handle(_geometry);
    return tf::make_vertex_link_like(std::as_const(_vertex_link).make_range());
  }
  /// The unit normal of every face, and of every point — read off the stored
  /// points, so they name this mesh's own geometry and its frame is not
  /// applied.
  auto face_normals() const {
    if (!_face_normals.is_valid())
      _face_normals = _cache->face_normals_handle(_geometry);
    return tf::make_unit_vectors<3>(std::as_const(_face_normals).make_range());
  }
  auto point_normals() const {
    if (!_point_normals.is_valid())
      _point_normals = _cache->point_normals_handle(_geometry);
    return tf::make_unit_vectors<3>(std::as_const(_point_normals).make_range());
  }

  /// @brief The same mesh, read in its own frame.
  ///
  /// The frame is always tagged, so a caller who wants the operand where it
  /// was authored says identity rather than asking for a form without one.
  auto at_identity() const -> mesh {
    return mesh(_geometry, identity_transformation_view<Real, Dims>(), *_cache,
                _keepalive);
  }

  /// The polygons, the tree and the frame: what a spatial query takes. It
  /// names ONE structure, the tree, which is what a caller prebuilds for it.
  auto form() const { return polygons() | tf::tag(tree()) | tf::tag(_frame); }

  /// What the arrangement takes: the same, with the topology it walks. It
  /// names THREE structures — the tree, the face membership and the manifold
  /// edge link — which is what a caller prebuilds for every cut entry.
  auto topology_form() const {
    return polygons() | tf::tag(tree()) | tf::tag(face_membership()) |
           tf::tag(manifold_edge_link()) | tf::tag(_frame);
  }

  auto cache() const -> cache_type & { return *_cache; }

private:
  /// The same reading, placed elsewhere: the generations are the ones this mesh
  /// was assembled over, not whatever the cache is at now.
  mesh(geometry_type geometry, frame_type frame, cache_type &cache,
       std::shared_ptr<const void> keepalive)
      : _geometry(std::move(geometry)), _frame(frame), _cache(&cache),
        _keepalive(std::move(keepalive)) {}

  /// The flat arrays a faces and a points view stand on: a reading is flat, and
  /// this is the one place the two shapes of a faces view are read back.
  template <typename Faces, typename Points>
  static auto reading_of(const Faces &faces, const Points &points,
                         std::uint64_t faces_stamp, std::uint64_t points_stamp)
      -> geometry_type {
    const auto coordinates = detail::const_range_of(points.begin().base_iter(),
                                                    points.end().base_iter());
    if constexpr (Ngon == 3) {
      return {{},
              detail::const_range_of(faces.begin().base_iter(),
                                     faces.end().base_iter()),
              coordinates,
              faces_stamp,
              points_stamp};
    } else {
      const auto blocks = faces.begin();
      const auto span = faces.size() ? faces.size() + 1 : 0;
      const auto &indices = blocks.dereference_policy().range;
      return {detail::const_range_of(blocks.base_iter(), span),
              detail::const_range_of(indices.begin(), indices.end()),
              coordinates, faces_stamp, points_stamp};
    }
  }

  geometry_type _geometry;
  frame_type _frame;
  /// A structure is asked for, not taken: a body that never wants one pays
  /// nothing for it. Retaining what it hands back is not optional — a tag
  /// keeps views into the structure's arrays, and the cache REPLACES the whole
  /// of one when it rebuilds it, so a reference alone would dangle the moment
  /// another reading asks. One slot each, filled once, for this mesh's
  /// lifetime.
  mutable std::shared_ptr<const tree_type> _tree;
  mutable std::shared_ptr<const tf::winding_moments<Real>> _winding_moments;
  mutable std::shared_ptr<const tf::half_edges<Index>> _half_edges;
  mutable offset_blocked_buffer<Index, Index> _face_membership;
  mutable typename cache_type::face_blocks_type _manifold_edge_link;
  mutable offset_blocked_buffer<Index, Index> _face_link;
  mutable offset_blocked_buffer<Index, Index> _vertex_link;
  mutable nd_array<Real> _face_normals;
  mutable nd_array<Real> _point_normals;
  /// The cache is BORROWED, exactly as the geometry is: it is the caller's, and
  /// the caller outlives this mesh. Keeping it alive would protect nothing — a
  /// mesh whose arrays are gone reads freed memory whatever its caches say.
  cache_type *_cache;
  std::shared_ptr<const void> _keepalive;
};

} // namespace tf::cpp
