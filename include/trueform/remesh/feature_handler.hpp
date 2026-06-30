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

#include "../core/algorithm/parallel_copy.hpp"
#include "../core/algorithm/parallel_fill.hpp"
#include "../core/algorithm/parallel_for_each.hpp"
#include "../core/buffer.hpp"
#include "../core/index_map.hpp"
#include "../core/none.hpp"
#include "../core/views/drop.hpp"
#include "../core/views/indirect_range.hpp"
#include "../core/views/sequence_range.hpp"
#include "../topology/half_edges.hpp"
#include "./feature_mask.hpp"
#include "./make_feature_mask.hpp"
#include "./regions/region_label.hpp"

#include <cstddef>
#include <type_traits>
#include <utility>

namespace tf::remesh {

template <typename Index, typename Label = Index>
struct feature_handler {
  feature_mask mask;
  tf::buffer<Label> face_labels;
  /// Per-vertex: pinned by tf::protect_vertices. A pinned vertex is never
  /// removed by collapse and is frozen (corner) during relaxation. Persistent
  /// source of truth: re-applied after every recompute, compacted every pass.
  tf::buffer<bool> protected_vertices;

  auto empty() const -> bool { return mask.empty(); }
  auto has_regions() const -> bool { return face_labels.size() != 0; }
  auto has_protection() const -> bool { return protected_vertices.size() != 0; }
  auto is_feature(std::size_t e) const -> bool { return mask.is_feature(e); }
  auto vertex_type(std::size_t v) const -> vertex_feature_type {
    return mask.vertex_type(v);
  }
  auto is_crease(std::size_t v) const -> bool { return mask.is_crease(v); }
  auto is_corner(std::size_t v) const -> bool { return mask.is_corner(v); }
  auto is_collapse_forbidden(std::size_t e, std::size_t v) const -> bool {
    if (protected_vertices.size() != 0 && protected_vertices[v])
      return true;
    return mask.is_collapse_forbidden(e, v);
  }
  auto face_label(std::size_t f) const -> Label { return face_labels[f]; }

  /// Build the mask from a dihedral feature angle.
  template <typename PointsPolicy>
  auto init(const tf::half_edges<Index> &he,
            const tf::points<PointsPolicy> &points,
            tf::rad<tf::coordinate_type<PointsPolicy>> angle) -> void {
    recompute(he, points, angle);
  }

  /// Build the mask purely from per-face region labels: an edge is a
  /// feature iff its two adjacent faces have different labels.
  template <typename PointsPolicy, typename FaceLabels>
  auto init_regions(const tf::half_edges<Index> &he,
                    const tf::points<PointsPolicy> &points,
                    FaceLabels user_face_labels) -> void {
    using Real = tf::coordinate_type<PointsPolicy>;
    face_labels.allocate(user_face_labels.size());
    tf::parallel_copy(tf::make_range(user_face_labels),
                      tf::make_range(face_labels));
    recompute(he, points, tf::rad<Real>(Real(-1)));
  }

  /// Combine dihedral features with region-label boundaries.
  template <typename PointsPolicy, typename FaceLabels>
  auto init(const tf::half_edges<Index> &he,
            const tf::points<PointsPolicy> &points,
            tf::rad<tf::coordinate_type<PointsPolicy>> angle,
            FaceLabels user_face_labels) -> void {
    face_labels.allocate(user_face_labels.size());
    tf::parallel_copy(tf::make_range(user_face_labels),
                      tf::make_range(face_labels));
    recompute(he, points, angle);
  }

  /// Record the per-vertex protection mask (true = pinned). Persistent: stored
  /// once, then re-applied by every recompute and carried through compact. The
  /// caller is responsible for a matching dihedral/region init or recompute so
  /// the mask is allocated -- protect-only callers pass angle < 0.
  template <typename Mask>
  auto set_protection(Mask vertex_mask) -> void {
    protected_vertices.clear();
    protected_vertices.allocate(vertex_mask.size());
    tf::parallel_copy(tf::make_range(vertex_mask),
                      tf::make_range(protected_vertices));
  }

  /// Force every pinned vertex to a corner so collapse and relaxation both
  /// freeze it. Called at the tail of recompute, after vertex types are
  /// (re)derived from feature edges, since that pass would otherwise demote it.
  auto apply_protection() -> void {
    if (protected_vertices.size() == 0)
      return;
    tf::parallel_for_each(
        tf::make_sequence_range(Index(protected_vertices.size())),
        [&](Index v) {
          if (protected_vertices[v])
            mask.vertices[v] = vertex_feature_type::corner;
        });
  }

  /// Re-derive the mask from the current mesh state. Preserves face_labels;
  /// rebuilds mask.edges and mask.vertices.
  template <typename PointsPolicy>
  auto recompute(const tf::half_edges<Index> &he,
                 const tf::points<PointsPolicy> &points,
                 tf::rad<tf::coordinate_type<PointsPolicy>> angle) -> void {
    using Real = tf::coordinate_type<PointsPolicy>;
    auto n_edges = static_cast<Index>(he.half_edges_buffer().size() / 2);
    if (angle.value >= 0)
      mask = tf::make_feature_mask(he, points, angle);
    else {
      mask.edges.clear();
      mask.edges.allocate(n_edges);
      tf::parallel_fill(tf::make_range(mask.edges), false);
    }
    if (face_labels.size() != 0) {
      tf::parallel_for_each(tf::make_sequence_range(n_edges), [&](Index eid) {
        if (mask.edges[eid])
          return;
        auto eh = tf::edge_handle<Index>{eid};
        auto heh0 = he.half_edge_handle(tf::unsafe, eh, false);
        auto heh1 = he.opposite(tf::unsafe, heh0);
        if (!heh0.is_valid() || !heh1.is_valid() ||
            !he.is_simple(tf::unsafe, heh0) ||
            !he.is_simple(tf::unsafe, heh1))
          return;
        auto f0 = he.half_edge(heh0).face;
        auto f1 = he.half_edge(heh1).face;
        if (face_labels[f0] != face_labels[f1])
          mask.edges[eid] = true;
      });
    }
    tf::remesh::recompute_vertex_types(he, mask);
    tf::remesh::harden_bent_creases(he, points, mask,
                                    tf::deg<Real>(Real(30)));
    apply_protection();
  }

  template <typename PointsPolicy, typename FaceRange, typename EdgeRange>
  auto update(const tf::half_edges<Index> &he,
              const tf::points<PointsPolicy> &points,
              FaceRange parent_face_for_new,
              EdgeRange parent_edge_for_new) -> void {
    using Real = tf::coordinate_type<PointsPolicy>;
    if (face_labels.size() != 0 && parent_face_for_new.size() != 0) {
      auto n0 = face_labels.size();
      face_labels.reallocate(n0 + parent_face_for_new.size());
      tf::parallel_copy(
          tf::make_indirect_range(parent_face_for_new,
                                  tf::make_range(face_labels)),
          tf::drop(tf::make_range(face_labels), n0));
    }
    if (mask.edges.size() != 0) {
      auto n0 = mask.edges.size();
      Index n_new = Index(parent_edge_for_new.size());
      if (n_new != 0) {
        mask.edges.reallocate(n0 + n_new);
        tf::parallel_for_each(
            tf::make_sequence_range(n_new), [&, n0](Index i) {
              Index parent = parent_edge_for_new[i];
              mask.edges[n0 + i] =
                  (parent == Index(-1)) ? false : mask.edges[parent];
            });
      }
      tf::remesh::recompute_vertex_types(he, mask);
      // recompute_vertex_types classifies purely by feature-edge count, which
      // demotes a bent in-plane corner (only 2 feature edges) back to a crease.
      // Re-harden so such corners survive the collapse pass that follows a
      // split -- a crease can be collapsed along its feature edge, a corner
      // cannot.
      tf::remesh::harden_bent_creases(he, points, mask,
                                      tf::deg<Real>(Real(30)));
      // Grow the protection mask to cover split-added vertices, which are never
      // pinned. recompute_vertex_types has just sized mask.vertices to the
      // current vertex count, so match it. Without this, is_collapse_forbidden
      // and the next compact would index protected_vertices out of bounds.
      if (protected_vertices.size() != 0 &&
          protected_vertices.size() < mask.vertices.size()) {
        auto old_n = protected_vertices.size();
        protected_vertices.reallocate(mask.vertices.size());
        tf::parallel_fill(
            tf::drop(tf::make_range(protected_vertices), old_n), false);
      }
      apply_protection();
    }
  }

  template <typename FaceRange0, typename FaceRange1, typename EdgeRange0,
            typename EdgeRange1, typename VertRange0, typename VertRange1>
  auto compact(const tf::index_map<FaceRange0, FaceRange1> &face_im,
               const tf::index_map<EdgeRange0, EdgeRange1> &edge_im,
               const tf::index_map<VertRange0, VertRange1> &vert_im) -> void {
    mask.compact(edge_im, vert_im);
    if (protected_vertices.size() != 0) {
      tf::buffer<bool> new_protected;
      new_protected.allocate(vert_im.kept_ids().size());
      tf::parallel_copy(
          tf::make_indirect_range(vert_im.kept_ids(),
                                  tf::make_range(protected_vertices)),
          tf::make_range(new_protected));
      protected_vertices = std::move(new_protected);
    }
    if (face_labels.size() == 0)
      return;
    tf::buffer<Label> new_labels;
    new_labels.allocate(face_im.kept_ids().size());
    tf::parallel_copy(
        tf::make_indirect_range(face_im.kept_ids(),
                                tf::make_range(face_labels)),
        tf::make_range(new_labels));
    face_labels = std::move(new_labels);
  }

  struct view {
    feature_handler *_h;

    template <typename PointsPolicy, typename FaceRange, typename EdgeRange>
    auto update(const tf::half_edges<Index> &he,
                const tf::points<PointsPolicy> &points, FaceRange pf,
                EdgeRange pe) const -> void {
      _h->update(he, points, pf, pe);
    }

    auto empty() const -> bool { return _h->empty(); }
    auto has_regions() const -> bool { return _h->has_regions(); }
    auto is_feature(std::size_t e) const -> bool { return _h->is_feature(e); }
    auto vertex_type(std::size_t v) const -> vertex_feature_type {
      return _h->vertex_type(v);
    }
    auto is_crease(std::size_t v) const -> bool { return _h->is_crease(v); }
    auto is_corner(std::size_t v) const -> bool { return _h->is_corner(v); }
    auto is_collapse_forbidden(std::size_t e, std::size_t v) const -> bool {
      return _h->is_collapse_forbidden(e, v);
    }
    auto face_label(std::size_t f) const -> Label {
      return _h->face_label(f);
    }
  };

  auto as_view() -> view { return {this}; }
};

/// @brief Build a feature_handler from the optional dihedral feature angle,
/// region labels, and vertex-protection mask -- the shared front end of every
/// remesh core. Regions is tf::none_t or tf::preserve_regions_t; Protection is
/// tf::none_t or tf::protect_vertices_t. The protection mask is recorded first
/// so the build's recompute pins it; a protect-only call (no regions, angle < 0)
/// still builds a trivial mask so the pinned vertices take effect.
template <typename Index, typename PointsPolicy, typename Regions,
          typename Protection>
auto build_feature_handler(
    const tf::half_edges<Index> &he, const tf::points<PointsPolicy> &points,
    tf::rad<tf::coordinate_type<PointsPolicy>> angle, Regions regions,
    Protection protection)
    -> feature_handler<Index, tf::remesh::region_label_t<Regions, Index>> {
  using Label = tf::remesh::region_label_t<Regions, Index>;
  feature_handler<Index, Label> features;
  if constexpr (!std::is_same_v<Protection, tf::none_t>)
    features.set_protection(protection.vertex_mask);
  if constexpr (std::is_same_v<Regions, tf::none_t>) {
    if (angle.value >= 0)
      features.init(he, points, angle);
    else if constexpr (!std::is_same_v<Protection, tf::none_t>)
      features.recompute(he, points, angle);
  } else {
    if (angle.value >= 0)
      features.init(he, points, angle, regions.face_regions);
    else
      features.init_regions(he, points, regions.face_regions);
  }
  return features;
}

/// @brief A remesh core's result: the maintained feature_handler plus, when a
/// vertex map was requested, the original->final vertex index map (empty
/// otherwise). Used by the cores that compact internally (error_remesh,
/// isotropic_remesh). The map is VERTICES ONLY -- the flip and split passes make
/// a face map meaningless. A collapsed-away original maps to the none sentinel;
/// a split-created output vertex (isotropic) has no original preimage.
template <typename Index, typename Label>
struct remesh_result {
  feature_handler<Index, Label> features;
  tf::index_map_buffer<Index> vertex_map;
};

} // namespace tf::remesh
