/*
 * Copyright (c) 2025 XLAB
 * All rights reserved.
 *
 * Author: Žiga Sajovic
 */
#pragma once

#include "../core/algorithm/parallel_copy.hpp"
#include "../core/algorithm/parallel_for_each.hpp"
#include "../core/buffer.hpp"
#include "../core/index_map.hpp"
#include "../core/views/drop.hpp"
#include "../core/views/indirect_range.hpp"
#include "../core/views/sequence_range.hpp"
#include "../topology/half_edges.hpp"
#include "./feature_mask.hpp"
#include "./make_feature_mask.hpp"

#include <cstddef>
#include <utility>

namespace tf::remesh {

template <typename Index, typename Label = Index>
struct feature_handler {
  feature_mask mask;
  tf::buffer<Label> face_labels;

  auto empty() const -> bool { return mask.empty(); }
  auto has_regions() const -> bool { return face_labels.size() != 0; }
  auto is_feature(std::size_t e) const -> bool { return mask.is_feature(e); }
  auto vertex_type(std::size_t v) const -> vertex_feature_type {
    return mask.vertex_type(v);
  }
  auto is_crease(std::size_t v) const -> bool { return mask.is_crease(v); }
  auto is_corner(std::size_t v) const -> bool { return mask.is_corner(v); }
  auto is_collapse_forbidden(std::size_t e, std::size_t v) const -> bool {
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
    }
  }

  template <typename FaceRange0, typename FaceRange1, typename EdgeRange0,
            typename EdgeRange1, typename VertRange0, typename VertRange1>
  auto compact(const tf::index_map<FaceRange0, FaceRange1> &face_im,
               const tf::index_map<EdgeRange0, EdgeRange1> &edge_im,
               const tf::index_map<VertRange0, VertRange1> &vert_im) -> void {
    mask.compact(edge_im, vert_im);
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

} // namespace tf::remesh
