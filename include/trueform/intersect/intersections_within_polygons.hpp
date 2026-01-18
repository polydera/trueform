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
#include "../clean/index_map/points.hpp"
#include "../core/algorithm/compute_offsets.hpp"
#include "../core/algorithm/generic_generate.hpp"
#include "../core/algorithm/mask_to_map.hpp"
#include "../core/algorithm/parallel_copy.hpp"
#include "../core/local_buffer.hpp"
#include "../core/views/zip.hpp"
#include "../core/polygons.hpp"
#include "../spatial/policy/tree.hpp"
#include "../spatial/search_self.hpp"
#include "../topology/edge_representation.hpp"
#include "../topology/policy/face_membership.hpp"
#include "../topology/policy/manifold_edge_link.hpp"
#include "../topology/vertex_representation.hpp"
#include "./detail/compute_self_simplification_mask.hpp"
#include "./detail/duplicate_self_intersection.hpp"
#include "./detail/normal_intervals.hpp"
#include "./generate/self_polygon_polygon.hpp"
#include "./polygon/handle.hpp"
#include "./types/intersection.hpp"

namespace tf {

/// @ingroup intersect_data
/// @brief Low-level self-intersection data within a polygon mesh.
///
/// Computes and stores all points where a mesh intersects itself
/// (excluding adjacent faces). Use @ref tf::make_self_intersection_curves
/// for high-level curve extraction.
///
/// Use pipe syntax with @ref tf::tag to add the required tree policy
/// (@ref tf::tree or @ref tf::mod_tree) and topology policies
/// (@ref tf::face_membership and @ref tf::manifold_edge_link).
///
/// @tparam Index The index type.
/// @tparam RealT The coordinate type.
/// @tparam Dims The number of dimensions.
template <typename Index, typename RealT, std::size_t Dims>
class intersections_within_polygons {
public:
  template <typename Policy> auto build(const tf::polygons<Policy> &_form) {
    static_assert(tf::has_tree_policy<Policy>,
                  "Use polygons | tf::tag(tree)");
    static_assert(tf::has_face_membership_policy<Policy>,
                  "Use polygons | tf::tag(face_membership)");
    static_assert(tf::has_manifold_edge_link_policy<Policy>,
                  "Use polygons | tf::tag(manifold_edge_link)");
    //
    clear();
    const auto &form = tf::wrap_map(_form, [](auto &&x) {
      return tf::core::make_polygons(x.faces(),
                                     x.points().template as<RealT>());
    });
    auto [intersection_ids, intersections, intersection_points] =
        compute_buffers(form);
    if (!intersections.size())
      return;

    auto keep_mask =
        tf::intersect::compute_self_simplification_mask(intersection_ids, form);
    tf::buffer<Index> map;
    map.allocate(keep_mask.size());
    auto n_ids = tf::mask_to_map(keep_mask, map);
    _intersection_points.allocate(n_ids);

    tf::generic_generate(
        tf::zip(intersections, intersection_points), _intersections,
        [&, none = Index(map.size())](auto pair, auto &buffer) {
          auto &&[intersection, point] = pair;
          if (map[intersection.id] == none)
            return;
          intersection.id = map[intersection.id];
          _intersection_points[intersection.id] = point;
          tf::intersect::duplicate_self_intersection(
              form.faces(), intersection, form.face_membership(),
              form.manifold_edge_link(), buffer);
        });
    collapse_points();
    finalize(n_ids);
  }

  auto intersections() const {
    return tf::make_offset_block_range(_intersections_offsets, _intersections);
  }

  auto intersection_points() const {
    return tf::make_range(_intersection_points);
  }

  auto clear() {
    _intersections.clear();
    _intersections_offsets.clear();
    _intersection_points.clear();
  }

  auto flat_intersections() const { return tf::make_range(_intersections); }

  auto get_flat_index(const intersect::intersection<Index> &i) const -> Index {
    return &i - _intersections.begin();
  }

private:
  auto collapse_points() {
    auto im = tf::make_clean_index_map<Index>(
        tf::make_points(_intersection_points), tf::epsilon<RealT>);
    if (im.kept_ids().size() == _intersection_points.size())
      return;
    tf::buffer<tf::point<RealT, Dims>> points;
    points.allocate(im.kept_ids().size());
    tf::parallel_copy(
        tf::make_indirect_range(im.kept_ids(), _intersection_points), points);
    _intersection_points = std::move(points);
    tf::parallel_for_each(_intersections, [&](auto &i) { i.id = im.f()[i.id]; });
  }

  template <typename Policy>
  auto compute_buffers(const tf::polygons<Policy> &form) {
    auto make_handle = [&](auto poly, const auto &fe, const auto &mel) {
      return tf::intersect::polygon::make_handle(
          poly, poly.id(),
          tf::make_vertex_representation(Index(poly.id()), poly.indices(), fe),
          tf::make_edge_representation(Index(poly.id()), mel));
    };
    auto check_early_out = [&form](const auto &poly0, const auto &poly1) {
      const auto &points = form.points();
      int count = 0;
      std::array<Index, 2> ids;
      int i = 0;

      for (auto id0 : poly0.indices()) {
        bool shared = false;
        for (auto id1 : poly1.indices())
          if (id0 == id1) {
            shared = true;
            break;
          }
        count += shared;
        if (!shared && i < 2)
          ids[i++] = id0;
      }

      const RealT eps = tf::epsilon<RealT>;

      if (count == 2) {
        if (std::abs(distance(poly1.plane(), points[ids[0]])) > eps)
          return true;
      }
      if (count == 1) {
        if (i != 2)
          return false;
        auto d0 = distance(poly1.plane(), points[ids[0]]);
        auto d1 = distance(poly1.plane(), points[ids[1]]);
        if (std::abs(d0) <= eps)
          d0 = RealT(0);
        if (std::abs(d1) <= eps)
          d1 = RealT(0);
        if ((d0 > 0 && d1 > 0) || (d0 < 0 && d1 < 0))
          return true;
      }
      return false;
    };
    tf::local_buffer<tf::intersect::intersection_id<Index>> l_intersection_ids;
    tf::local_buffer<tf::intersect::intersection<Index>> l_intersections;
    tf::local_buffer<tf::point<RealT, Dims>> l_intersection_points;
    l_intersection_points.reserve_all(1000);
    l_intersections.reserve_all(1000);
    tf::search_self(form, tf::intersects_f,
                    [&](const auto &object, const auto &object_other) {
                      auto poly1 = tf::tag_plane(object_other);
                      if (check_early_out(object, poly1))
                        return;
                      auto poly0 = tf::tag_plane(object);
                      if (!tf::intersect::normal_intervals(poly0, poly1))
                        return;
                      tf::intersect::generate::self_polygon_polygon(
                          make_handle(poly0, form.face_membership(),
                                      form.manifold_edge_link()),
                          make_handle(poly1, form.face_membership(),
                                      form.manifold_edge_link()),
                          *l_intersections, *l_intersection_ids,
                          *l_intersection_points);
                    });
    auto to_buffer = [](const auto &vs) {
      tf::buffer<std::decay_t<decltype(vs[0])>> out;
      auto size = vs.total_size();
      out.allocate(size);
      std::size_t offset = 0;
      auto it = out.begin();
      for (const auto &v : vs.buffers()) {
        for (auto e : v) {
          e.id += offset;
          *it++ = e;
        }
        offset += v.size();
      }

      return out;
    };
    return std::make_tuple(to_buffer(l_intersection_ids),
                           to_buffer(l_intersections),
                           l_intersection_points.to_buffer());
  }
  auto finalize(Index n_ids) {
    if (n_ids == 0)
      return;
    tbb::parallel_sort(_intersections.begin(), _intersections.end());
    _intersections_offsets.reserve(n_ids * 2 + 1);
    tf::compute_offsets(_intersections,
                        std::back_inserter(_intersections_offsets), Index(0),
                        [](const auto &x0, const auto &x1) {
                          return x0.object_key() == x1.object_key();
                        });
  }

  tf::buffer<tf::intersect::intersection<Index>> _intersections;
  tf::buffer<Index> _intersections_offsets;
  tf::buffer<tf::point<RealT, Dims>> _intersection_points;
};
} // namespace tf
