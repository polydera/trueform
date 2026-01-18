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
#include "../core/algorithm/generic_generate.hpp"
#include "../core/algorithm/mask_to_map.hpp"
#include "../core/algorithm/parallel_copy.hpp"
#include "../core/epsilon.hpp"
#include "../core/local_buffer.hpp"
#include "../core/views/zip.hpp"
#include "../core/polygons.hpp"
#include "../spatial/policy/tree.hpp"
#include "../spatial/search.hpp"
#include "../topology/edge_representation.hpp"
#include "../topology/policy/face_membership.hpp"
#include "../topology/policy/manifold_edge_link.hpp"
#include "../topology/vertex_representation.hpp"
#include "./detail/compute_simplification_mask.hpp"
#include "./detail/duplicate_intersection.hpp"
#include "./detail/normal_intervals.hpp"
#include "./generate/polygon_polygon.hpp"
#include "./polygon/handle.hpp"
#include "./types/tagged_intersections.hpp"

namespace tf {

/// @ingroup intersect_data
/// @brief Low-level intersection data between two polygon meshes.
///
/// Computes and stores all intersection points between two meshes,
/// along with topological information about where intersections occur.
/// Use @ref tf::make_intersection_curves for high-level curve extraction.
///
/// Use pipe syntax with @ref tf::tag to add the required tree policy
/// (@ref tf::tree or @ref tf::mod_tree) and topology policies
/// (@ref tf::face_membership and @ref tf::manifold_edge_link).
///
/// @tparam Index The index type.
/// @tparam RealType The coordinate type.
/// @tparam Dims The number of dimensions.
template <typename Index, typename RealType, std::size_t Dims>
class intersections_between_polygons
    : public intersect::tagged_intersections<Index, RealType, Dims> {
  using base_t = intersect::tagged_intersections<Index, RealType, Dims>;

public:
  template <typename Policy0, typename Policy1>
  auto build(const tf::polygons<Policy0> &_form0,
             const tf::polygons<Policy1> &_form1) {
    static_assert(tf::has_tree_policy<Policy0>,
                  "Use polygons | tf::tag(tree)");
    static_assert(tf::has_tree_policy<Policy1>,
                  "Use polygons | tf::tag(tree)");
    static_assert(tf::has_face_membership_policy<Policy0>,
                  "Use polygons | tf::tag(face_membership)");
    static_assert(tf::has_face_membership_policy<Policy1>,
                  "Use polygons | tf::tag(face_membership)");
    static_assert(tf::has_manifold_edge_link_policy<Policy0>,
                  "Use polygons | tf::tag(manifold_edge_link)");
    static_assert(tf::has_manifold_edge_link_policy<Policy1>,
                  "Use polygons | tf::tag(manifold_edge_link)");
    auto make_form = [](const auto &form) {
      return tf::wrap_map(form, [](auto &&x) {
        return tf::core::make_polygons(x.faces(),
                                       x.points().template as<RealType>());
      });
    };
    const auto &form0 = make_form(_form0);
    const auto &form1 = make_form(_form1);
    //
    base_t::clear();
    auto [intersection_ids, intersections, intersection_points] =
        compute_buffers(form0, form1);
    if (!intersections.size())
      return;

    auto keep_mask = tf::intersect::compute_simplification_mask(
        intersection_ids, form0, form1);
    tf::buffer<Index> map;
    map.allocate(keep_mask.size());
    auto n_ids = tf::mask_to_map(keep_mask, map);
    base_t::_intersection_points.allocate(n_ids);

    tf::generic_generate(
        tf::zip(intersections, intersection_points), base_t::_intersections,
        [&, none = Index(map.size())](auto pair, auto &buffer) {
          auto &&[intersection, point] = pair;
          if (map[intersection.id] == none)
            return;
          intersection.id = map[intersection.id];
          base_t::_intersection_points[intersection.id] = point;
          tf::intersect::duplicate_intersection(
              form0.faces(), form1.faces(), intersection,
              form0.face_membership(), form0.manifold_edge_link(),
              form1.face_membership(), form1.manifold_edge_link(), buffer);
        });
    collapse_points();
    base_t::finalize(n_ids);
  }

private:
  auto collapse_points() {
    auto im = tf::make_clean_index_map<Index>(
        tf::make_points(base_t::_intersection_points), tf::epsilon<RealType>);
    if (im.kept_ids().size() == base_t::_intersection_points.size())
      return;
    tf::buffer<tf::point<RealType, Dims>> points;
    points.allocate(im.kept_ids().size());
    tf::parallel_copy(
        tf::make_indirect_range(im.kept_ids(), base_t::_intersection_points),
        points);
    base_t::_intersection_points = std::move(points);
    tf::parallel_for_each(base_t::_intersections,
                       [&](auto &i) { i.id = im.f()[i.id]; });
  }

  template <typename Policy0, typename Policy1>
  auto compute_buffers(const tf::polygons<Policy0> &form0,
                       const tf::polygons<Policy1> &form1) {

    auto make_handle = [&](auto poly, const auto &fe, const auto &mel) {
      return tf::intersect::polygon::make_handle(
          poly, poly.id(),
          tf::make_vertex_representation(Index(poly.id()), poly.indices(), fe),
          tf::make_edge_representation(Index(poly.id()), mel));
    };

    tf::local_buffer<tf::intersect::intersection_id<Index>> l_intersection_ids;
    tf::local_buffer<tf::intersect::tagged_intersection<Index>> l_intersections;
    tf::local_buffer<tf::point<RealType, Dims>> l_intersection_points;
    l_intersection_points.reserve_all(1000);
    l_intersections.reserve_all(1000);
    l_intersection_ids.reserve_all(1000);
    tf::search(form0, form1, tf::intersects_f,
               [&](const auto &object, const auto &object_other) {
                 auto poly0 = tf::tag_plane(object);
                 auto poly1 = tf::tag_plane(object_other);
                 if (!tf::intersect::normal_intervals(poly0, poly1))
                   return;
                 tf::intersect::generate::polygon_polygon(
                     make_handle(poly0, form0.face_membership(),
                                 form0.manifold_edge_link()),
                     make_handle(poly1, form1.face_membership(),
                                 form1.manifold_edge_link()),
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
};
} // namespace tf
