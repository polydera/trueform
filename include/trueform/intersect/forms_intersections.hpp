/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../core/algorithm/compute_offsets.hpp"
#include "../core/algorithm/generic_generate.hpp"
#include "../core/algorithm/mask_to_map.hpp"
#include "../core/intersects.hpp"
#include "../core/local_buffer.hpp"
#include "../core/views/zip.hpp"
#include "../spatial/form.hpp"
#include "../spatial/search.hpp"
#include "../topology/policy/face_membership.hpp"
#include "../topology/policy/manifold_edge_link.hpp"
#include "./compute_simplification_mask.hpp"
#include "./duplicate_intersection.hpp"
#include "./generate/polygon_polygon.hpp"
#include "./normal_intervals.hpp"
#include "./polygon/handle.hpp"

namespace tf {
template <typename Index, typename RealType, std::size_t Dims>
class forms_intersections {
public:
  template <typename Policy0, typename Policy1>
  auto build(const tf::form<Dims, Policy0> &form0,
             const tf::form<Dims, Policy1> &form1) {
    static_assert(tf::has_face_membership_policy<Policy0>);
    static_assert(tf::has_face_membership_policy<Policy1>);
    static_assert(tf::has_manifold_edge_link_policy<Policy0>);
    static_assert(tf::has_manifold_edge_link_policy<Policy1>);
    //
    clear();
    auto [intersection_ids, intersections, intersection_points] =
        compute_buffers(form0, form1);
    if (!intersections.size())
      return;

    auto keep_mask = tf::intersect::compute_simplification_mask(
        intersection_ids, form0, form1);
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
          tf::intersect::duplicate_intersection(
              form0.faces(), form1.faces(), intersection,
              form0.face_membership(), form0.manifold_edge_link(),
              form1.face_membership(), form1.manifold_edge_link(), buffer);
        });
    finalize(n_ids);
  }

  auto intersections() const {
    return tf::make_offset_block_range(_intersections_offsets, _intersections);
  }

  auto intersections0() const {
    return tf::make_offset_block_range(
        tf::make_range(_intersections_offsets.begin(), _partition_id),
        _intersections);
  }

  auto intersections1() const {
    return tf::make_offset_block_range(
        tf::make_range(_intersections_offsets.begin() + _partition_id,
                       _intersections_offsets.end()),
        _intersections);
  }

  auto intersection_points() const {
    return tf::make_range(_intersection_points);
  }

  auto clear() {
    _intersections.clear();
    _intersections_offsets.clear();
    _intersection_points.clear();
    _partition_id = 0;
  }

private:
  auto finalize(Index n_ids) {
    if (n_ids == 0)
      return;
    tbb::parallel_sort(_intersections.begin(), _intersections.end());
    _intersections_offsets.reserve(n_ids * 2 + 1);
    tf::compute_offsets(_intersections,
                        std::back_inserter(_intersections_offsets), Index(0),
                        [](const auto &x0, const auto &x1) {
                          return x0.polygon_key() == x1.polygon_key();
                        });
    auto r = tf::make_indirect_range(
        tf::make_range(_intersections_offsets.begin(),
                       _intersections_offsets.size() - 1),
        _intersections);
    _partition_id = std::upper_bound(r.begin(), r.end(), 0,
                                     [](const auto &value, const auto &r1) {
                                       return value < r1.mesh;
                                     }) -
                    r.begin();
  }

  template <typename Policy0, typename Policy1>
  auto compute_buffers(const tf::form<Dims, Policy0> &form0,
                       const tf::form<Dims, Policy1> &form1) {

    auto make_vertex_representation = [&](Index id, const auto &face,
                                          const auto &fe) {
      constexpr std::size_t N = tf::static_size_v<decltype(face)>;
      std::array<bool, N> out;
      for (std::size_t i = 0; i < N; ++i)
        out[i] = fe[face[i]].front() == id;
      return out;
    };

    auto make_edge_representation = [&](Index id, const auto &mel) {
      constexpr std::size_t N = tf::static_size_v<decltype(mel[id])>;
      std::array<bool, N> out;
      for (std::size_t i = 0; i < N; ++i)
        out[i] = mel[id][i].is_representative(id);
      return out;
    };

    auto make_handle = [&](auto poly, const auto &fe, const auto &mel) {
      return tf::intersect::polygon::make_handle(
          poly, poly.id(),
          make_vertex_representation(poly.id(), poly.indices(), fe),
          make_edge_representation(poly.id(), mel));
    };

    tf::local_buffer<tf::intersect::intersection_id<Index>> l_intersection_ids;
    tf::local_buffer<tf::intersect::intersection<Index>> l_intersections;
    tf::local_buffer<tf::point<RealType, Dims>> l_intersection_points;
    l_intersection_points.reserve_all(1000);
    l_intersections.reserve_all(1000);
    l_intersection_ids.reserve_all(1000);
    tf::search(form0, form1, tf::intersects_f,
               [&](const auto &obj0, const auto &obj1) {
                 auto poly0 = tf::tag_plane(obj0);
                 auto poly1 = tf::tag_plane(obj1);
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

  Index _partition_id = 0;
  tf::buffer<intersect::intersection<Index>> _intersections;
  tf::buffer<Index> _intersections_offsets;
  tf::buffer<tf::point<RealType, Dims>> _intersection_points;
};
} // namespace tf
