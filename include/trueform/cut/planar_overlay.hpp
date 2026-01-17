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
#include "../core/algorithm/parallel_copy_by_map_with_nones.hpp"
#include "../core/segments_buffer.hpp"
#include "../core/views/slide_range.hpp"
#include "../intersect/intersections_within_segments.hpp"

namespace tf {

/// @ingroup cut_planar
/// @brief Compute segment overlay with intersection processing.
///
/// Builds an overlay of @ref tf::segments, splitting at intersection points.
/// The result contains the subdivided segments with all intersections
/// resolved. Inherits from @ref tf::segments_buffer for edge/point access.
///
/// @tparam Index The index type.
/// @tparam RealT The coordinate type.
/// @tparam Dims The number of dimensions.
template <typename Index, typename RealT, std::size_t Dims>
class planar_overlay : public segments_buffer<Index, RealT, Dims> {
  using base_t = segments_buffer<Index, RealT, Dims>;

public:
  template <typename Policy>
  auto build(const tf::segments<Policy> &segments,
             tf::intersections_within_segments<Index, RealT, Dims> &si) {
    base_t::clear();
    auto [used_points, used_edges, raw_created_edges] =
        build_edges(segments, si);
    auto pt_map = handle_points(used_points, segments, si);

    // handle edges
    tf::buffer<Index> edge_map;
    edge_map.allocate(segments.size());
    auto n_edges = tf::mask_to_map(used_edges, edge_map);
    base_t::edges_buffer().data_buffer().allocate(n_edges * 2 +
                                                  raw_created_edges.size());
    auto raw0 = tf::make_range(base_t::edges_buffer().data_buffer().begin(),
                               raw_created_edges.size());
    auto raw1 = tf::make_range(base_t::edges_buffer().data_buffer().begin() +
                                   raw_created_edges.size(),
                               base_t::edges_buffer().data_buffer().end());
    tf::parallel_copy(raw_created_edges, raw0);
    tf::parallel_copy_by_map_with_nones(
        tf::make_mapped_range(
            segments.edges(),
            [offset = Index(si.intersection_points().size())](const auto &x) {
              return tf::point<Index, 2>{Index(x[0] + offset),
                                         Index(x[1] + offset)};
            }),
        tf::make_points<2>(raw1), edge_map);

    tf::parallel_for_each(
        base_t::edges_buffer().data_buffer(),
        [&pt_map, offset = Index(si.intersection_points().size())](auto &x) {
          if (x > offset)
            x = pt_map[x - offset] + offset;
        });
  }

private:
  template <typename Policy>
  auto build_edges(const tf::segments<Policy> &segments,
                   tf::intersections_within_segments<Index, RealT, Dims> &si) {
    tf::buffer<bool> used_points;
    used_points.allocate(segments.points().size());
    tf::parallel_fill(used_points, true);
    tf::buffer<bool> used_edges;
    used_edges.allocate(segments.size());
    tf::parallel_fill(used_edges, true);
    Index points_offset = si.intersection_points().size();
    tf::buffer<Index> _raw_edges;

    tf::generic_generate(
        si.intersections(), _raw_edges,
        tf::small_vector<std::pair<Index, RealT>, 4>{},
        [&](const auto &r, auto &buffer, auto &work_buffer) {
          auto edge_id = r.front().object;
          used_edges[edge_id] = false;
          auto [id0, id1] = segments.edges()[edge_id];
          if (id0 == id1)
            return;
          work_buffer.clear();
          work_buffer.reserve(r.size());
          std::optional<Index> vid0;
          std::optional<Index> vid1;
          auto dir = segments.points()[id1] - segments.points()[id0];
          for (const intersect::intersection<Index> &i : r) {
            if (i.target.label == tf::topo_type::vertex) {
              used_points[i.target.id] = false;
              if (id0 == i.target.id)
                vid0 = i.id;
              if (id1 == i.target.id)
                vid1 = i.id;
            } else
              work_buffer.push_back(
                  {i.id, tf::dot(dir, si.intersection_points()[i.id])});
          }
          std::sort(work_buffer.begin(), work_buffer.end(),
                    [](const auto &x0, const auto &x1) {
                      return x0.second < x1.second;
                    });
          auto push_edge = [&](Index i0, Index i1) {
            if (i0 != i1) {
              buffer.push_back(i0);
              buffer.push_back(i1);
            }
          };
          if (!work_buffer.size()) {
            push_edge(vid0 ? *vid0 : id0 + points_offset,
                      vid1 ? *vid1 : id1 + points_offset);
            return;
          }
          push_edge(vid0 ? *vid0 : id0 + points_offset,
                    work_buffer.front().first);
          if (work_buffer.size() > 1)
            for (auto [a, b] : tf::make_slide_range<2>(work_buffer))
              push_edge(a.first, b.first);
          push_edge(work_buffer.back().first,
                    vid1 ? *vid1 : id1 + points_offset);
        });
    // we might have edges (a, b), (c, d)
    // intersecting: a--c--d--b
    // leaving the edge c--d duplicated
    auto edges_as_points = tf::make_points<2>(_raw_edges);
    auto make_edge = [](auto v) {
      if (v[0] < v[1])
        return std::make_pair(v[0], v[1]);
      else
        return std::make_pair(v[1], v[0]);
    };
    tbb::parallel_sort(edges_as_points, [&](auto x, auto y) {
      return make_edge(x) < make_edge(y);
    });
    auto n_unique = (std::unique(edges_as_points.begin(), edges_as_points.end(),
                                 [&](auto x, auto y) {
                                   return make_edge(x) == make_edge(y);
                                 }) -
                     edges_as_points.begin()) *
                    2;
    _raw_edges.erase(_raw_edges.begin() + n_unique, _raw_edges.end());
    return std::make_tuple(std::move(used_points), std::move(used_edges),
                           std::move(_raw_edges));
  }

  template <typename Policy>
  auto
  handle_points(const tf::buffer<bool> &used_points,
                const tf::segments<Policy> &segments,
                tf::intersections_within_segments<Index, RealT, Dims> &si) {
    tf::buffer<Index> pt_map;
    pt_map.allocate(segments.points().size());
    auto n_pts = tf::mask_to_map(used_points, pt_map);
    base_t::points_buffer().data_buffer().allocate(
        (n_pts + si.intersection_points().size()) * Dims);
    auto raw0 = tf::make_range(base_t::points_buffer().data_buffer().begin(),
                               Dims * si.intersection_points().size());
    auto raw1 = tf::make_range(base_t::points_buffer().data_buffer().begin() +
                                   Dims * si.intersection_points().size(),
                               base_t::points_buffer().data_buffer().end());
    tf::parallel_copy(si.intersection_points(), tf::make_points<Dims>(raw0));
    tf::parallel_copy_by_map_with_nones(segments.points(),
                                        tf::make_points<Dims>(raw1), pt_map);
    return pt_map;
  }
};
} // namespace tf
