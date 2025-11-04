/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../../core/concatenated_blocked_ranges.hpp"
#include "../../core/views/block_indirect_range.hpp"
#include "../../reindex/concatenated.hpp"
#include "../classify/tagged.hpp"
#include "./ids_common.hpp"
#include "./triangulate_cut_faces.hpp"

namespace tf::cut {

template <typename Policy, typename Index, typename RealT, std::size_t Dims,
          typename Range0, typename Range1>
auto make_boolean_pair(
    const tf::polygons<Policy> _polygons,
    tf::cut::polygon_arrangement_ids<Index> &pai,
    const tf::intersect::tagged_intersections<Index, RealT, Dims> &ibp,
    const Range0 &descriptors, const Range1 &mapped_loops,
    tf::strict_containment cclass) {
  auto make_polygons = [](const auto &form) {
    return tf::wrap_map(form, [](auto &&x) {
      return tf::core::make_polygons(x.faces(),
                                     x.points().template as<RealT>());
    });
  };
  auto polygons = make_polygons(_polygons);

  tf::buffer<Index> original_map;
  original_map.allocate(polygons.points().size());
  tf::parallel_fill(original_map, -1);
  tf::buffer<Index> created_map;
  created_map.allocate(ibp.intersection_points().size());
  tf::parallel_fill(created_map, -1);
  Index original_current = 0;
  Index create_current = 0;
  tf::buffer<Index> original_ids;
  original_ids.reserve(polygons.points().size());
  tf::buffer<Index> created_ids;
  created_ids.reserve(created_map.size());
  for (const auto &loop : mapped_loops)
    for (auto v : loop) {
      if (v.source == tf::loop::vertex_source::created) {
        if (created_map[v.id] == -1) {
          created_ids.push_back(v.id);
          created_map[v.id] = create_current++;
        }
      } else {
        if (original_map[v.id] == -1) {
          original_map[v.id] = original_current++;
          original_ids.push_back(v.id);
        }
      }
    }

  int index = cclass == tf::strict_containment::outside;
  for (const auto &face : polygons.faces()) {
    for (auto v : face)
      if (original_map[v] == -1) {
        original_map[v] = original_current++;
        original_ids.push_back(v);
      }
  }

  auto map_vertex_f = [&](auto v) {
    if (v.source == tf::loop::vertex_source::created)
      return created_map[v.id] + original_current;
    else
      return original_map[v.id];
  };

  auto make_projector = [&](auto d) {
    auto frame = tf::frame_of(polygons);
    auto proj = tf::make_simple_projector(
        tf::transformed_normal(tf::make_normal(polygons[d.object]), frame));
    return [proj, &polygons, &ibp, frame](auto v) {
      if (v.source == tf::loop::vertex_source::original)
        return proj(tf::transformed(polygons.points()[v.id], frame));
      else
        return proj(ibp.intersection_points()[v.id]);
    };
  };

  tf::blocked_buffer<Index, 3> triangles;
  tf::cut::triangulate_cut_faces(
      tf::make_indirect_range(pai.cut_faces[index],
                              tf::zip(descriptors, mapped_loops)),
      make_projector, map_vertex_f, triangles.data_buffer());

  auto mapped_faces = tf::make_indirect_range(
      pai.polygons[index],
      tf::make_block_indirect_range(polygons.faces(), original_map));

  auto direction = cclass == tf::strict_containment::inside
                       ? tf::direction::reverse
                       : tf::direction::forward;
  auto faces = tf::core::concatenated_blocked_ranges_directed<Index>(
      std::make_pair(tf::make_range(mapped_faces), direction),
      std::make_pair(tf::make_range(triangles), direction));
  auto make_mapped_points = [](const auto &ids, const auto &polygons) {
    auto frame = tf::frame_of(polygons);
    return tf::make_points(tf::make_indirect_range(
        ids, tf::make_mapped_range(polygons.points(), [frame](auto pt) {
          return tf::transformed(pt, frame);
        })));
  };
  auto points = tf::concatenated(
      make_mapped_points(original_ids, _polygons),
      tf::make_points(
          tf::make_indirect_range(created_ids, ibp.intersection_points()))
          .template as<tf::coordinate_type<Policy>>());
  return tf::make_polygons_buffer(std::move(faces), std::move(points));
}

template <typename LabelType, typename Policy0, typename Policy1,
          typename Index, typename RealT, std::size_t Dims>
auto make_boolean_pair(
    const tf::polygons<Policy0> _polygons0,
    const tf::polygons<Policy1> &_polygons1,
    const tf::intersect::tagged_intersections<Index, RealT, Dims> &ibp,
    const tf::tagged_cut_faces<Index> &tcf,
    std::array<tf::strict_containment, 2> classes) {
  auto make_polygons = [](const auto &form) {
    return tf::wrap_map(form, [](auto &&x) {
      return tf::core::make_polygons(x.faces(),
                                     x.points().template as<RealT>());
    });
  };
  auto polygons0 = make_polygons(_polygons0);
  auto polygons1 = make_polygons(_polygons1);
  auto [pal0, pal1] =
      tf::cut::make_classifications<LabelType>(polygons0, polygons1, ibp, tcf);
  tf::cut::polygon_arrangement_ids<Index> pai0;
  tf::cut::polygon_arrangement_ids<Index> pai1;
  tbb::parallel_invoke(
      [&pal0 = pal0, &pai0] {
        pai0 = tf::cut::make_polygon_arrangement_ids<Index>(pal0);
      },
      [&pal1 = pal1, &pai1] {
        pai1 = tf::cut::make_polygon_arrangement_ids<Index>(pal1);
      });
  using res_t =
      decltype(make_boolean_pair(_polygons0, pai0, ibp, tcf.descriptors0(),
                                 tcf.mapped_loops0(), classes[0]));
  res_t left;
  res_t right;
  tbb::parallel_invoke(
      [&] {
        left = make_boolean_pair(_polygons0, pai0, ibp, tcf.descriptors0(),
                                 tcf.mapped_loops0(), classes[0]);
      },
      [&] {
        right = make_boolean_pair(_polygons1, pai1, ibp, tcf.descriptors1(),
                                  tcf.mapped_loops1(), classes[1]);
      });
  return std::make_pair(std::move(left), std::move(right));
}
} // namespace tf::cut
