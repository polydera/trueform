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
#include "../../core/algorithm/ids_to_index_map.hpp"
#include "../../core/concatenated_blocked_ranges.hpp"
#include "../../core/index_map.hpp"
#include "../../core/stitch_index_maps.hpp"
#include "../../core/views/block_indirect_range.hpp"
#include "../../reindex/concatenated.hpp"
#include "../../reindex/return_index_map.hpp"
#include "../arrangement_class.hpp"
#include "../classify/tagged.hpp"
#include "./ids_common.hpp"
#include "./triangulate_cut_faces.hpp"

namespace tf::cut {
template <typename LabelType, typename Policy0, typename Policy1,
          typename Index, typename RealT, std::size_t Dims, bool MakeMaps>
auto make_boolean(
    const tf::polygons<Policy0> _polygons0,
    const tf::polygons<Policy1> &_polygons1,
    const tf::intersect::tagged_intersections<Index, RealT, Dims> &ibp,
    const tf::tagged_cut_faces<Index> &tcf,
    std::array<tf::arrangement_class, 2> classes,
    std::integral_constant<bool, MakeMaps>) {
  auto make_polygons = [](const auto &form) {
    return tf::wrap_map(form, [](auto &&x) {
      return tf::core::make_polygons(x.faces(),
                                     x.points().template as<RealT>());
    });
  };
  auto polygons0 = make_polygons(_polygons0);
  auto polygons1 = make_polygons(_polygons1);
  auto [pal0, pal1] = tf::cut::make_classifications<LabelType>(
      polygons0, polygons1, ibp, tcf, classes);
  tf::cut::polygon_arrangement_ids<Index> pai0;
  tf::cut::polygon_arrangement_ids<Index> pai1;
  tbb::parallel_invoke(
      [&pal0 = pal0, &pai0] {
        pai0 = tf::cut::make_polygon_arrangement_ids<Index>(pal0);
      },
      [&pal1 = pal1, &pai1] {
        pai1 = tf::cut::make_polygon_arrangement_ids<Index>(pal1);
      });
  tf::buffer<Index> original_map0;
  original_map0.allocate(polygons0.points().size());
  tf::parallel_fill(original_map0, -1);
  tf::buffer<Index> original_map1;
  original_map1.allocate(polygons1.points().size());
  tf::parallel_fill(original_map1, -1);
  tf::buffer<Index> created_map;
  created_map.allocate(ibp.intersection_points().size());
  tf::parallel_fill(created_map, -1);
  Index original_current0 = 0;
  Index original_current1 = 0;
  Index create_current = 0;
  tf::buffer<Index> original_ids0;
  original_ids0.reserve(polygons0.points().size());
  tf::buffer<Index> original_ids1;
  original_ids1.reserve(polygons1.points().size());
  tf::buffer<Index> created_ids;
  created_ids.reserve(created_map.size());
  auto process_map_on_loops = [&created_map, &create_current,
                               &created_ids](const auto &loops, auto &omap,
                                             auto &ocurr, auto &oids) {
    for (const auto &loop : loops)
      for (auto v : loop) {
        if (v.source == tf::loop::vertex_source::created) {
          if (created_map[v.id] == -1) {
            created_ids.push_back(v.id);
            created_map[v.id] = create_current++;
          }
        } else {
          if (omap[v.id] == -1) {
            omap[v.id] = ocurr++;
            oids.push_back(v.id);
          }
        }
      }
  };
  int index0 = 1; // always use include bucket
  int index1 = 1;
  process_map_on_loops(
      tf::make_indirect_range(pai0.cut_faces[index0], tcf.mapped_loops0()),
      original_map0, original_current0, original_ids0);
  process_map_on_loops(
      tf::make_indirect_range(pai1.cut_faces[index1], tcf.mapped_loops1()),
      original_map1, original_current1, original_ids1);
  auto fill_map = [](const auto &faces, auto &map, auto &curr, auto &ids) {
    for (const auto &face : faces) {
      for (auto v : face)
        if (map[v] == -1) {
          map[v] = curr++;
          ids.push_back(v);
        }
    }
  };
  tbb::parallel_invoke(
      [&] {
        fill_map(
            tf::make_indirect_range(pai0.polygons[index0], polygons0.faces()),
            original_map0, original_current0, original_ids0);
      },
      [&] {
        fill_map(
            tf::make_indirect_range(pai1.polygons[index1], polygons1.faces()),
            original_map1, original_current1, original_ids1);
      });

  auto map_vertex_f0 = [&](auto v) {
    if (v.source == tf::loop::vertex_source::created)
      return created_map[v.id] + original_current0 + original_current1;
    else
      return original_map0[v.id];
  };

  auto map_vertex_f1 = [&](auto v) {
    if (v.source == tf::loop::vertex_source::created)
      return created_map[v.id] + original_current0 + original_current1;
    else
      return original_map1[v.id] + original_current0;
  };
  auto make_projector = [&](auto d, const auto &polygons) {
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

  tf::blocked_buffer<Index, 3> triangles0;
  tf::blocked_buffer<Index, 3> triangles1;
  auto make_triangles = [&make_projector](const auto &polygons,
                                          const auto &loops, auto map_vertex_f,
                                          auto &buffer) {
    tf::cut::triangulate_cut_faces(
        loops, [&](auto d) { return make_projector(d, polygons); },
        map_vertex_f, buffer);
  };
  tbb::parallel_invoke(
      [&] {
        make_triangles(polygons0,
                       tf::make_indirect_range(
                           pai0.cut_faces[index0],
                           tf::zip(tcf.descriptors0(), tcf.mapped_loops0())),
                       map_vertex_f0, triangles0.data_buffer());
      },
      [&] {
        make_triangles(polygons1,
                       tf::make_indirect_range(
                           pai1.cut_faces[index1],
                           tf::zip(tcf.descriptors1(), tcf.mapped_loops1())),
                       map_vertex_f1, triangles1.data_buffer());
      });

  auto mapped_faces0 = tf::make_indirect_range(
      pai0.polygons[index0],
      tf::make_block_indirect_range(polygons0.faces(), original_map0));
  auto mapped_faces1 = tf::make_indirect_range(
      pai1.polygons[index1],
      tf::make_block_indirect_range(
          polygons1.faces(), tf::make_mapped_range(original_map1, [&](auto x) {
            return x + original_current0;
          })));

  auto [direction0, direction1] = tf::make_directions(classes[0], classes[1]);
  auto faces = tf::core::concatenated_blocked_ranges_directed<Index>(
      std::make_pair(tf::make_range(mapped_faces0), direction0),
      std::make_pair(tf::make_range(mapped_faces1), direction1),
      std::make_pair(tf::make_range(triangles0), direction0),
      std::make_pair(tf::make_range(triangles1), direction1));
  auto make_mapped_points = [](const auto &ids, const auto &polygons) {
    auto frame = tf::frame_of(polygons);
    return tf::make_points(tf::make_indirect_range(
        ids, tf::make_mapped_range(polygons.points(), [frame](auto pt) {
          return tf::transformed(pt, frame);
        })));
  };
  auto points = tf::concatenated(
      make_mapped_points(original_ids0, _polygons0),
      make_mapped_points(original_ids1, _polygons1),
      tf::make_points(
          tf::make_indirect_range(created_ids, ibp.intersection_points()))
          .template as<tf::coordinate_type<Policy0, Policy1>>());
  tf::buffer<std::int8_t> labels;
  labels.allocate(faces.size());
  tf::parallel_fill(tf::take(labels, mapped_faces0.size()), 0);
  tf::parallel_fill(
      tf::take(tf::drop(labels, mapped_faces0.size()), mapped_faces1.size()),
      1);
  tf::parallel_fill(
      tf::take(tf::drop(labels, mapped_faces0.size() + mapped_faces1.size()),
               triangles0.size()),
      0);
  tf::parallel_fill(
      tf::take(tf::drop(labels, mapped_faces0.size() + mapped_faces1.size() +
                                    triangles0.size()),
               triangles1.size()),
      1);
  if constexpr (MakeMaps) {
    tf::index_map_buffer<Index> original_im0;
    original_im0.f() = std::move(original_map0);
    original_im0.kept_ids() = std::move(original_ids0);
    tf::index_map_buffer<Index> original_im1;
    original_im1.f() = std::move(original_map1);
    original_im1.kept_ids() = std::move(original_ids1);
    tf::index_map_buffer<Index> created_im;
    created_im.f() = std::move(created_map);
    created_im.kept_ids() = std::move(created_ids);
    tf::index_map_buffer<Index> polygons_im0;
    tf::index_map_buffer<Index> polygons_im1;
    tf::ids_to_index_map(pai0.polygons[1], polygons_im0,
                         Index(_polygons0.size()), Index(0), Index(-1));
    tf::ids_to_index_map(pai1.polygons[1], polygons_im1,
                         Index(_polygons1.size()), Index(0), Index(-1));
    return std::make_tuple(
        tf::make_polygons_buffer(std::move(faces), std::move(points)),
        std::move(labels),
        tf::stitch_index_maps<Index>{
            std::move(original_im0), Index(0), std::move(original_im1),
            Index(original_current0), std::move(created_im),
            Index(original_current0 + original_current1),
            std::move(polygons_im0), Index(0), std::move(polygons_im1),
            Index(mapped_faces0.size()), direction0, direction1});
  } else {
    return std::make_pair(
        tf::make_polygons_buffer(std::move(faces), std::move(points)),
        std::move(labels));
  }
}

template <typename LabelType, typename Policy0, typename Policy1,
          typename Index, typename RealT, std::size_t Dims>
auto make_boolean(
    const tf::polygons<Policy0> _polygons0,
    const tf::polygons<Policy1> &_polygons1,
    const tf::intersect::tagged_intersections<Index, RealT, Dims> &ibp,
    const tf::tagged_cut_faces<Index> &tcf,
    std::array<tf::arrangement_class, 2> classes) {
  return make_boolean<LabelType>(_polygons0, _polygons1, ibp, tcf, classes,
                                 std::false_type{});
}

template <typename LabelType, typename Policy0, typename Policy1,
          typename Index, typename RealT, std::size_t Dims>
auto make_boolean(
    const tf::polygons<Policy0> _polygons0,
    const tf::polygons<Policy1> &_polygons1,
    const tf::intersect::tagged_intersections<Index, RealT, Dims> &ibp,
    const tf::tagged_cut_faces<Index> &tcf,
    std::array<tf::arrangement_class, 2> classes, tf::return_index_map_t) {
  return make_boolean<LabelType>(_polygons0, _polygons1, ibp, tcf, classes,
                                 std::true_type{});
}
} // namespace tf::cut
