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
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/algorithm/parallel_for.hpp"
#include "../../core/buffer.hpp"
#include "../../core/polygons.hpp"
#include "../../core/small_vector.hpp"
#include "../../core/views/zip.hpp"
#include "../../intersect/types/simple_intersections.hpp"
#include "../../topology/topo_type.hpp"
#include "../loop/vertex_source.hpp"
#include "../scalar_cut_faces.hpp"
#include "./ids_common.hpp"

namespace tf::cut {
template <typename LabelType, typename Policy, typename Range0, typename Range1,
          typename F>
auto make_cut_faces_scalar_labels(const tf::polygons<Policy> &polygons,
                                  const Range0 &loops,
                                  const Range1 &intersections,
                                  const F &get_category,
                                  std::size_t n_categories) {
  /*
   * This method assumes:
   *  1. no edge has more than one intersection
   *     except if it lies on the vertices
   *  2. == cutoffvalue => intersection on vertex
   *
   *  This implies that each loop has at least one original
   *  vertex
   */
  tf::buffer<LabelType> out;
  out.allocate(loops.size());
  tf::parallel_for(
      tf::zip(loops, out),
      [&](auto begin, auto end) {
        tf::small_vector<LabelType, 20> counts;
        for (auto &&[loop, label] : tf::make_range(begin, end)) {
          counts.clear();
          counts.resize(n_categories);
          for (const auto &v : loop) {
            auto i_id = v.intersection_index;
            decltype(i_id) v_id;
            if (v.source == tf::loop::vertex_source::original)
              v_id = v.id;
            else if (intersections[i_id].target.label !=
                     tf::topo_type::vertex) {
              continue;
            } else {
              v_id = polygons.faces()[intersections[i_id].object]
                                     [intersections[i_id].target.id];
            }
            counts[get_category(v_id)]++;
          }
          label =
              std::max_element(counts.begin(), counts.end()) - counts.begin();
        }
      },
      tf::checked);
  return out;
}

template <typename LabelType, typename Policy, typename Range, typename F>
auto make_surface_scalar_labels(const tf::polygons<Policy> &polygons,
                                const Range &intersections,
                                const F &get_category) {
  tf::buffer<LabelType> out;
  out.allocate(polygons.size());
  tf::parallel_fill(out, -2);
  tf::parallel_for_each(
      intersections, [&](const auto &r) { out[r.front().object] = -1; },
      tf::checked);
  tf::parallel_for_each(
      tf::zip(out, polygons.faces()),
      [&](auto pair) {
        auto &&[label, face] = pair;
        if (label != -1)
          label = get_category(face[0]);
      },
      tf::checked);
  return out;
}

template <typename LabelType, typename Policy, typename Index, typename RealT,
          std::size_t Dims, typename Range, typename Iterator, std::size_t N>
auto make_scalar_labels(
    const tf::polygons<Policy> &polygons,
    const tf::intersect::simple_intersections<Index, RealT, Dims> &si,
    const tf::scalar_cut_faces<Index> &scf, const Range &scalars,
    tf::range<Iterator, N> cut_values) {
  tf::cut::polygon_arrangement_labels<LabelType> lbls;
  lbls.n_components = cut_values.size() + 1;

  tf::buffer<LabelType> categories;
  categories.allocate(scalars.size());
  tf::parallel_for_each(tf::zip(scalars, categories), [&](auto pair) {
    auto &&[scalar, category] = pair;
    category = std::lower_bound(cut_values.begin(), cut_values.end(), scalar) -
               cut_values.begin();
  });
  auto get_category = [&](auto x) { return categories[x]; };
  tbb::parallel_invoke(
      [&] {
        lbls.polygon_labels = make_surface_scalar_labels<LabelType>(
            polygons, si.intersections(), get_category);
      },
      [&] {
        lbls.cut_labels = make_cut_faces_scalar_labels<LabelType>(
            polygons, scf.mapped_loops(), si.flat_intersections(), get_category,
            cut_values.size() + 1);
      });
  return lbls;
}

template <typename LabelType, typename Policy, typename Index, typename RealT,
          std::size_t Dims, typename Range, typename Iterator, std::size_t N>
auto make_polygon_arrangement_ids(
    const tf::polygons<Policy> &polygons,
    const tf::intersect::simple_intersections<Index, RealT, Dims> &si,
    const tf::scalar_cut_faces<Index> &scf, const Range &scalars,
    tf::range<Iterator, N> cut_values) {
  auto lbls =
      make_scalar_labels<LabelType>(polygons, si, scf, scalars, cut_values);
  return tf::cut::make_polygon_arrangement_ids<Index>(lbls);
}
} // namespace tf::cut
