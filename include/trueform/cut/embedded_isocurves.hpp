/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../core/curves_buffer.hpp"
#include "../intersect/scalar_field_intersections.hpp"
#include "../topology/connect_edges_to_paths.hpp"
#include "./impl/embedded_isocurves.hpp"
#include "./return_curves.hpp"
#include "./scalar_cut_faces.hpp"

namespace tf {
template <typename Index, typename Policy, typename Range0, typename Iterator0,
          std::size_t N0>
auto embedded_isocurves(const tf::polygons<Policy> &polygons,
                        const Range0 &scalars,
                        const tf::range<Iterator0, N0> &cut_values) {
  tf::buffer<std::decay_t<decltype(cut_values[0])>> cut_vals;
  cut_vals.reserve(cut_values.size());
  std::copy(cut_values.begin(), cut_values.end(), std::back_inserter(cut_vals));
  std::sort(cut_vals.begin(), cut_vals.end());
  tf::scalar_field_intersections<Index, tf::coordinate_type<Policy>,
                                 tf::coordinate_dims_v<Policy>>
      sfi;
  sfi.build_many(polygons, scalars, cut_vals);
  tf::scalar_cut_faces<Index> scf;
  scf.build(polygons, sfi);
  return cut::embedded_isocurves<Index>(polygons, sfi, scf, scalars,
                                        tf::make_range(cut_vals));
}

template <typename Index, typename Policy, typename Range0, typename Iterator0,
          std::size_t N0>
auto embedded_isocurves(const tf::polygons<Policy> &polygons,
                        const Range0 &scalars,
                        const tf::range<Iterator0, N0> &cut_values,
                        tf::return_curves_t) {
  tf::buffer<std::decay_t<decltype(cut_values[0])>> cut_vals;
  cut_vals.reserve(cut_values.size());
  std::copy(cut_values.begin(), cut_values.end(), std::back_inserter(cut_vals));
  std::sort(cut_vals.begin(), cut_vals.end());
  tf::scalar_field_intersections<Index, tf::coordinate_type<Policy>,
                                 tf::coordinate_dims_v<Policy>>
      sfi;
  sfi.build_many(polygons, scalars, cut_vals);
  tf::scalar_cut_faces<Index> scf;
  scf.build(polygons, sfi);
  auto res_polygons = cut::embedded_isocurves<Index>(
      polygons, sfi, scf, scalars, tf::make_range(cut_vals));
  auto ie = tf::make_mapped_range(scf.intersection_edges(), [](auto e) {
    return std::array<Index, 2>{e[0].id, e[1].id};
  });
  auto paths = tf::connect_edges_to_paths(tf::make_edges(ie));
  tf::curves_buffer<Index, tf::coordinate_type<Policy>,
                    tf::coordinate_dims_v<Policy>>
      cb;
  cb.paths_buffer() = std::move(paths);
  cb.points_buffer().allocate(sfi.intersection_points().size());
  tf::parallel_copy(sfi.intersection_points(), cb.points());
  return std::make_pair(std::move(res_polygons), std::move(cb));
}
} // namespace tf
