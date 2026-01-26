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

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <optional>
#include <string>
#include <trueform/core/frame.hpp>
#include <trueform/core/points.hpp>
#include <trueform/core/policy/frame.hpp>
#include <trueform/core/polygons.hpp>
#include <trueform/core/range.hpp>
#include <trueform/core/transformation_view.hpp>
#include <trueform/core/views/blocked_range.hpp>
#include <trueform/io/write_obj.hpp>

namespace tf::py {

/// @brief Template implementation for write_obj
/// @tparam Index The index type (int or int64_t)
/// @tparam RealT The real type for points (float or double)
/// @tparam Ngon The number of vertices per face (3 or 4)
/// @param faces Numpy array of face indices (N, Ngon) with dtype Index
/// @param points Numpy array of points (M, 3) with dtype RealT
/// @param transformation_opt Optional transformation matrix (4, 4) with dtype
/// float32
/// @param filename Output filename
/// @return true if write succeeded, false otherwise
template <typename Index, typename RealT, std::size_t Ngon>
auto write_obj_impl(
    nanobind::ndarray<nanobind::numpy, Index, nanobind::shape<-1, Ngon>>
        faces_array,
    nanobind::ndarray<nanobind::numpy, RealT, nanobind::shape<-1, 3>>
        points_array,
    std::optional<
        nanobind::ndarray<nanobind::numpy, float, nanobind::shape<4, 4>>>
        transformation_opt,
    const std::string &filename) -> bool {

  // Create view into numpy arrays to build polygons
  RealT *data_pts = static_cast<RealT *>(points_array.data());
  std::size_t count_pts = points_array.shape(0) * 3;
  auto pts = tf::make_points<3>(tf::make_range(data_pts, count_pts));

  Index *data_fcs = static_cast<Index *>(faces_array.data());
  std::size_t count_fcs = faces_array.shape(0) * Ngon;
  auto faces =
      tf::make_blocked_range<Ngon>(tf::make_range(data_fcs, count_fcs));

  auto polygons = tf::make_polygons(faces, pts);

  // Apply transformation if provided
  if (transformation_opt.has_value()) {
    const auto &trans_array = *transformation_opt;
    auto transformation_view =
        tf::make_transformation_view<3>(trans_array.data());
    auto transformed_polygons =
        polygons | tf::tag(tf::make_frame(transformation_view));
    return tf::write_obj(transformed_polygons, filename);
  } else {
    return tf::write_obj(polygons, filename);
  }
}

auto register_io_write_obj(nanobind::module_ &m) -> void;

} // namespace tf::py
