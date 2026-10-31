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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/tuple.h>
#include <trueform/core/blocked_buffer.hpp>
#include <trueform/core/buffer.hpp>
#include <trueform/core/offset_block_buffer.hpp>
#include <trueform/fill/fill_holes.hpp>
#include <trueform/fill/hole_fill_config.hpp>
#include <trueform/fill/hole_fill_result.hpp>
#include <trueform/fill/make_filled_mesh.hpp>
#include <trueform/python/spatial/mesh.hpp>
#include <trueform/python/util/make_numpy_array.hpp>
#include <trueform/topology/boundary_rims.hpp>
#include <trueform/topology/policy/face_membership.hpp>
#include <utility>

namespace tf::py {

/// Copy one (offsets, data) pair of NumPy arrays into an offset block.
template <typename Index>
auto fill_offset_blocks_from(
    nanobind::ndarray<nanobind::numpy, const Index, nanobind::shape<-1>>
        offsets,
    nanobind::ndarray<nanobind::numpy, const Index, nanobind::shape<-1>> data,
    tf::offset_block_buffer<Index, Index> &out) -> void {
  out.offsets_buffer().allocate(offsets.shape(0));
  std::copy(offsets.data(), offsets.data() + offsets.shape(0),
            out.offsets_buffer().begin());
  out.data_buffer().allocate(data.shape(0));
  std::copy(data.data(), data.data() + data.shape(0),
            out.data_buffer().begin());
}

/// Copy a buffer of fixed-size records into one flat buffer of its scalars.
template <typename T, typename Record>
auto make_flat_buffer(const tf::buffer<Record> &records, std::size_t width)
    -> tf::buffer<T> {
  tf::buffer<T> flat;
  flat.allocate(records.size() * width);
  for (std::size_t k = 0; k < records.size(); ++k)
    for (std::size_t c = 0; c < width; ++c)
      flat[k * width + c] = T(records[k][c]);
  return flat;
}

/// A fill's result as NumPy lanes: group offsets; triangles and minted
/// points as (offsets, rows); status, charted, refined and faired as uint8;
/// seam angles; offending (rim, edge) pairs; splits as (S, 4) rows of
/// (face, v0, v1, point) beside their uint8 parameters; plan faces and plan
/// triangles as (offsets, rows).
template <typename Index, typename RealT>
auto make_fill_result_tuple(tf::hole_fill_result<Index, RealT> &&result) {
  const std::size_t groups = result.size();
  const auto lane = [groups](const auto &statuses) {
    tf::buffer<std::uint8_t> out;
    out.allocate(groups);
    for (std::size_t g = 0; g < groups; ++g)
      out[g] = std::uint8_t(statuses[g]);
    return make_numpy_array(std::move(out));
  };
  const std::size_t n_triangles = result.triangles.data_buffer().size();
  auto triangles = make_flat_buffer<Index>(result.triangles.data_buffer(), 3);
  const std::size_t n_minted = result.minted_points.data_buffer().size();
  auto minted = make_flat_buffer<RealT>(result.minted_points.data_buffer(), 3);
  tf::buffer<Index> splits;
  tf::buffer<std::uint8_t> parameters;
  for (const auto &split : result.splits) {
    splits.push_back(split.face);
    splits.push_back(split.v0);
    splits.push_back(split.v1);
    splits.push_back(split.point);
    parameters.push_back(split.parameter);
  }
  const std::size_t n_splits = result.splits.size();
  const std::size_t n_plan = result.plan_triangles.data_buffer().size();
  auto plan = make_flat_buffer<Index>(result.plan_triangles.data_buffer(), 3);
  return nanobind::make_tuple(
      make_numpy_array(std::move(result.group_offsets)),
      nanobind::make_tuple(
          make_numpy_array(std::move(result.triangles.offsets_buffer())),
          make_numpy_array(std::move(triangles), {n_triangles, 3})),
      nanobind::make_tuple(
          make_numpy_array(std::move(result.minted_points.offsets_buffer())),
          make_numpy_array(std::move(minted), {n_minted, 3})),
      lane(result.status), lane(result.charted), lane(result.refined),
      lane(result.faired), make_numpy_array(std::move(result.seam_max_angle)),
      make_numpy_array(std::move(result.offending)),
      make_numpy_array(std::move(splits), {n_splits, 4}),
      make_numpy_array(std::move(parameters)),
      make_numpy_array(std::move(result.plan_faces)),
      nanobind::make_tuple(
          make_numpy_array(std::move(result.plan_triangles.offsets_buffer())),
          make_numpy_array(std::move(plan), {n_plan, 3})));
}

/// tf::fill_holes over a mesh and the rims tf.boundary_rims states for it,
/// each rim lane handed as its (offsets, data) arrays.
template <typename Index, typename RealT, std::size_t Ngon>
auto fill_holes_impl(
    mesh_wrapper<Index, RealT, Ngon, 3> &mesh,
    nanobind::ndarray<nanobind::numpy, const Index, nanobind::shape<-1>>
        vertex_offsets,
    nanobind::ndarray<nanobind::numpy, const Index, nanobind::shape<-1>>
        vertices,
    nanobind::ndarray<nanobind::numpy, const Index, nanobind::shape<-1>>
        face_offsets,
    nanobind::ndarray<nanobind::numpy, const Index, nanobind::shape<-1>> faces,
    nanobind::ndarray<nanobind::numpy, const std::int8_t, nanobind::shape<-1>>
        closed,
    double min_quality, bool fairing, bool split_rim) {
  tf::boundary_rims<Index> rims;
  fill_offset_blocks_from(vertex_offsets, vertices, rims.vertices);
  fill_offset_blocks_from(face_offsets, faces, rims.faces);
  rims.closed.allocate(closed.shape(0));
  for (std::size_t r = 0; r < closed.shape(0); ++r)
    rims.closed[r] = closed.data()[r] != 0;
  tf::hole_fill_config config;
  config.min_quality = min_quality;
  config.fairing = fairing;
  config.split_rim = split_rim;
  auto form = mesh.make_primitive_range() | tf::tag(mesh.face_membership());
  return make_fill_result_tuple(tf::fill_holes(form, rims, config));
}

/// Copy one (offsets, rows) pair of NumPy arrays into an offset block of
/// fixed-size records.
template <typename Index, typename T, std::size_t Width, typename Record>
auto fill_blocks_from(
    nanobind::ndarray<nanobind::numpy, const Index, nanobind::shape<-1>>
        offsets,
    nanobind::ndarray<nanobind::numpy, const T, nanobind::shape<-1, Width>>
        rows,
    tf::offset_block_buffer<Index, Record> &out) -> void {
  out.offsets_buffer().allocate(offsets.shape(0));
  for (std::size_t k = 0; k < offsets.shape(0); ++k)
    out.offsets_buffer()[k] = offsets.data()[k];
  out.data_buffer().allocate(rows.shape(0));
  for (std::size_t k = 0; k < rows.shape(0); ++k)
    for (std::size_t c = 0; c < Width; ++c)
      out.data_buffer()[k][c] = rows.data()[k * Width + c];
}

/// tf::make_filled_mesh over the lanes of a fill result it replays: the
/// patches, the minted points, and the plans of the carrying faces.
template <typename Index, typename RealT, std::size_t Ngon>
auto filled_mesh_impl(
    mesh_wrapper<Index, RealT, Ngon, 3> &mesh,
    nanobind::ndarray<nanobind::numpy, const Index, nanobind::shape<-1>>
        triangle_offsets,
    nanobind::ndarray<nanobind::numpy, const Index, nanobind::shape<-1, 3>>
        triangles,
    nanobind::ndarray<nanobind::numpy, const Index, nanobind::shape<-1>>
        minted_offsets,
    nanobind::ndarray<nanobind::numpy, const RealT, nanobind::shape<-1, 3>>
        minted,
    nanobind::ndarray<nanobind::numpy, const Index, nanobind::shape<-1>>
        plan_faces,
    nanobind::ndarray<nanobind::numpy, const Index, nanobind::shape<-1>>
        plan_offsets,
    nanobind::ndarray<nanobind::numpy, const Index, nanobind::shape<-1, 3>>
        plan_triangles) {
  tf::hole_fill_result<Index, RealT> result;
  fill_blocks_from<Index, Index, 3>(triangle_offsets, triangles,
                                    result.triangles);
  fill_blocks_from<Index, RealT, 3>(minted_offsets, minted,
                                    result.minted_points);
  result.plan_faces.allocate(plan_faces.shape(0));
  for (std::size_t k = 0; k < plan_faces.shape(0); ++k)
    result.plan_faces[k] = plan_faces.data()[k];
  fill_blocks_from<Index, Index, 3>(plan_offsets, plan_triangles,
                                    result.plan_triangles);
  return make_numpy_array(
      tf::make_filled_mesh(mesh.make_primitive_range(), result));
}

} // namespace tf::py
