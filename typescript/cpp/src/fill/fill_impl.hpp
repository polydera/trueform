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

#include "trueform/core/buffer.hpp"
#include "trueform/core/offset_block_buffer.hpp"
#include "trueform/fill/fill_holes.hpp"
#include "trueform/fill/hole_fill_config.hpp"
#include "trueform/fill/hole_fill_result.hpp"
#include "trueform/fill/make_filled_mesh.hpp"
#include "trueform/topology/boundary_rims.hpp"
#include "trueform/topology/policy/face_membership.hpp"
#include "trueform/ts/core/promise.hpp"
#include "trueform/ts/core/wasm_mesh.hpp"
#include "trueform/ts/core/wasm_ndarray.hpp"
#include "trueform/ts/core/wasm_offset_blocked_buffer.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace tf {
namespace ts {

/// A fill's result as WASM lanes. Jagged lanes travel as their offsets beside
/// rows: `triangles` and `plan_triangles` are [T, 3], `minted_points` is
/// [M, 3], `offending` is [G, 2] and `splits` is [S, 4] of
/// (face, v0, v1, point). Statuses are int8.
template <typename Real> struct hole_fill_t {
  wasm_ndarray<int> group_offsets;
  wasm_ndarray<int> triangle_offsets;
  wasm_ndarray<int> triangles;
  wasm_ndarray<int> minted_offsets;
  wasm_ndarray<Real> minted_points;
  wasm_ndarray<std::int8_t> status;
  wasm_ndarray<std::int8_t> charted;
  wasm_ndarray<std::int8_t> refined;
  wasm_ndarray<std::int8_t> faired;
  wasm_ndarray<Real> seam_max_angle;
  wasm_ndarray<int> offending;
  wasm_ndarray<int> splits;
  wasm_ndarray<int> split_parameters;
  wasm_ndarray<int> plan_faces;
  wasm_ndarray<int> plan_offsets;
  wasm_ndarray<int> plan_triangles;
};

template <typename T, typename Record>
auto make_fill_rows(const tf::buffer<Record> &records, int width)
    -> wasm_ndarray<T> {
  tf::buffer<T> flat;
  flat.allocate(records.size() * std::size_t(width));
  for (std::size_t k = 0; k < records.size(); ++k)
    for (int c = 0; c < width; ++c)
      flat[k * width + c] = T(records[k][c]);
  return wasm_ndarray<T>::from_buffer(std::move(flat),
                                      {int(records.size()), width});
}

template <typename Status>
auto make_fill_statuses(const tf::buffer<Status> &statuses)
    -> wasm_ndarray<std::int8_t> {
  tf::buffer<std::int8_t> out;
  out.allocate(statuses.size());
  for (std::size_t g = 0; g < statuses.size(); ++g)
    out[g] = std::int8_t(statuses[g]);
  return wasm_ndarray<std::int8_t>::from_buffer(std::move(out));
}

template <typename Real>
auto make_hole_fill(tf::hole_fill_result<int, Real> &&result)
    -> hole_fill_t<Real> {
  const auto n_groups = int(result.size());
  tf::buffer<int> splits;
  tf::buffer<int> parameters;
  for (const auto &split : result.splits) {
    splits.push_back(split.face);
    splits.push_back(split.v0);
    splits.push_back(split.v1);
    splits.push_back(split.point);
    parameters.push_back(split.parameter);
  }
  const auto n_splits = int(result.splits.size());
  return {
      wasm_ndarray<int>::from_buffer(std::move(result.group_offsets)),
      wasm_ndarray<int>::from_buffer(
          std::move(result.triangles.offsets_buffer())),
      make_fill_rows<int>(result.triangles.data_buffer(), 3),
      wasm_ndarray<int>::from_buffer(
          std::move(result.minted_points.offsets_buffer())),
      make_fill_rows<Real>(result.minted_points.data_buffer(), 3),
      make_fill_statuses(result.status),
      make_fill_statuses(result.charted),
      make_fill_statuses(result.refined),
      make_fill_statuses(result.faired),
      wasm_ndarray<Real>::from_buffer(std::move(result.seam_max_angle)),
      wasm_ndarray<int>::from_buffer(
          std::move(result.offending.data_buffer()), {n_groups, 2}),
      wasm_ndarray<int>::from_buffer(std::move(splits), {n_splits, 4}),
      wasm_ndarray<int>::from_buffer(std::move(parameters)),
      wasm_ndarray<int>::from_buffer(std::move(result.plan_faces)),
      wasm_ndarray<int>::from_buffer(
          std::move(result.plan_triangles.offsets_buffer())),
      make_fill_rows<int>(result.plan_triangles.data_buffer(), 3)};
}

inline auto copy_fill_blocks(const wasm_offset_blocked_buffer<int, int> &from,
                             tf::offset_block_buffer<int, int> &to) -> void {
  const auto &offsets = *from.raw_offsets();
  const auto &data = *from.raw_data();
  to.offsets_buffer().allocate(offsets.size());
  std::copy(offsets.begin(), offsets.end(), to.offsets_buffer().begin());
  to.data_buffer().allocate(data.size());
  std::copy(data.begin(), data.end(), to.data_buffer().begin());
}

template <typename T, typename Record>
auto copy_fill_rows(const wasm_ndarray<int> &offsets,
                    const wasm_ndarray<T> &rows,
                    tf::offset_block_buffer<int, Record> &to) -> void {
  to.offsets_buffer().allocate(offsets.length());
  std::copy(offsets.raw_data(), offsets.raw_data() + offsets.length(),
            to.offsets_buffer().begin());
  const std::size_t n = rows.length() / 3;
  to.data_buffer().allocate(n);
  const T *in = rows.raw_data();
  for (std::size_t k = 0; k < n; ++k)
    for (std::size_t c = 0; c < 3; ++c)
      to.data_buffer()[k][c] = in[k * 3 + c];
}

template <typename Real>
auto sync_fill_holes(wasm_mesh<Real> &m,
                     wasm_offset_blocked_buffer<int, int> &rim_vertices,
                     wasm_offset_blocked_buffer<int, int> &rim_faces,
                     wasm_ndarray<std::int8_t> &closed, double min_quality,
                     bool fairing, bool split_rim) -> hole_fill_t<Real> {
  tf::boundary_rims<int> rims;
  copy_fill_blocks(rim_vertices, rims.vertices);
  copy_fill_blocks(rim_faces, rims.faces);
  rims.closed.allocate(closed.length());
  for (std::size_t r = 0; r < closed.length(); ++r)
    rims.closed[r] = closed.raw_data()[r] != 0;
  tf::hole_fill_config config;
  config.min_quality = min_quality;
  config.fairing = fairing;
  config.split_rim = split_rim;
  auto poly = m.polygons_range() | tf::tag(m.face_membership_range());
  return make_hole_fill<Real>(tf::fill_holes(poly, rims, config));
}

template <typename Real>
auto sync_filled_mesh(wasm_mesh<Real> &m, wasm_ndarray<int> &triangle_offsets,
                      wasm_ndarray<int> &triangles,
                      wasm_ndarray<int> &minted_offsets,
                      wasm_ndarray<Real> &minted_points,
                      wasm_ndarray<int> &plan_faces,
                      wasm_ndarray<int> &plan_offsets,
                      wasm_ndarray<int> &plan_triangles) -> wasm_mesh<Real> {
  tf::hole_fill_result<int, Real> result;
  copy_fill_rows(triangle_offsets, triangles, result.triangles);
  copy_fill_rows(minted_offsets, minted_points, result.minted_points);
  result.plan_faces.allocate(plan_faces.length());
  std::copy(plan_faces.raw_data(), plan_faces.raw_data() + plan_faces.length(),
            result.plan_faces.begin());
  copy_fill_rows(plan_offsets, plan_triangles, result.plan_triangles);
  return wasm_mesh<Real>::from_polygons_buffer(
      tf::make_filled_mesh(m.polygons_range(), result));
}

template <typename Real>
auto async_fill_holes(wasm_mesh<Real> &m,
                      wasm_offset_blocked_buffer<int, int> &rim_vertices,
                      wasm_offset_blocked_buffer<int, int> &rim_faces,
                      wasm_ndarray<std::int8_t> &closed, double min_quality,
                      bool fairing, bool split_rim) -> promise_t {
  return promise([a = m, v = rim_vertices, f = rim_faces, c = closed,
                  min_quality, fairing, split_rim]() -> hole_fill_t<Real> {
    return sync_fill_holes<Real>(
        const_cast<wasm_mesh<Real> &>(a),
        const_cast<wasm_offset_blocked_buffer<int, int> &>(v),
        const_cast<wasm_offset_blocked_buffer<int, int> &>(f),
        const_cast<wasm_ndarray<std::int8_t> &>(c), min_quality, fairing,
        split_rim);
  });
}

template <typename Real>
auto async_filled_mesh(wasm_mesh<Real> &m, wasm_ndarray<int> &triangle_offsets,
                       wasm_ndarray<int> &triangles,
                       wasm_ndarray<int> &minted_offsets,
                       wasm_ndarray<Real> &minted_points,
                       wasm_ndarray<int> &plan_faces,
                       wasm_ndarray<int> &plan_offsets,
                       wasm_ndarray<int> &plan_triangles) -> promise_t {
  return promise([a = m, to = triangle_offsets, t = triangles,
                  mo = minted_offsets, mp = minted_points, pf = plan_faces,
                  po = plan_offsets, pt = plan_triangles]() -> wasm_mesh<Real> {
    return sync_filled_mesh<Real>(
        const_cast<wasm_mesh<Real> &>(a), const_cast<wasm_ndarray<int> &>(to),
        const_cast<wasm_ndarray<int> &>(t), const_cast<wasm_ndarray<int> &>(mo),
        const_cast<wasm_ndarray<Real> &>(mp),
        const_cast<wasm_ndarray<int> &>(pf), const_cast<wasm_ndarray<int> &>(po),
        const_cast<wasm_ndarray<int> &>(pt));
  });
}

} // namespace ts
} // namespace tf
