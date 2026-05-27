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
#include "../core/algorithm/parallel_for_each.hpp"
#include "../core/buffer.hpp"
#include "../core/constants.hpp"
#include "../core/curves.hpp"
#include "../core/dot.hpp"
#include "../core/epsilon.hpp"
#include "../core/polygons_buffer.hpp"
#include "../core/unit_vectors_buffer.hpp"
#include "../core/views/sequence_range.hpp"
#include "../core/views/slice.hpp"
#include "./make_curve_frames.hpp"

namespace tf {

/// @ingroup geometry_meshing
/// @brief Creates a tube mesh around a single curve.
///
/// Generates a triangulated tube using parallel transport frames.
/// Closed curves (first ~ last point) are detected automatically
/// and produce a seamless manifold tube.
///
/// @tparam Index The index type (default: int).
/// @param curve The input curve.
/// @param radius The tube radius.
/// @param radial_segments Vertices per cross-section ring.
/// @return A polygons_buffer containing the tube triangle mesh.
template <typename Index = int, typename CurvePolicy, typename RealT>
auto make_tube_mesh(const tf::curve<3, CurvePolicy> &curve, RealT radius,
               Index radial_segments = 8)
    -> tf::polygons_buffer<Index, RealT, 3, 3> {
  tf::polygons_buffer<Index, RealT, 3, 3> mesh;

  const auto n = static_cast<std::size_t>(curve.size());
  if (n < 2)
    return mesh;

  const auto R = static_cast<std::size_t>(radial_segments);

  // Detect closed
  auto diff = curve[n - 1] - curve[0];
  bool closed = tf::dot(diff, diff) < tf::epsilon<RealT> * tf::epsilon<RealT>;
  const auto ring_count = closed ? n - 1 : n;
  const auto num_connections = closed ? ring_count : ring_count - 1;

  mesh.points_buffer().allocate(static_cast<Index>(ring_count * R));
  mesh.faces_buffer().allocate(static_cast<Index>(num_connections * R * 2));

  // Compute frames
  auto [tangents, normals, binormals] = tf::make_curve_frames(curve);

  auto &points = mesh.points_buffer();
  auto &faces = mesh.faces_buffer();
  auto inv_R = RealT{1} / static_cast<RealT>(R);

  tbb::task_group tg;

  // Ring vertices (parallel)
  tg.run([&, &normals = normals, &binormals = binormals] {
    tf::parallel_for_each(
        tf::make_sequence_range(ring_count),
        [&, &normals = normals, &binormals = binormals](std::size_t i) {
          auto pt = curve[i];
          auto &&ni = normals[i];
          auto &&bi = binormals[i];
          for (std::size_t j = 0; j < R; ++j) {
            auto theta = tf::two_pi<RealT> * static_cast<RealT>(j) * inv_R;
            points[static_cast<Index>(i * R + j)] =
                pt + ni * (radius * std::cos(theta)) +
                bi * (radius * std::sin(theta));
          }
        },
        tf::checked);
  });

  // Face indices (parallel)
  tg.run([&] {
    tf::parallel_for_each(
        tf::make_sequence_range(num_connections),
        [&](std::size_t i) {
          auto next = closed ? ((i + 1) % ring_count) : (i + 1);
          for (std::size_t j = 0; j < R; ++j) {
            auto jn = (j + 1) % R;
            auto a = static_cast<Index>(i * R + j);
            auto b = static_cast<Index>(i * R + jn);
            auto c = static_cast<Index>(next * R + j);
            auto d = static_cast<Index>(next * R + jn);

            auto fi = static_cast<Index>((i * R + j) * 2);
            faces[fi] = tf::make_array_like(std::array<Index, 3>{a, c, d});
            faces[fi + 1] = tf::make_array_like(std::array<Index, 3>{a, d, b});
          }
        },
        tf::checked);
  });

  tg.wait();
  return mesh;
}

/// @ingroup geometry_meshing
/// @brief Creates tube meshes around multiple curves.
///
/// Each curve is processed in parallel. Closed curves are detected
/// automatically. Single allocation for all frame buffers.
///
/// @tparam Index The index type (default: int).
/// @param curves The input curves.
/// @param radius The tube radius.
/// @param radial_segments Vertices per cross-section ring.
/// @return A polygons_buffer containing all tubes combined.
template <typename Index = int, typename CurvesPolicy, typename RealT>
auto make_tube_mesh(const tf::curves<CurvesPolicy> &curves, RealT radius,
                Index radial_segments = 8)
    -> tf::polygons_buffer<Index, RealT, 3, 3> {
  tf::polygons_buffer<Index, RealT, 3, 3> mesh;

  const auto num_curves = curves.size();
  if (num_curves == 0)
    return mesh;

  const auto R = static_cast<std::size_t>(radial_segments);

  // Per-curve offsets
  tf::buffer<Index> pt_off;
  tf::buffer<Index> v_off;
  tf::buffer<Index> f_off;
  pt_off.allocate(num_curves + 1);
  v_off.allocate(num_curves + 1);
  f_off.allocate(num_curves + 1);
  pt_off[0] = 0;
  v_off[0] = 0;
  f_off[0] = 0;

  for (std::size_t ci = 0; ci < num_curves; ++ci) {
    auto c = curves[ci];
    auto n = static_cast<std::size_t>(c.size());
    pt_off[ci + 1] = pt_off[ci] + n;
    if (n < 2) {
      v_off[ci + 1] = v_off[ci];
      f_off[ci + 1] = f_off[ci];
      continue;
    }
    auto diff = c[n - 1] - c[0];
    bool closed = tf::dot(diff, diff) < tf::epsilon<RealT> * tf::epsilon<RealT>;
    auto ring_count = closed ? n - 1 : n;
    auto num_connections = closed ? ring_count : ring_count - 1;
    v_off[ci + 1] = v_off[ci] + static_cast<Index>(ring_count * R);
    f_off[ci + 1] = f_off[ci] + static_cast<Index>(num_connections * R * 2);
  }

  if (f_off[num_curves] == 0)
    return mesh;

  // Single allocation for all frames
  auto total_points = pt_off[num_curves];
  tf::unit_vectors_buffer<RealT, 3> tangents;
  tf::unit_vectors_buffer<RealT, 3> normals;
  tf::unit_vectors_buffer<RealT, 3> binormals;
  tangents.allocate(total_points);
  normals.allocate(total_points);
  binormals.allocate(total_points);

  // Compute frames for all curves in parallel
  tf::parallel_for_each(
      tf::make_sequence_range(num_curves),
      [&](std::size_t ci) {
        auto c = curves[ci];
        if (c.size() < 2)
          return;
        auto from = pt_off[ci];
        auto to = pt_off[ci + 1];
        tf::make_curve_frames(
            c,
            tf::make_unit_vectors(tf::slice(tangents.unit_vectors(), from, to)),
            tf::make_unit_vectors(tf::slice(normals.unit_vectors(), from, to)),
            tf::make_unit_vectors(tf::slice(binormals.unit_vectors(), from, to)));
      },
      tf::checked);

  mesh.points_buffer().allocate(v_off[num_curves]);
  mesh.faces_buffer().allocate(f_off[num_curves]);

  auto &points = mesh.points_buffer();
  auto &faces = mesh.faces_buffer();
  auto inv_R = RealT{1} / static_cast<RealT>(R);

  // Generate vertices and faces in parallel per curve
  tf::parallel_for_each(
      tf::make_sequence_range(num_curves),
      [&](std::size_t ci) {
        auto c = curves[ci];
        auto n = static_cast<std::size_t>(c.size());
        if (n < 2)
          return;

        auto diff = c[n - 1] - c[0];
        bool closed = tf::dot(diff, diff) <
                      tf::epsilon<RealT> * tf::epsilon<RealT>;
        auto ring_count = closed ? n - 1 : n;
        auto num_connections = closed ? ring_count : ring_count - 1;
        auto frame_off = pt_off[ci];
        auto vo = v_off[ci];
        auto fo = f_off[ci];

        // Vertices
        for (std::size_t i = 0; i < ring_count; ++i) {
          auto pt = c[i];
          auto &&ni = normals[frame_off + i];
          auto &&bi = binormals[frame_off + i];
          for (std::size_t j = 0; j < R; ++j) {
            auto theta = tf::two_pi<RealT> * static_cast<RealT>(j) * inv_R;
            points[vo + static_cast<Index>(i * R + j)] =
                pt + ni * (radius * std::cos(theta)) +
                bi * (radius * std::sin(theta));
          }
        }

        // Faces
        for (std::size_t i = 0; i < num_connections; ++i) {
          auto next = closed ? ((i + 1) % ring_count) : (i + 1);
          for (std::size_t j = 0; j < R; ++j) {
            auto jn = (j + 1) % R;
            auto a = vo + static_cast<Index>(i * R + j);
            auto b = vo + static_cast<Index>(i * R + jn);
            auto c_idx = vo + static_cast<Index>(next * R + j);
            auto d = vo + static_cast<Index>(next * R + jn);

            auto fi = fo + static_cast<Index>((i * R + j) * 2);
            faces[fi] = tf::make_array_like(std::array<Index, 3>{a, c_idx, d});
            faces[fi + 1] = tf::make_array_like(std::array<Index, 3>{a, d, b});
          }
        }
      },
      tf::checked);

  return mesh;
}

} // namespace tf
