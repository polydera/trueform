/*
 * Copyright (c) 2026 XLAB
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
#include "../../core/buffer.hpp"
#include "../../core/cross.hpp"
#include "../../core/dot.hpp"
#include "../../core/point.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>

namespace tf::fill {

/// One triangle's contribution to one edge's cotangent weight, carried until
/// the contributions of an edge stand together.
///
/// `seq` is the slot the contribution was stated from — the triangle's
/// position in the canonical block times three plus the edge's own — so an
/// edge sums its weights in the one order the stencil states them in.
template <typename Index> struct hole_cotangent_record {
  Index v0;
  Index v1;
  Index seq;
  double weight;
};

/// The discrete operators one stencil states: the cotangent Laplacian in
/// compressed rows whose columns ascend, and the barycentric lumped mass.
///
/// @tparam Index The index type the stencil is numbered in.
template <typename Index> struct hole_fairing_system {
  tf::buffer<Index> row_offsets;
  tf::buffer<Index> columns;
  tf::buffer<double> weights;
  tf::buffer<double> diagonal;
  tf::buffer<double> mass;

  auto size() const -> std::size_t { return mass.size(); }
};

/// The scratch one assembly walks on.
template <typename Index> struct hole_fairing_assembly_scratch {
  tf::buffer<tf::fill::hole_cotangent_record<Index>> records;
  tf::buffer<Index> cursor;
};

/// The frame a stencil is solved in: its own box's centre, and its diagonal
/// as the unit of length. A box with no diagonal states no frame.
inline auto hole_fairing_frame(
    const tf::buffer<tf::point<double, 3>> &positions,
    tf::point<double, 3> &centre, double &scale) -> bool {
  if (!positions.size())
    return false;
  auto low = positions[0], high = positions[0];
  for (const auto &position : positions)
    for (std::size_t axis = 0; axis < 3; ++axis) {
      low[axis] = std::min(low[axis], position[axis]);
      high[axis] = std::max(high[axis], position[axis]);
    }
  const auto span = high - low;
  scale = std::sqrt(tf::dot(span, span));
  for (std::size_t axis = 0; axis < 3; ++axis)
    centre[axis] = 0.5 * (low[axis] + high[axis]);
  return std::isfinite(scale) && scale > 0.0;
}

/// Assemble the cotangent Laplacian and the lumped mass of a stencil.
///
/// The block's order is the sum's order: every triangle states its three edge
/// weights and its three mass shares in the order it stands in, an edge's
/// weights are summed in that order, and a row's diagonal is summed over its
/// own ascending columns — so one stencil states one operator, whatever
/// assembled it.
///
/// Refuses a triangle that states no area, a weight that does not evaluate
/// finitely, and a vertex whose lumped mass is not positive; the mass is what
/// the operator divides by, so it is proven before any of it is read.
template <typename Index>
auto assemble_hole_fairing_system(
    const tf::buffer<std::array<Index, 3>> &triangles,
    const tf::buffer<tf::point<double, 3>> &positions,
    tf::fill::hole_fairing_assembly_scratch<Index> &scratch,
    tf::fill::hole_fairing_system<Index> &system) -> bool {
  const std::size_t n = positions.size();
  system.mass.allocate(n);
  for (std::size_t k = 0; k < n; ++k)
    system.mass[k] = 0.0;
  scratch.records.clear();

  for (std::size_t t = 0; t < triangles.size(); ++t) {
    const auto corners = triangles[t];
    const tf::point<double, 3> p[3] = {
        positions[std::size_t(corners[0])], positions[std::size_t(corners[1])],
        positions[std::size_t(corners[2])]};
    const auto normal = tf::cross(p[1] - p[0], p[2] - p[0]);
    const double two_area = std::sqrt(tf::dot(normal, normal));
    if (!std::isfinite(two_area) || two_area <= 0.0)
      return false;
    for (int corner = 0; corner < 3; ++corner)
      system.mass[std::size_t(corners[std::size_t(corner)])] +=
          two_area / 6.0;
    for (int slot = 0; slot < 3; ++slot) {
      const Index a = corners[std::size_t(slot)];
      const Index b = corners[std::size_t((slot + 1) % 3)];
      const auto u = p[std::size_t(slot)] - p[std::size_t((slot + 2) % 3)];
      const auto v = p[std::size_t((slot + 1) % 3)] -
                     p[std::size_t((slot + 2) % 3)];
      const double weight = 0.5 * tf::dot(u, v) / two_area;
      if (!std::isfinite(weight))
        return false;
      scratch.records.push_back({std::min(a, b), std::max(a, b),
                                 Index(3 * t + std::size_t(slot)), weight});
    }
  }
  for (std::size_t k = 0; k < n; ++k)
    if (!(system.mass[k] > 0.0) || !std::isfinite(system.mass[k]))
      return false;

  std::sort(scratch.records.begin(), scratch.records.end(),
            [](const tf::fill::hole_cotangent_record<Index> &x,
               const tf::fill::hole_cotangent_record<Index> &y) {
              if (x.v0 != y.v0)
                return x.v0 < y.v0;
              if (x.v1 != y.v1)
                return x.v1 < y.v1;
              return x.seq < y.seq;
            });

  std::size_t edges = 0;
  for (std::size_t k = 0; k < scratch.records.size();) {
    std::size_t run = k;
    double weight = 0.0;
    while (run < scratch.records.size() &&
           scratch.records[run].v0 == scratch.records[k].v0 &&
           scratch.records[run].v1 == scratch.records[k].v1)
      weight += scratch.records[run++].weight;
    scratch.records[edges] = {scratch.records[k].v0, scratch.records[k].v1,
                              Index(0), weight};
    ++edges;
    k = run;
  }
  scratch.records.erase_till_end(scratch.records.begin() +
                                 std::ptrdiff_t(edges));

  system.row_offsets.allocate(n + 1);
  scratch.cursor.allocate(n);
  for (std::size_t k = 0; k <= n; ++k)
    system.row_offsets[k] = Index(0);
  for (std::size_t k = 0; k < edges; ++k) {
    ++system.row_offsets[std::size_t(scratch.records[k].v0) + 1];
    ++system.row_offsets[std::size_t(scratch.records[k].v1) + 1];
  }
  for (std::size_t k = 0; k < n; ++k)
    system.row_offsets[k + 1] = Index(system.row_offsets[k + 1] +
                                      system.row_offsets[k]);
  for (std::size_t k = 0; k < n; ++k)
    scratch.cursor[k] = system.row_offsets[k];

  system.columns.allocate(std::size_t(system.row_offsets[n]));
  system.weights.allocate(std::size_t(system.row_offsets[n]));
  // The edges ascend by their lower then their higher end, so a row takes its
  // lower neighbours first and its higher ones after: its columns ascend.
  for (std::size_t k = 0; k < edges; ++k) {
    const auto &record = scratch.records[k];
    auto &low = scratch.cursor[std::size_t(record.v0)];
    system.columns[std::size_t(low)] = record.v1;
    system.weights[std::size_t(low)] = -record.weight;
    ++low;
    auto &high = scratch.cursor[std::size_t(record.v1)];
    system.columns[std::size_t(high)] = record.v0;
    system.weights[std::size_t(high)] = -record.weight;
    ++high;
  }

  system.diagonal.allocate(n);
  for (std::size_t k = 0; k < n; ++k) {
    double value = 0.0;
    for (Index slot = system.row_offsets[k]; slot < system.row_offsets[k + 1];
         ++slot)
      value -= system.weights[std::size_t(slot)];
    system.diagonal[k] = value;
  }
  return true;
}

} // namespace tf::fill
