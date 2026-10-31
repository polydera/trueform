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
#include "../../core/algorithm/block_reduce_sequenced_aggregate.hpp"
#include "../../core/algorithm/parallel_contains.hpp"
#include "../../core/algorithm/parallel_for.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/cross.hpp"
#include "../../core/dot.hpp"
#include "../../core/point.hpp"
#include "../../core/polygons.hpp"
#include "../../core/reallocate.hpp"
#include "../../core/static_size.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../topology/policy/face_membership.hpp"
#include "./laplacian_system.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>

namespace tf::geometry {

/// What the operator reads of one triangle: the weight of each of its edges,
/// edge `slot` running from corner `slot` to corner `slot + 1`, and twice its
/// area.
struct laplacian_face {
  std::array<double, 3> weights;
  double two_area;
};

/// One edge weight a row gathers: the neighbour it couples, and the slot
/// `3 t + slot` it was stated from.
template <typename Index> struct laplacian_record {
  Index column;
  Index seq;
  double weight;
};

/// The scratch one assembly walks on.
struct laplacian_assembly_scratch {
  tf::buffer<tf::geometry::laplacian_face> faces;
};

/// The cotangent weights: edge `slot` of a triangle — the edge from corner
/// `slot` to corner `slot + 1` — weighs half the cotangent of the angle at
/// the corner opposite it.
struct cotangent_weights {
  auto operator()(std::size_t, std::size_t slot,
                  const tf::point<double, 3> (&corners)[3],
                  double two_area) const -> double {
    const auto u = corners[slot] - corners[(slot + 2) % 3];
    const auto v = corners[(slot + 1) % 3] - corners[(slot + 2) % 3];
    return 0.5 * tf::dot(u, v) / two_area;
  }
};

/// Assemble the weighted Laplacian and the lumped mass of a triangle mesh
/// tagged with its face membership.
///
/// `weight_of(t, slot, corners, two_area)` states the weight of edge `slot`
/// of triangle `t` — the edge from corner `slot` to corner `slot + 1`,
/// opposite corner `slot + 2` — over the corners read in double precision,
/// and @ref tf::geometry::cotangent_weights states the cotangent ones.
///
/// Every triangle forms its @ref tf::geometry::laplacian_face once, and every
/// row is then gathered on its own from the triangles its membership names:
/// its edge weights summed per neighbour in the order the triangles state
/// them, its diagonal over its own ascending columns, its mass over its
/// triangles in ascending order — so one mesh states one operator, whatever
/// the partition. A triangle that names one point twice couples it to
/// nothing.
///
/// Refuses a weight that does not evaluate finitely and a point whose lumped
/// mass is not positive; the mass is what the operator divides by, so it is
/// proven before any of it is read.
template <typename Policy, typename Index, typename WeightOf>
auto assemble_laplacian_system(
    const tf::polygons<Policy> &polygons, const WeightOf &weight_of,
    tf::geometry::laplacian_assembly_scratch &scratch,
    tf::geometry::laplacian_system<Index> &system) -> bool {
  static_assert(tf::static_size_v<decltype(polygons.faces()[0])> == 3,
                "a Laplacian is assembled over triangles");
  static_assert(tf::has_face_membership_policy<Policy>,
                "the polygons must carry their face membership");
  const auto faces = polygons.faces();
  const auto points = polygons.points();
  const auto &membership = polygons.face_membership();
  const std::size_t n = points.size();

  auto &images = scratch.faces;
  images.allocate(faces.size());
  tf::parallel_for(
      std::size_t(0), faces.size(),
      [&](std::size_t first, std::size_t last) {
        for (std::size_t t = first; t < last; ++t) {
          const auto face = faces[t];
          const tf::point<double, 3> p[3] = {
              points[std::size_t(face[0])].template as<double>(),
              points[std::size_t(face[1])].template as<double>(),
              points[std::size_t(face[2])].template as<double>()};
          const auto normal = tf::cross(p[1] - p[0], p[2] - p[0]);
          auto &image = images[t];
          image.two_area = std::sqrt(tf::dot(normal, normal));
          for (std::size_t slot = 0; slot < 3; ++slot)
            image.weights[slot] = weight_of(t, slot, p, image.two_area);
        }
      },
      tf::checked);
  if (tf::parallel_contains(
          images,
          [](const tf::geometry::laplacian_face &image) {
            return !std::isfinite(image.weights[0]) ||
                   !std::isfinite(image.weights[1]) ||
                   !std::isfinite(image.weights[2]);
          },
          tf::checked))
    return false;

  struct local_t {
    tf::buffer<Index> touching;
    tf::buffer<tf::geometry::laplacian_record<Index>> records;
    tf::buffer<Index> counts;
    tf::buffer<Index> columns;
    tf::buffer<double> weights;
  };
  system.mass.allocate(n);
  system.diagonal.allocate(n);
  system.row_offsets.clear();
  system.row_offsets.push_back(Index(0));
  system.columns.clear();
  system.weights.clear();
  tf::blocked_reduce_sequenced_aggregate(
      tf::make_sequence_range(n), system, local_t{},
      [&](const auto &rows, local_t &local) {
        local.counts.clear();
        local.columns.clear();
        local.weights.clear();
        for (std::size_t v : rows) {
          local.touching.clear();
          for (const auto t : membership[v])
            local.touching.push_back(Index(t));
          std::sort(local.touching.begin(), local.touching.end());
          local.records.clear();
          double mass = 0.0;
          for (const auto t : local.touching) {
            const auto face = faces[std::size_t(t)];
            const auto &image = images[std::size_t(t)];
            for (std::size_t corner = 0; corner < 3; ++corner) {
              if (std::size_t(face[corner]) != v)
                continue;
              mass += image.two_area / 6.0;
              const std::size_t before = (corner + 2) % 3;
              const Index next = Index(face[(corner + 1) % 3]);
              const Index previous = Index(face[before]);
              if (std::size_t(next) != v)
                local.records.push_back({next,
                                         Index(3 * std::size_t(t) + corner),
                                         image.weights[corner]});
              if (std::size_t(previous) != v)
                local.records.push_back({previous,
                                         Index(3 * std::size_t(t) + before),
                                         image.weights[before]});
            }
          }
          std::sort(local.records.begin(), local.records.end(),
                    [](const tf::geometry::laplacian_record<Index> &x,
                       const tf::geometry::laplacian_record<Index> &y) {
                      return x.column != y.column ? x.column < y.column
                                                  : x.seq < y.seq;
                    });
          const std::size_t first = local.columns.size();
          double diagonal = 0.0;
          for (std::size_t k = 0; k < local.records.size();) {
            double weight = 0.0;
            std::size_t run = k;
            while (run < local.records.size() &&
                   local.records[run].column == local.records[k].column)
              weight += local.records[run++].weight;
            local.columns.push_back(local.records[k].column);
            local.weights.push_back(-weight);
            diagonal += weight;
            k = run;
          }
          local.counts.push_back(Index(local.columns.size() - first));
          system.mass[v] = mass;
          system.diagonal[v] = diagonal;
        }
      },
      [](const local_t &local, tf::geometry::laplacian_system<Index> &out) {
        for (auto count : local.counts)
          out.row_offsets.push_back(Index(out.row_offsets.back() + count));
        tf::core::append(local.columns, out.columns);
        tf::core::append(local.weights, out.weights);
      },
      tf::checked);

  return !tf::parallel_contains(
      system.mass,
      [](double mass) { return !(mass > 0.0) || !std::isfinite(mass); },
      tf::checked);
}

} // namespace tf::geometry
