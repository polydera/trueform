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
#include "../../core/point.hpp"
#include "./drain_constrained_delaunay_refinement_generation.hpp"
#include "./locate_constrained_delaunay_refinement_point.hpp"
#include "./queue_constrained_delaunay_refinement_encroachments.hpp"
#include "./queue_constrained_delaunay_refinement_triangle.hpp"
#include "./split_constrained_delaunay_refinement_edge.hpp"
#include "./split_constrained_delaunay_refinement_triangle.hpp"
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace tf::topology::cdt {

template <typename Owner>
auto refine_constrained_delaunay(Owner &owner, double min_quality) -> void {
  using Index = typename Owner::index_type;
  using Int = typename Owner::int_type;

  if (min_quality <= 0.0 && !owner._split_encroached)
    return;
  const std::int64_t point_budget =
      std::int64_t(64) * std::int64_t(owner._ip.size()) + 4096;
  owner._min_quality = min_quality;
  for (Index face = 0; face < Index(owner._t.size()); ++face)
    if (owner._label[std::size_t(face)] % 2 == 1)
      queue_constrained_delaunay_refinement_triangle(owner, face);

  std::size_t head = 0;
  while (head < owner._queue.size()) {
    if (std::int64_t(owner._ip.size()) >= point_budget)
      return;
    auto [face, stamp] = owner._queue[head++];
    if (head > (std::size_t(1) << 16) && head * 2 > owner._queue.size()) {
      owner._queue.erase(owner._queue.begin(), owner._queue.begin() + head);
      head = 0;
    }
    if (std::uint32_t(stamp) != owner._t[face].stamp)
      continue;
    if (queue_constrained_delaunay_refinement_encroachments(owner, face)) {
      drain_constrained_delaunay_refinement_generation(owner);
      continue;
    }
    if (owner.quality(face) >= min_quality)
      continue;

    const auto &a = owner._dp[std::size_t(owner._t[face].v[0])];
    const auto &b = owner._dp[std::size_t(owner._t[face].v[1])];
    const auto &c = owner._dp[std::size_t(owner._t[face].v[2])];
    double ex = b[0] - a[0], ey = b[1] - a[1];
    double fx = c[0] - a[0], fy = c[1] - a[1];
    double d = 2.0 * (ex * fy - ey * fx);
    if (d == 0)
      continue;
    double le2 = ex * ex + ey * ey, lf2 = fx * fx + fy * fy;
    double ccx = a[0] + (fy * le2 - ey * lf2) / d;
    double ccy = a[1] + (ex * lf2 - fx * le2) / d;
    double gx = ccx * owner._scale + owner._offx;
    double gy = ccy * owner._scale + owner._offy;
    const double lattice_max = double(std::numeric_limits<Int>::max()) * 0.5;
    if (!(std::abs(gx) < lattice_max) || !(std::abs(gy) < lattice_max))
      continue;

    Index point = Index(owner._ip.size());
    owner._ip.push_back(
        tf::point<Int, 2>{Int(std::llround(gx)), Int(std::llround(gy))});
    owner._dp.push_back(
        {(double(owner._ip[point][0]) - owner._offx) / owner._scale,
         (double(owner._ip[point][1]) - owner._offy) / owner._scale});

    auto [host, blocked] =
        locate_constrained_delaunay_refinement_point(owner, face, point);
    if (host == Owner::none) {
      owner._ip.pop_back();
      owner._dp.pop_back();
      continue;
    }
    if (blocked >= 0) {
      owner._ip.pop_back();
      owner._dp.pop_back();
      Index u = owner._t[host].v[blocked];
      Index v = owner._t[host].v[(blocked + 1) % 3];
      if (owner.encroaches(owner._dp[std::size_t(u)],
                           owner._dp[std::size_t(v)], {ccx, ccy}) &&
          owner.splittable(host, blocked))
        owner._pending.push_back(
            {host, Index(blocked), owner._t[host].stamp});
    } else {
      owner._cavity.clear();
      owner._cavity.push_back(host);
      bool encroaches_capped = false;
      typename Owner::split_request encroached_split{Owner::none, Index(0), 0};
      for (std::size_t cavity = 0;
           cavity < owner._cavity.size() && cavity < 64; ++cavity) {
        Index cavity_face = owner._cavity[cavity];
        for (int edge = 0; edge < 3; ++edge) {
          Index u = owner._t[cavity_face].v[edge];
          Index v = owner._t[cavity_face].v[(edge + 1) % 3];
          if (owner.constrained(cavity_face, edge)) {
            if (owner.encroaches(owner._dp[std::size_t(u)],
                                 owner._dp[std::size_t(v)],
                                 owner._dp[std::size_t(point)])) {
              if (owner.splittable(cavity_face, edge))
                encroached_split = {cavity_face, Index(edge),
                                    owner._t[cavity_face].stamp};
              else
                encroaches_capped = true;
            }
            continue;
          }
          Index neighbor = owner._t[cavity_face].n[edge];
          if (neighbor == Owner::none)
            continue;
          bool seen = false;
          for (Index value : owner._cavity)
            if (value == neighbor) {
              seen = true;
              break;
            }
          if (seen)
            continue;
          int back = owner.edge_back(neighbor, cavity_face);
          Index apex = owner._t[neighbor].v[(back + 2) % 3];
          if (owner.incircle(u, v,
                             owner._t[cavity_face].v[(edge + 2) % 3],
                             apex) >= 0 ||
              owner.incircle(owner._t[neighbor].v[back],
                             owner._t[neighbor].v[(back + 1) % 3], apex,
                             point) > 0)
            owner._cavity.push_back(neighbor);
        }
      }
      if (encroaches_capped || encroached_split.f != Owner::none) {
        owner._ip.pop_back();
        owner._dp.pop_back();
        if (encroached_split.f != Owner::none) {
          owner._pending.push_back(encroached_split);
          queue_constrained_delaunay_refinement_triangle(owner, face);
          drain_constrained_delaunay_refinement_generation(owner);
        }
        continue;
      }
      bool duplicate = false;
      for (int corner = 0; corner < 3; ++corner)
        if (owner._ip[owner._t[host].v[corner]][0] == owner._ip[point][0] &&
            owner._ip[owner._t[host].v[corner]][1] == owner._ip[point][1])
          duplicate = true;
      if (duplicate) {
        owner._ip.pop_back();
        owner._dp.pop_back();
        continue;
      }
      int on_edge = -1;
      for (int edge = 0; edge < 3; ++edge)
        if (owner.orient(owner._t[host].v[edge],
                         owner._t[host].v[(edge + 1) % 3], point) == 0) {
          on_edge = edge;
          break;
        }
      if (on_edge >= 0) {
        if (owner.constrained(host, on_edge) ||
            owner._t[host].n[on_edge] == Owner::none) {
          owner._ip.pop_back();
          owner._dp.pop_back();
          continue;
        }
        owner._con_nbr.push_back({Owner::none, Owner::none});
        split_constrained_delaunay_refinement_edge(
            owner, host, on_edge, point, Owner::none, Owner::none);
      } else {
        owner._con_nbr.push_back({Owner::none, Owner::none});
        split_constrained_delaunay_refinement_triangle(owner, host, point);
      }
    }

    drain_constrained_delaunay_refinement_generation(owner);
  }
}

} // namespace tf::topology::cdt
