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
#include "../../core/checked.hpp"
#include "../../core/cross.hpp"
#include "../../core/vector.hpp"
#include "../../core/views/sequence_range.hpp"
#include "./raw_winding_moment.hpp"
#include <cstddef>

namespace tf::spatial {

/// Fill each leaf's raw moment row from its primitives — fan
/// decomposition, double accumulation. The first pass settles the
/// leaf's area, dipole, and centroid; the second states the second
/// and third moments about that centroid, exactly: a fan triangle's
/// own first moment about its centroid vanishes, and the edge-midpoint
/// rule integrates its quadratic exactly. Every other row is zeroed,
/// so the sweep reads settled children and an empty node stays a zero
/// row.
template <typename Nodes, typename Ids, typename Polygons>
auto aggregate_winding_leaves(raw_winding_moment *moments, const Nodes &nodes,
                              const Ids &ids, const Polygons &polygons)
    -> void {
  tf::parallel_for_each(
      tf::make_sequence_range(std::size_t{0}, nodes.size()),
      [&](std::size_t i) {
        const auto &node = nodes[i];
        auto &m = moments[i];
        m = raw_winding_moment{};
        if (!node.is_leaf())
          return;
        const auto data = node.get_data();
        tf::vector<double, 3> centroid_sum{0, 0, 0};
        tf::vector<double, 3> directed_area{0, 0, 0};
        double area = 0;
        for (auto it = ids.begin() + data[0], end = it + data[1]; it != end;
             ++it) {
          const auto polygon = polygons[*it];
          const auto sz = std::size_t(polygon.size());
          const auto p0 = polygon[0].template as<double>();
          for (std::size_t j = 1; j + 1 < sz; ++j) {
            const auto p1 = polygon[j].template as<double>();
            const auto p2 = polygon[j + 1].template as<double>();
            const auto da = tf::cross(p1 - p0, p2 - p0) * 0.5;
            const auto a = da.length();
            for (std::size_t d = 0; d < 3; ++d)
              centroid_sum[d] += a * (p0[d] + p1[d] + p2[d]) / 3.;
            directed_area = directed_area + da;
            area += a;
          }
        }
        tf::vector<double, 3> centroid{0, 0, 0};
        if (area > 0)
          for (std::size_t d = 0; d < 3; ++d)
            centroid[d] = centroid_sum[d] / area;
        double second[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
        double third[18] = {0};
        for (auto it = ids.begin() + data[0], end = it + data[1]; it != end;
             ++it) {
          const auto polygon = polygons[*it];
          const auto sz = std::size_t(polygon.size());
          const auto p0 = polygon[0].template as<double>();
          for (std::size_t j = 1; j + 1 < sz; ++j) {
            const auto p1 = polygon[j].template as<double>();
            const auto p2 = polygon[j + 1].template as<double>();
            const auto da = tf::cross(p1 - p0, p2 - p0) * 0.5;
            for (std::size_t r = 0; r < 3; ++r)
              for (std::size_t c = 0; c < 3; ++c)
                second[3 * r + c] +=
                    da[r] * ((p0[c] + p1[c] + p2[c]) / 3. - centroid[c]);
            double mid[3][3];
            for (std::size_t d = 0; d < 3; ++d) {
              mid[0][d] = 0.5 * (p0[d] + p1[d]) - centroid[d];
              mid[1][d] = 0.5 * (p1[d] + p2[d]) - centroid[d];
              mid[2][d] = 0.5 * (p2[d] + p0[d]) - centroid[d];
            }
            double s6[6] = {0, 0, 0, 0, 0, 0};
            for (std::size_t e = 0; e < 3; ++e) {
              s6[0] += mid[e][0] * mid[e][0];
              s6[1] += mid[e][0] * mid[e][1];
              s6[2] += mid[e][0] * mid[e][2];
              s6[3] += mid[e][1] * mid[e][1];
              s6[4] += mid[e][1] * mid[e][2];
              s6[5] += mid[e][2] * mid[e][2];
            }
            for (std::size_t r = 0; r < 3; ++r)
              for (std::size_t k = 0; k < 6; ++k)
                third[6 * r + k] += da[r] * s6[k] / 3.;
          }
        }
        for (std::size_t d = 0; d < 3; ++d) {
          m.position[d] = centroid[d];
          m.directed_area[d] = directed_area[d];
        }
        for (std::size_t k = 0; k < 9; ++k)
          m.second_moment[k] = second[k];
        for (std::size_t k = 0; k < 18; ++k)
          m.third_moment[k] = third[k];
        m.area = area;
      },
      tf::checked);
}

} // namespace tf::spatial
