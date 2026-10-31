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
#include "../../core/algorithm/block_reduce.hpp"
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/constants.hpp"
#include "../../core/coordinate_type.hpp"
#include "../../core/determinant_of_rotation.hpp"
#include "../../core/frame_of.hpp"
#include "../../core/none.hpp"
#include "../../core/point.hpp"
#include "../../core/points_buffer.hpp"
#include "../../core/polygons.hpp"
#include "../../core/transformed.hpp"
#include "../../core/views/mapped_range.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../exact/meta.hpp"
#include "../../exact/vertex.hpp"
#include "../../spatial/policy/tree.hpp"
#include "../../spatial/policy/winding.hpp"
#include "../../spatial/winding_moments.hpp"
#include "../../spatial/winding_number.hpp"
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <type_traits>

namespace tf::csg::graph {

/// @ingroup csg_graph_internals
/// @brief The batch from which @ref winding_side answers through winding
///        moments: below it one pass over the sheet costs no more than
///        folding them.
inline constexpr std::size_t winding_moments_from = 2;

/// @ingroup csg_graph_internals
/// @brief Side of a sheet for a batch of query points, by generalized
///        winding number.
///
/// A query is the centroid of a lattice triangle, stated by its corners. A
/// batch below @ref winding_moments_from takes one parallel pass over the
/// sheet's triangles, accumulating the solid angle of every triangle at
/// every query at once, on the lattice. A larger batch folds the winding
/// moments of the sheet's own tree over the sheet where the reader states
/// it, in the form's own space, and answers each query through
/// @ref tf::winding_number. A reflecting frame negates every solid angle,
/// so the folded winding takes the sign of the frame's determinant; a frame
/// that is not a similarity bends the winding's zero set out of its own
/// space, so it keeps the pass on the lattice. Winding has no feature cases
/// where pseudo-normal classification fails.
///
/// @param form    The sheet form (faces + points, frame respected).
/// @param tag     The sheet's form tag in `reader`.
/// @param reader  The one reader of an original vertex's position
///        (@ref tf::exact::input_lattice_reader).
/// @param corners Per query, the corners of the triangle whose centroid
///        it is.
/// @return One char per query: 1 behind the sheet's normal, else 0.
template <typename Int, typename Form, typename Index, typename Reader>
auto winding_side(const Form &form, Index tag, const Reader &reader,
                  const tf::buffer<std::array<tf::exact::pt3<Int>, 3>> &corners)
    -> tf::buffer<char> {
  using T1 = typename tf::exact::meta<Int>::T1;
  using WidePt = tf::exact::pt3<T1>;
  const std::size_t m = corners.size();
  tf::buffer<char> side;
  side.allocate(m);
  if (m == 0)
    return side;

  const auto base = reader.vertex_offsets[std::size_t(tag)];
  const auto &frame = tf::frame_of(form);
  tf::buffer<double> total;
  total.allocate(m);

  const double orientation = double(tf::determinant_of_rotation(frame));
  // a similarity's columns are orthogonal and of one length, to the frame's
  // own precision
  bool similar = true;
  {
    const auto &t = frame.transformation();
    using real_t = std::decay_t<decltype(t(0, 0))>;
    double gram[3][3];
    for (std::size_t i = 0; i < 3; ++i)
      for (std::size_t j = 0; j < 3; ++j)
        gram[i][j] = double(t(0, i)) * double(t(0, j)) +
                     double(t(1, i)) * double(t(1, j)) +
                     double(t(2, i)) * double(t(2, j));
    const double s2 = (gram[0][0] + gram[1][1] + gram[2][2]) / 3.0;
    const double tol =
        64.0 * double(std::numeric_limits<real_t>::epsilon()) * s2;
    for (std::size_t i = 0; i < 3; ++i)
      for (std::size_t j = 0; j < 3; ++j)
        similar = similar && std::abs(gram[i][j] - (i == j ? s2 : 0.0)) <= tol;
  }

  using tree_t = std::decay_t<decltype(form.tree())>;
  constexpr bool static_tree = !decltype(tf::spatial::winding_is_mod_tree(
      static_cast<const tree_t *>(nullptr)))::value;
  auto linear = [&] {
    for (std::size_t j = 0; j < m; ++j)
      total[j] = 0.0;
    tf::buffer<WidePt> queries;
    queries.allocate(m);
    for (std::size_t j = 0; j < m; ++j)
      for (std::size_t d = 0; d < 3; ++d)
        queries[j][d] = T1(corners[j][0][d]) + T1(corners[j][1][d]) +
                        T1(corners[j][2][d]);
    // the queries are corner sums, so the sheet is read at three times
    // the lattice: solid angle does not see the scale
    auto exact_point = [&](auto id) -> WidePt {
      const auto p = reader.point_in(form.points(), frame, base, Index(id));
      return WidePt{T1(p[0]) * T1(3), T1(p[1]) * T1(3), T1(p[2]) * T1(3)};
    };

    struct local_t {
      tf::buffer<double> acc;
    };

    auto task = [&](auto &&faces, local_t &local) {
      if (local.acc.size() != m) {
        local.acc.allocate(m);
        for (std::size_t j = 0; j < m; ++j)
          local.acc[j] = 0.0;
      }
      for (auto &&face : faces) {
        const auto p0 = exact_point(face[0]);
        const auto p1 = exact_point(face[1]);
        const auto p2 = exact_point(face[2]);
        for (std::size_t j = 0; j < m; ++j) {
          const auto &q = queries[j];
          // Differences widened to T1 (full-range Int would overflow),
          // exact, then double.
          const double ax = double(T1(p0[0]) - T1(q[0])),
                       ay = double(T1(p0[1]) - T1(q[1])),
                       az = double(T1(p0[2]) - T1(q[2]));
          const double bx = double(T1(p1[0]) - T1(q[0])),
                       by = double(T1(p1[1]) - T1(q[1])),
                       bz = double(T1(p1[2]) - T1(q[2]));
          const double cx = double(T1(p2[0]) - T1(q[0])),
                       cy = double(T1(p2[1]) - T1(q[1])),
                       cz = double(T1(p2[2]) - T1(q[2]));
          const double la = tf::sqrt(ax * ax + ay * ay + az * az);
          const double lb = tf::sqrt(bx * bx + by * by + bz * bz);
          const double lc = tf::sqrt(cx * cx + cy * cy + cz * cz);
          const double det = ax * (by * cz - bz * cy) -
                             ay * (bx * cz - bz * cx) +
                             az * (bx * cy - by * cx);
          const double den = la * lb * lc +
                             (ax * bx + ay * by + az * bz) * lc +
                             (bx * cx + by * cy + bz * cz) * la +
                             (cx * ax + cy * ay + cz * az) * lb;
          local.acc[j] += 2.0 * std::atan2(det, den);
        }
      }
    };

    auto agg = [&](const local_t &local, const tf::none_t &) {
      if (local.acc.size() != m)
        return;
      for (std::size_t j = 0; j < m; ++j)
        total[j] += local.acc[j];
    };

    tf::blocked_reduce(form.faces(), tf::none, local_t{}, task, agg);
  };
  // the tree's boxes bound the input, and a moved vertex moved within the
  // band, so the moments fold over the sheet where it is placed, taken back
  // into the form's own space
  auto folded = [&](const auto &sheet) {
    using real_t = tf::coordinate_type<decltype(sheet.points())>;
    const auto inverse = frame.inverse_transformation();
    auto placed = tf::make_mapped_range(
        tf::make_sequence_range(Index(sheet.points().size())),
        [&](Index id) -> tf::point<real_t, 3> {
          const auto row = reader.moved_row(base + id);
          if (row == Index(-1))
            return tf::point<real_t, 3>(sheet.points()[id]);
          return tf::point<real_t, 3>(tf::transformed(
              reader.converter.deconvert(reader.moved[std::size_t(row)]),
              inverse));
        });
    auto polygons = tf::make_polygons(sheet.faces(), placed);
    auto moments = tf::make_winding_moments(sheet.tree(), polygons);
    tf::points_buffer<double, 3> points;
    points.allocate(m);
    tf::parallel_for_each(
        tf::make_sequence_range(m),
        [&](std::size_t j) {
          tf::point<double, 3> c{0.0, 0.0, 0.0};
          for (const auto &corner : corners[j]) {
            const auto p = reader.converter.deconvert(corner);
            for (std::size_t d = 0; d < 3; ++d)
              c[d] += double(p[d]) / 3.0;
          }
          points[j] = tf::transformed(c, inverse);
        },
        tf::checked);
    tf::winding_number(polygons | tf::tag(sheet.tree()) | tf::tag(moments),
                       points.points(), total);
    for (std::size_t j = 0; j < m; ++j)
      total[j] *= orientation * 4.0 * tf::pi<double>;
  };
  if constexpr (static_tree) {
    if (m < winding_moments_from || !similar)
      linear();
    else
      folded(form);
  } else {
    (void)similar;
    linear();
  }

  // A sign test with a dead zone just above the accumulation noise floor,
  // so the membrane sits at w = 0: a genuine back side is at least the
  // sheet's subtended angle (a bounded sheet seen from afar or near its
  // rim can be far below 2*pi, so no fixed fraction of the winding works),
  // while a closed sheet's exterior totals exactly 0 plus float noise and
  // reads as the front.
  constexpr double noise_floor = 1e-6;
  for (std::size_t j = 0; j < m; ++j)
    side[j] = total[j] > noise_floor ? char(1) : char(0);
  return side;
}

} // namespace tf::csg::graph
