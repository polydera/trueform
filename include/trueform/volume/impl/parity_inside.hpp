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
#include "../../core/algorithm/generic_generate.hpp"
#include "../../core/algorithm/parallel_fill.hpp"
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../core/none.hpp"
#include "../../core/point.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../exact/orient2d.hpp"
#include "../../exact/orient3d.hpp"
#include "../../exact/vertex.hpp"
#include "./grid_lines.hpp"
#include "tbb/parallel_sort.h"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace tf {
namespace volume_detail {

/// @brief The banded magnitude's discovery, stated all-or-nothing: asking
/// for it fills every member — the crossing-adjacent `seeds`, the `shell`
/// of end samples whose nearest crossing lies past the grid, and per axis
/// the rows an IN-GRID crossing informed.
struct band_discovery {
  tf::buffer<char> seeds;
  tf::buffer<char> shell;
  std::array<tf::buffer<char>, 3> rows_crossed;
};

/// THE LINE MECHANISM. One grid line of `axis` is one infinite lattice line:
/// a face crosses it where the line's two ortho coordinates lie inside the
/// face's projection under SoS; the crossing's side flips exactly once along
/// the line, one binary search seats it among the samples, and the
/// containment's own orientation is the crossing's direction through the
/// face's winding. Everything below is a reader of these facts: axis 0
/// accumulates them into the winding parity, any axis marks the
/// crossing-adjacent samples as seeds, and a crossing seated past either end
/// marks that end's sample as boundary shell.
///
/// `samples` are the per-axis lattice coordinates in ASCENDING order;
/// `flips[d]` says axis `d`'s grid runs opposite to that order, and every
/// write maps back through it.
template <typename Int, typename Faces, typename Inside = tf::none_t,
          typename Discovery = tf::none_t>
void axis_crossings(const Faces &faces,
                    const tf::buffer<tf::point<Int, 3>> &points,
                    const std::array<const tf::buffer<Int> *, 3> &samples,
                    const std::array<bool, 3> &flips, int axis,
                    Inside inside = {}, Discovery discovery = {}) {
  using vertex_t = tf::exact::vertex<std::int64_t, Int>;
  const int o0 = (axis + 1) % 3, o1 = (axis + 2) % 3;
  const auto na = std::ptrdiff_t(samples[std::size_t(axis)]->size());
  const auto n0 = std::ptrdiff_t(samples[std::size_t(o0)]->size());
  const auto n1 = std::ptrdiff_t(samples[std::size_t(o1)]->size());
  if (na == 0 || n0 * n1 == 0 || faces.size() == 0)
    return;
  const auto &sa = *samples[std::size_t(axis)];
  const auto &s0 = *samples[std::size_t(o0)];
  const auto &s1 = *samples[std::size_t(o1)];
  const std::array<std::ptrdiff_t, 3> dims{
      std::ptrdiff_t(samples[0]->size()), std::ptrdiff_t(samples[1]->size()),
      std::ptrdiff_t(samples[2]->size())};
  const auto stride = grid_strides(dims);

  tf::buffer<std::array<std::int64_t, 2>> records;
  // a face's row span is unbounded relative to the face count, so the
  // binning is unconditionally parallel
  tf::generic_generate(
      tf::make_sequence_range(std::ptrdiff_t(faces.size())), records,
      [&](std::ptrdiff_t f, tf::buffer<std::array<std::int64_t, 2>> &out) {
        auto face = faces[f];
        const auto n = face.size();
        if (n < 3)
          return;
        Int lo0 = points[std::size_t(face[0])][o0], hi0 = lo0;
        Int lo1 = points[std::size_t(face[0])][o1], hi1 = lo1;
        for (std::size_t k = 1; k < n; ++k) {
          const auto &p = points[std::size_t(face[k])];
          lo0 = std::min(lo0, p[o0]);
          hi0 = std::max(hi0, p[o0]);
          lo1 = std::min(lo1, p[o1]);
          hi1 = std::max(hi1, p[o1]);
        }
        const auto b0 = std::lower_bound(s0.begin(), s0.end(), lo0);
        const auto e0 = std::upper_bound(s0.begin(), s0.end(), hi0);
        const auto b1 = std::lower_bound(s1.begin(), s1.end(), lo1);
        const auto e1 = std::upper_bound(s1.begin(), s1.end(), hi1);
        for (auto i1 = b1; i1 != e1; ++i1)
          for (auto i0 = b0; i0 != e0; ++i0)
            out.push_back(
                {(i0 - s0.begin()) + n0 * (i1 - s1.begin()), std::int64_t(f)});
      });
  if (records.size() == 0)
    return;
  tbb::parallel_sort(records.begin(), records.end());

  tf::buffer<std::ptrdiff_t> offsets;
  tf::buffer<std::ptrdiff_t> live_rows;
  offsets.allocate(std::size_t(n0 * n1) + 1);
  {
    std::ptrdiff_t r = 0;
    offsets[0] = 0;
    for (std::ptrdiff_t i = 0; i < std::ptrdiff_t(records.size()); ++i)
      while (r < records[std::size_t(i)][0])
        offsets[std::size_t(++r)] = i;
    while (r < n0 * n1)
      offsets[std::size_t(++r)] = std::ptrdiff_t(records.size());
    for (std::ptrdiff_t row = 0; row < n0 * n1; ++row)
      if (offsets[std::size_t(row)] != offsets[std::size_t(row) + 1])
        live_rows.push_back(row);
  }

  const auto qid = std::int64_t(points.size());
  const Int far_a = -std::numeric_limits<Int>::max();

  struct local_t {
    tf::buffer<std::int16_t> winding;
  };
  tf::parallel_for_each(
      tf::make_range(live_rows),
      [&, axis, o0, o1](std::ptrdiff_t row, local_t &local) {
        const auto begin = offsets[std::size_t(row)];
        const auto end = offsets[std::size_t(row) + 1];
        const auto i0 = row % n0;
        const auto i1 = row / n0;
        if constexpr (!std::is_same_v<Inside, tf::none_t>) {
          local.winding.allocate(std::size_t(na));
          std::fill(local.winding.begin(), local.winding.end(),
                    std::int16_t(0));
        }
        const auto g0 = flips[std::size_t(o0)] ? n0 - 1 - i0 : i0;
        const auto g1 = flips[std::size_t(o1)] ? n1 - 1 - i1 : i1;
        const auto line_base =
            g0 * stride[std::size_t(o0)] + g1 * stride[std::size_t(o1)];
        const auto line_step = stride[std::size_t(axis)];
        const bool flip_a = flips[std::size_t(axis)];
        auto grid_at = [&](std::ptrdiff_t s) {
          return line_base + (flip_a ? na - 1 - s : s) * line_step;
        };
        vertex_t q{qid, {}};
        q.pt[o0] = s0[std::size_t(i0)];
        q.pt[o1] = s1[std::size_t(i1)];
        vertex_t far_end = q;
        far_end.pt[axis] = far_a;
        for (auto i = begin; i < end; ++i) {
          auto face = faces[records[std::size_t(i)][1]];
          const auto n = face.size();
          const vertex_t v0{std::int64_t(face[0]),
                            points[std::size_t(face[0])]};
          for (std::size_t k = 1; k + 1 < n; ++k) {
            const vertex_t va{std::int64_t(face[k]),
                              points[std::size_t(face[k])]};
            const vertex_t vb{std::int64_t(face[k + 1]),
                              points[std::size_t(face[k + 1])]};
            const bool s01 = tf::exact::orient2d_sos(v0, va, q, o0, o1);
            const bool s12 = tf::exact::orient2d_sos(va, vb, q, o0, o1);
            if (s01 != s12)
              continue;
            if (tf::exact::orient2d_sos(vb, v0, q, o0, o1) != s12)
              continue;
            const bool side_minus = tf::exact::orient3d_sos(
                std::array<vertex_t, 4>{v0, va, vb, far_end});
            auto side_at = [&](std::ptrdiff_t s) {
              q.pt[axis] = sa[std::size_t(s)];
              return tf::exact::orient3d_sos(
                  std::array<vertex_t, 4>{v0, va, vb, q});
            };
            const bool at0 = side_at(0) != side_minus;
            std::ptrdiff_t lo = 0, hi = na;
            while (lo + 1 < hi) {
              const auto mid = lo + (hi - lo) / 2;
              if ((side_at(mid) != side_minus) == at0)
                lo = mid;
              else
                hi = mid;
            }
            if constexpr (!std::is_same_v<Inside, tf::none_t>) {
              const std::int16_t dir = s01 ? std::int16_t(-1)
                                           : std::int16_t(1);
              if (at0)
                local.winding[0] = std::int16_t(local.winding[0] + dir);
              else if (hi < na)
                local.winding[std::size_t(hi)] =
                    std::int16_t(local.winding[std::size_t(hi)] + dir);
            }
            if constexpr (!std::is_same_v<Discovery, tf::none_t>) {
              if (!at0 && hi < na) {
                discovery->seeds[std::size_t(grid_at(hi))] = 1;
                if (hi > 0)
                  discovery->seeds[std::size_t(grid_at(hi - 1))] = 1;
                // only an in-grid crossing informs the row: a sample whose
                // every statement lies past the grid is blind, and the
                // measured set answers it
                discovery->rows_crossed[std::size_t(axis)][std::size_t(row)] =
                    1;
              } else if (at0) {
                discovery->shell[std::size_t(grid_at(0))] = 1;
              } else {
                discovery->shell[std::size_t(grid_at(na - 1))] = 1;
              }
            }
          }
        }
        if constexpr (!std::is_same_v<Inside, tf::none_t>) {
          std::int16_t w = 0;
          for (std::ptrdiff_t s = 0; s < na; ++s) {
            w = std::int16_t(w + local.winding[std::size_t(s)]);
            (*inside)[std::size_t(grid_at(s))] = char(w > 0);
          }
        }
      },
      local_t{});
}

/// @brief The inside mask of a closed mesh on a sample grid — winding
/// parity of the exact axis-0 crossings accumulated in an int16 (a line
/// would need 32767 same-direction crossings to wrap), negative inside by
/// winding — and, on request, the @ref band_discovery of all three axis
/// sweeps.
template <typename Int, typename Faces, typename Discovery = tf::none_t>
void parity_inside(const Faces &faces,
                   const tf::buffer<tf::point<Int, 3>> &points,
                   const std::array<const tf::buffer<Int> *, 3> &samples,
                   const std::array<bool, 3> &flips, tf::buffer<char> &inside,
                   Discovery discovery = {}) {
  constexpr bool with_discovery = !std::is_same_v<Discovery, tf::none_t>;
  const auto n_vox =
      samples[0]->size() * samples[1]->size() * samples[2]->size();
  inside.allocate(n_vox);
  tf::parallel_fill(inside, char(0));
  if constexpr (with_discovery) {
    discovery->seeds.allocate(n_vox);
    tf::parallel_fill(discovery->seeds, char(0));
    discovery->shell.allocate(n_vox);
    tf::parallel_fill(discovery->shell, char(0));
    for (int axis = 0; axis < 3; ++axis) {
      const int o0 = (axis + 1) % 3, o1 = (axis + 2) % 3;
      auto &rows = discovery->rows_crossed[std::size_t(axis)];
      rows.allocate(samples[std::size_t(o0)]->size() *
                    samples[std::size_t(o1)]->size());
      tf::parallel_fill(rows, char(0));
    }
    axis_crossings<Int>(faces, points, samples, flips, 0, &inside, discovery);
    axis_crossings<Int>(faces, points, samples, flips, 1, tf::none, discovery);
    axis_crossings<Int>(faces, points, samples, flips, 2, tf::none, discovery);
  } else {
    axis_crossings<Int>(faces, points, samples, flips, 0, &inside);
  }
}

} // namespace volume_detail
} // namespace tf
