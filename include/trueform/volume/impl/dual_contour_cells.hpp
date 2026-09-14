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
#include "../../core/buffer.hpp"
#include "./dual_contour_tables.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

namespace tf {
namespace volume_detail {

/// @brief The extractors' shared grid view: dimensions, cell counts and the
/// maps from a voxel or a row to its place in the flat buffers.
struct dc_grid {
  int nx = 0, ny = 0, nz = 0;
  int cx = 0, cy = 0, cz = 0;
  std::array<double, 3> origin{};
  std::array<double, 3> spacing{};

  auto sample_index(int x, int y, int z) const -> std::ptrdiff_t {
    return (static_cast<std::ptrdiff_t>(z) * ny + y) * nx + x;
  }
  auto row(int y, int z) const -> std::ptrdiff_t {
    return static_cast<std::ptrdiff_t>(z) * ny + y;
  }
  auto x_case_row(int y, int z) const -> std::ptrdiff_t {
    return row(y, z) * (nx - 1);
  }
  auto cell_row(int y, int z) const -> std::ptrdiff_t {
    return static_cast<std::ptrdiff_t>(z) * cy + y;
  }
  /// @brief A cell with a full complement of dual quads around every crossing.
  auto interior(int x, int y, int z) const -> bool {
    return x >= 1 && x <= cx - 2 && y >= 1 && y <= cy - 2 && z >= 1 &&
           z <= cz - 2;
  }
};

/// @brief A cell's composite case: the four x-edge cases of its four rows,
/// which together state all eight corner signs.
inline auto cell_case(const unsigned char *x0, const unsigned char *x1,
                      const unsigned char *x2, const unsigned char *x3)
    -> unsigned char {
  return static_cast<unsigned char>(*x0 | (*x1 << 2) | (*x2 << 4) | (*x3 << 6));
}

constexpr unsigned char k_left_inside = 1u;
constexpr unsigned char k_right_inside = 2u;

// Each task takes at least this many cell rows, so a row's four x-case rows
// stay cache-warm across it. The carrier is the row, not the slice: measuring
// eight SLICES per task capped the parallel width at cz/8 and cost 12% at 256
// cubed on eight threads.
constexpr int k_dc_row_grain = 8;

/// @brief The compact record of the cells that carry surface: their x, their
/// case and the first output vertex each owns, in row-major blocks.
///
/// Every pass after the count reads the cell space through this. It is a view:
/// the builder owns the buffers.
template <typename Index> struct dual_contour_cells {
  dc_grid g;
  const unsigned char *x_cases = nullptr;
  const Index *row_meta = nullptr; // eight per cell row
  const std::ptrdiff_t *active_offsets = nullptr;
  const int *active_x = nullptr;
  const Index *active_base = nullptr;
  const unsigned char *active_case = nullptr;

  auto row_begin(std::ptrdiff_t row) const -> std::ptrdiff_t {
    return active_offsets[row];
  }
  auto row_end(std::ptrdiff_t row) const -> std::ptrdiff_t {
    return active_offsets[row + 1];
  }
  auto meta(std::ptrdiff_t row) const -> const Index * {
    return row_meta + row * 8;
  }
  auto case_at(int x, int y, int z) const -> unsigned char {
    return cell_case(&x_cases[g.x_case_row(y, z) + x],
                     &x_cases[g.x_case_row(y + 1, z) + x],
                     &x_cases[g.x_case_row(y, z + 1) + x],
                     &x_cases[g.x_case_row(y + 1, z + 1) + x]);
  }
  auto retained_at(int x, int y, int z) const -> unsigned {
    return g.interior(x, y, z) ? k_all_edges
                               : retained_edges(x, y, z, g.cx, g.cy, g.cz);
  }
  /// @brief The first vertex a cell owns, or -1. The active cells of a row are
  /// ascending in x, so this is a search in that row's block — the only random
  /// cell lookup in the build, and only the refinement's sparse neighbourhood
  /// asks for it.
  auto base_of(int x, int y, int z) const -> Index {
    auto lo = row_begin(g.cell_row(y, z));
    auto hi = row_end(g.cell_row(y, z));
    const auto end = hi;
    while (lo < hi) {
      const auto mid = lo + (hi - lo) / 2;
      if (active_x[mid] < x)
        lo = mid + 1;
      else
        hi = mid;
    }
    return (lo < end && active_x[lo] == x) ? active_base[lo] : Index(-1);
  }
};

template <typename Compute, typename Samples>
inline auto gradient_at(const dc_grid &g, const Samples &samples, int x, int y,
                        int z) -> std::array<Compute, 3> {
  auto sample = [&](int xi, int yi, int zi) {
    return static_cast<Compute>(samples[g.sample_index(xi, yi, zi)]);
  };
  std::array<Compute, 3> out;
  const int c[3] = {x, y, z};
  const int n[3] = {g.nx, g.ny, g.nz};
  for (int axis = 0; axis < 3; ++axis) {
    int lo[3] = {x, y, z};
    int hi[3] = {x, y, z};
    lo[axis] = std::max(c[axis] - 1, 0);
    hi[axis] = std::min(c[axis] + 1, n[axis] - 1);
    auto denom = static_cast<Compute>((hi[axis] - lo[axis])) * g.spacing[axis];
    out[axis] =
        (sample(hi[0], hi[1], hi[2]) - sample(lo[0], lo[1], lo[2])) / denom;
  }
  return out;
}

/// @brief The field's gradient at the sixty-four grid points a refinement
/// pool stands on, computed where it is first asked for and kept.
///
/// The twenty-seven cells of a refinement pool stand on sixty-four grid points,
/// and up to eight of those cells read the gradient at any one of them. The
/// block is the scope over which that fact is stated once and not again, and
/// sixty-four is what a single word of presence covers.
template <typename Compute, typename Samples> class pool_gradients {
public:
  pool_gradients(const dc_grid &g, const Samples &samples, int x, int y, int z)
      : _g(g), _samples(samples), _x(x), _y(y), _z(z) {}

  auto at(int gx, int gy, int gz) -> const std::array<Compute, 3> & {
    const auto i = std::size_t(((gz - _z) * 4 + (gy - _y)) * 4 + (gx - _x));
    if (!((_have >> i) & 1ull)) {
      _have |= 1ull << i;
      _value[i] = gradient_at<Compute>(_g, _samples, gx, gy, gz);
    }
    return _value[i];
  }

private:
  dc_grid _g;
  Samples _samples;
  int _x, _y, _z;
  std::uint64_t _have = 0;
  std::array<Compute, 3> _value[64];
};

/// @brief The field's gradient at the grid points of the four sample rows a
/// cell row stands on, kept across the run of rows one task walks.
///
/// Two of those four rows are the next cell row's as well, and within a row a
/// cell's four right-hand points are the next cell's four left-hand ones — so
/// a task that walks consecutive rows already holds most of what each cell
/// asks for. A sample row takes the slot its own parity names, which is what
/// makes both inheritances a comparison: a slot still holding its row keeps
/// every value under it, and one that has moved on raises its generation, which
/// retires them all at once without touching them.
template <typename Compute, typename Samples> class row_gradients {
public:
  row_gradients(const dc_grid &g, const Samples &samples)
      : _g(g), _samples(samples) {}

  /// @brief Take the four sample rows of cell row (@p y, @p z).
  auto open(int y, int z) -> void {
    for (int dz = 0; dz < 2; ++dz)
      for (int dy = 0; dy < 2; ++dy) {
        const int sy = y + dy, sz = z + dz;
        const auto key = static_cast<std::ptrdiff_t>(sz) * _g.ny + sy;
        const int slot = (sy & 1) | ((sz & 1) << 1);
        if (_held[slot] == key)
          continue;
        _held[slot] = key;
        ++_generation[slot];
      }
  }

  auto at(int gx, int gy, int gz) -> const std::array<Compute, 3> & {
    // a task whose rows carry no surface never asks, and never allocates
    if (_value.size() == 0) {
      _value.allocate(std::ptrdiff_t(4) * _g.nx);
      _stamp.allocate(std::ptrdiff_t(4) * _g.nx);
      for (auto &s : _stamp)
        s = 0;
    }
    const int slot = (gy & 1) | ((gz & 1) << 1);
    const auto i = static_cast<std::ptrdiff_t>(slot) * _g.nx + gx;
    if (_stamp[i] != _generation[slot]) {
      _stamp[i] = _generation[slot];
      _value[i] = gradient_at<Compute>(_g, _samples, gx, gy, gz);
    }
    return _value[i];
  }

private:
  dc_grid _g;
  Samples _samples;
  tf::buffer<std::array<Compute, 3>> _value;
  tf::buffer<int> _stamp;
  // a fresh slot's generation is raised past the zeroed stamps under it, so
  // nothing it holds is ever read before it is written
  std::array<int, 4> _generation{};
  std::array<std::ptrdiff_t, 4> _held{{-1, -1, -1, -1}};
};

/// @brief Where the field crosses a grid edge, stated exactly as
/// `cell_crossings` states it for a cell's own edges.
template <typename Compute, typename Samples>
inline auto primal_crossing(const dc_grid &g, const Samples &samples,
                            Compute iso, int gx, int gy, int gz, int axis)
    -> std::array<double, 3> {
  const int d[3] = {axis == 0, axis == 1, axis == 2};
  const auto s0 =
      static_cast<Compute>(samples[g.sample_index(gx, gy, gz)]) - iso;
  const auto s1 =
      static_cast<Compute>(
          samples[g.sample_index(gx + d[0], gy + d[1], gz + d[2])]) -
      iso;
  const auto t = s0 / (s0 - s1);
  const int c[3] = {gx, gy, gz};
  std::array<double, 3> p{};
  for (int i = 0; i < 3; ++i)
    p[std::size_t(i)] = g.origin[i] + (c[i] + t * d[i]) * g.spacing[i];
  return p;
}

} // namespace volume_detail
} // namespace tf
