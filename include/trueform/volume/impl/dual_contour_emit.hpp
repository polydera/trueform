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
#include "../../core/grain.hpp"
#include "../../core/views/sequence_range.hpp"
#include "./dual_contour_cells.hpp"
#include "./dual_contour_tables.hpp"
#include "./isosurface_classify.hpp"

#include <cstddef>

namespace tf {
namespace volume_detail {

/// @brief Emit the surface as triangles, subdividing every quad side whose face
/// was marked.
///
/// One polygon per owned interior crossing edge: four corners, with the arc
/// vertex of any marked side inserted between the two it joins. An ordinary
/// quad splits along its shorter 3D diagonal; a subdivided one fans around a
/// vertex private to it, so no diagonal it introduces can collide with another
/// polygon's. The winding is the one flying edges and marching cubes state.
///
/// @tparam RecordPolygons Also write the completed polygon boundaries, which is
///         the intermediate surface the triangles tessellate. A build that did
///         not ask instantiates without them and carries no test for them.
/// @param poly_offsets,poly_indices Read only when @p RecordPolygons; the
///         offsets are per quad, one block each, in the row order the
///         triangles follow.
template <bool RecordPolygons = false, typename Index, typename Points,
          typename Real, typename Samples>
auto emit_triangles(const dual_contour_cells<Index> &cells,
                    const Samples &samples, Real iso,
                    const unsigned char *split_mask, const Index *split_base,
                    Index f_origin, Index e_origin, Points points, Index *faces,
                    Index *poly_offsets = nullptr,
                    Index *poly_indices = nullptr) -> void {
  static_cast<void>(poly_offsets);
  static_cast<void>(poly_indices);
  const int cy = cells.g.cy, cz = cells.g.cz;
  const auto cell_rows = static_cast<std::ptrdiff_t>(cy) * cz;
  tf::parallel_for_each(
      tf::make_sequence_range(cell_rows),
      [&](std::ptrdiff_t row) {
        const int z = int(row / cy), y = int(row % cy);
        // A quad reaches only the four cell rows (y-1,z-1), (y,z-1),
        // (y-1,z) and (y,z), and within a row only cells x-1 then x — a
        // monotone query sequence. Each row therefore carries a cursor into
        // its own block of active cells, advanced only where a quad is
        // emitted, so a cell that bounds nothing costs nothing.
        struct row_reader {
          std::ptrdiff_t cursor, end;
          std::ptrdiff_t entry[2];
          Index base[2];
          unsigned retained[2];
          unsigned char cs[2];
          bool have[2];
        };
        row_reader r[4] = {};
        const bool live[4] = {y >= 1 && z >= 1, z >= 1, y >= 1, true};
        const int read_y[4] = {y - 1, y, y - 1, y};
        const int read_z[4] = {z - 1, z - 1, z, z};
        for (int k = 0; k < 4; ++k) {
          if (!live[k])
            continue;
          const auto nrow = cells.g.cell_row(read_y[k], read_z[k]);
          r[k].cursor = cells.active_offsets[nrow];
          r[k].end = cells.active_offsets[nrow + 1];
        }
        // slot 0 is the cell at x-1, slot 1 the cell at x; priming in that
        // order is what keeps every cursor monotone
        auto prime = [&](int k, int slot, int at_x) {
          if (r[k].have[slot])
            return;
          const int xi = at_x - 1 + slot;
          while (r[k].cursor < r[k].end && cells.active_x[r[k].cursor] < xi)
            ++r[k].cursor;
          r[k].entry[slot] = r[k].cursor;
          r[k].base[slot] = cells.active_base[r[k].cursor];
          r[k].cs[slot] = cells.active_case[r[k].cursor];
          r[k].retained[slot] = cells.retained_at(xi, read_y[k], read_z[k]);
          r[k].have[slot] = true;
        };
        auto vertex_at = [&](int k, int slot, int fe_edge) {
          return r[k].base[slot] +
                 Index(r[k].retained[slot] == k_all_edges
                           ? components()
                                 .comp[r[k].cs[slot]][std::size_t(fe_edge)]
                           : cell_vertex_index(r[k].cs[slot],
                                               r[k].retained[slot], fe_edge));
        };

        Index quad_at = cells.row_meta[row * 8 + 1];
        Index poly_at = cells.row_meta[row * 8 + 7];
        Index tri_at = cells.row_meta[row * 8 + 3];
        Index fan_at = cells.row_meta[row * 8 + 6];
        auto tri = [&](Index a, Index b, Index c) {
          auto *t = faces + std::ptrdiff_t(tri_at++) * 3;
          t[0] = a;
          t[1] = b;
          t[2] = c;
        };
        auto dist2 = [&](Index a, Index b) {
          double d = 0;
          for (int i = 0; i < 3; ++i) {
            const double s = double(points[a][i]) - double(points[b][i]);
            d += s * s;
          }
          return d;
        };
        auto emit = [&](int axis, int gx, bool flip) {
          Index bnd[8];
          int n = 0;
          for (int side = 0; side < 4; ++side) {
            const auto &corner =
                k_quad_corners[std::size_t(axis)][std::size_t(side)];
            bnd[n++] = vertex_at(corner.row, corner.slot, corner.edge);
            const auto &sd = k_quad_sides[std::size_t(axis)][std::size_t(side)];
            const auto &oc =
                k_quad_corners[std::size_t(axis)][std::size_t(sd.owner_corner)];
            const auto entry = r[oc.row].entry[oc.slot];
            const unsigned mask = split_mask[entry];
            if (!((mask >> sd.face_axis) & 1u))
              continue;
            const int arc = arc_of_ordinal(r[oc.row].cs[oc.slot],
                                           sd.face_axis * 2 + 1, sd.ordinal);
            bnd[n++] =
                f_origin + split_base[entry] +
                Index(2 * fe_popcount(mask & ((1u << sd.face_axis) - 1u))) +
                Index(arc);
          }
          if (flip)
            for (int a = 1, b = n - 1; a < b; ++a, --b) {
              const Index t = bnd[a];
              bnd[a] = bnd[b];
              bnd[b] = t;
            }
          if constexpr (RecordPolygons) {
            poly_offsets[quad_at] = poly_at;
            for (int i = 0; i < n; ++i)
              poly_indices[poly_at + i] = bnd[i];
            poly_at += n;
          }
          ++quad_at;
          if (n == 4) {
            // the shorter 3D diagonal after placement; a tie takes (v0, v2)
            if (dist2(bnd[0], bnd[2]) <= dist2(bnd[1], bnd[3])) {
              tri(bnd[0], bnd[1], bnd[2]);
              tri(bnd[0], bnd[2], bnd[3]);
            } else {
              tri(bnd[0], bnd[1], bnd[3]);
              tri(bnd[1], bnd[2], bnd[3]);
            }
            return;
          }
          // a subdivided quad fans around a vertex private to it, so no
          // diagonal it introduces can collide with another polygon's
          const Index centre = e_origin + fan_at++;
          const auto p = primal_crossing<double>(cells.g, samples, double(iso),
                                                 gx, y, z, axis);
          for (int d = 0; d < 3; ++d)
            points[centre][d] = Real(p[std::size_t(d)]);
          for (int i = 0; i < n; ++i)
            tri(centre, bnd[i], bnd[(i + 1) % n]);
        };

        // Each of the three conditions below states that two corners of this
        // cell differ, so a cell that carries a quad is a cell that carries
        // surface — and those the row already holds, compact and in x order,
        // with the composite each condition reads. The dense span between two
        // walls of a solid is not walked at all.
        const auto active_end = cells.row_end(row);
        for (auto e = cells.row_begin(row); e < active_end; ++e) {
          const int x = cells.active_x[e];
          const auto composite = cells.active_case[e];
          const auto cases = static_cast<unsigned char>(composite & 3u);
          const auto c0 = static_cast<unsigned char>(composite & 1u);
          const auto c2 = static_cast<unsigned char>((composite >> 2) & 1u);
          const auto c4 = static_cast<unsigned char>((composite >> 4) & 1u);
          const bool xq = y >= 1 && z >= 1 &&
                          (cases == k_left_inside || cases == k_right_inside);
          const bool yq = x >= 1 && z >= 1 && c0 != c2;
          const bool zq = x >= 1 && y >= 1 && c0 != c4;
          if (!(xq || yq || zq))
            continue;
          for (auto &rr : r)
            rr.have[0] = rr.have[1] = false;
          if (yq) {
            prime(1, 0, x);
            prime(3, 0, x);
          }
          if (zq) {
            prime(2, 0, x);
            prime(3, 0, x);
          }
          if (xq) {
            prime(0, 1, x);
            prime(1, 1, x);
            prime(2, 1, x);
            prime(3, 1, x);
            emit(0, x, cases == k_right_inside);
          }
          if (yq) {
            prime(1, 1, x);
            prime(3, 1, x);
            emit(1, x, c2 != 0u);
          }
          if (zq) {
            prime(2, 1, x);
            prime(3, 1, x);
            emit(2, x, c4 != 0u);
          }
        }
      },
      tf::grain(k_dc_row_grain));
}

} // namespace volume_detail
} // namespace tf
