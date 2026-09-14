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
#include "../../core/buffer.hpp"
#include "../../core/views/sequence_range.hpp"
#include "./dual_contour_cells.hpp"
#include "./dual_contour_tables.hpp"
#include "./isosurface_classify.hpp"

#include <cstddef>

namespace tf {
namespace volume_detail {

/// @brief A face whose two contour arcs join the same pair of patches, named by
/// the lower cell that owns it.
struct marked_face {
  std::ptrdiff_t entry;
  int x, y, z, axis;
};

/// @brief Find the faces whose two arcs would collapse into one dual edge.
///
/// Two neighbouring cells share exactly one face, and a checkerboard face
/// carries two contour arcs. When both arcs join the same pair of patches, the
/// two dual edges they should each own become one endpoint pair — four quads on
/// one indexed edge. That face is marked here, and only here: the lower cell
/// owns the decision, so all four quads around it read one answer.
///
/// @param split_mask One byte per active cell; bit `axis` is set when that
///        positive face is marked.
/// @param meta Entry [2] of each cell row receives the arc vertices that row's
///        marked faces will own.
template <typename Index>
auto discover_marked_faces(const dual_contour_cells<Index> &cells,
                           unsigned char *split_mask, Index *meta,
                           tf::buffer<marked_face> &marked) -> void {
  const int cx = cells.g.cx, cy = cells.g.cy, cz = cells.g.cz;
  const auto cell_rows = static_cast<std::ptrdiff_t>(cy) * cz;
  marked.clear();
  tf::generic_generate(
      tf::make_sequence_range(cell_rows), marked,
      [&](std::ptrdiff_t row, auto &out_marked) {
        const int z = int(row / cy), y = int(row % cy);
        const auto begin = cells.active_offsets[row];
        const auto end = cells.active_offsets[row + 1];
        std::ptrdiff_t cursor[3] = {begin, 0, 0};
        std::ptrdiff_t cursor_end[3] = {end, 0, 0};
        const bool has[3] = {true, y + 1 <= cy - 1, z + 1 <= cz - 1};
        if (has[1]) {
          cursor[1] = cells.active_offsets[row + 1];
          cursor_end[1] = cells.active_offsets[row + 2];
        }
        if (has[2]) {
          cursor[2] = cells.active_offsets[row + cy];
          cursor_end[2] = cells.active_offsets[row + cy + 1];
        }
        Index row_f = 0;
        for (auto e = begin; e < end; ++e) {
          const int x = cells.active_x[e];
          const auto cs = cells.active_case[e];
          const unsigned retained = cells.retained_at(x, y, z);
          unsigned char mask = 0;
          for (int axis = 0; axis < 3; ++axis) {
            const int high = axis * 2 + 1, low = axis * 2;
            if (patches().arc_count[cs][std::size_t(high)] != 2)
              continue;
            std::ptrdiff_t ue = -1;
            if (axis == 0) {
              if (x + 1 <= cx - 1 && e + 1 < end &&
                  cells.active_x[e + 1] == x + 1)
                ue = e + 1;
            } else if (has[axis]) {
              auto &c = cursor[axis];
              while (c < cursor_end[axis] && cells.active_x[c] < x)
                ++c;
              if (c < cursor_end[axis] && cells.active_x[c] == x)
                ue = c;
            }
            if (ue < 0)
              continue;
            const auto ucs = cells.active_case[ue];
            const unsigned uretained = cells.retained_at(
                x + (axis == 0), y + (axis == 1), z + (axis == 2));
            Index cl[2] = {0, 0}, cu[2] = {0, 0};
            bool usable = true;
            for (int i = 0; i < 2 && usable; ++i) {
              const auto pr =
                  patches().arc_pair[cs][std::size_t(high)][std::size_t(i)];
              const int o[2] = {pr >> 2, pr & 3};
              int pick = -1;
              for (int j = 0; j < 2 && pick < 0; ++j)
                if ((retained >>
                     k_face_edges[std::size_t(high)][std::size_t(o[j])]) &
                    1u)
                  pick = j;
              if (pick < 0) {
                usable = false; // no quad reads this arc
                break;
              }
              const int el =
                  k_face_edges[std::size_t(high)][std::size_t(o[pick])];
              const int eu =
                  k_face_edges[std::size_t(low)][std::size_t(o[pick])];
              cl[i] = cells.active_base[e] +
                      Index(cell_vertex_index(cs, retained, el));
              cu[i] = cells.active_base[ue] +
                      Index(cell_vertex_index(ucs, uretained, eu));
            }
            if (!usable || cl[0] != cl[1] || cu[0] != cu[1])
              continue;
            mask = static_cast<unsigned char>(mask | (1u << axis));
            out_marked.push_back(marked_face{e, x, y, z, axis});
          }
          split_mask[e] = mask;
          row_f += Index(2 * fe_popcount(mask));
        }
        meta[row * 8 + 2] = row_f;
      });
}

} // namespace volume_detail
} // namespace tf
