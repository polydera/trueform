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
#include "../../core/algorithm/parallel_fill.hpp"
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../core/polygons_buffer.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../isosurface_config.hpp"
#include "../volume.hpp"
#include "./dual_contour_cells.hpp"
#include "./dual_contour_emit.hpp"
#include "./dual_contour_faces.hpp"
#include "./dual_contour_passes.hpp"
#include "./dual_contour_refine.hpp"
#include "./dual_contour_rows.hpp"
#include "./dual_contour_vertices.hpp"
#include "./field_value.hpp"
#include "./isosurface_classify.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

namespace tf {
namespace volume_detail {

// Dual contouring on the flying-edges chassis: one vertex per surface
// component of a cell, one polygon per crossing grid edge, sharp features
// recovered from the crossings themselves.
//
// The surface a dual quad set names is not yet an indexed mesh: two contour
// arcs on one shared grid face join the same pair of components, and their two
// dual edges would carry one endpoint pair. `dual_contour_tables` gives the arc
// its own identity, `discover_marked_faces` finds where it matters, and the
// quads that read it become 5-8-gons fanned around a private centre. The result
// is manifold by construction, not by repair.
//
// A corner is inside when `sample < iso`, matching `flying_edges` and
// `marching_cubes`; the winding is theirs too.

/// @brief A quad one of whose sides a marked face subdivides.
struct touched_quad {
  std::ptrdiff_t row;
  int key;
};

/// @brief The dual-contouring extractor: a builder that owns the scratch its
/// passes read and write, and wires them together.
template <typename Index, typename Real> class dual_contouring {
public:
  /// @brief Also retain the completed polygon surface — quads and 5-8-gons —
  /// that the emitted triangles tessellate.
  ///
  /// Asked before the build, like any other request the consumer makes of it. A
  /// build that did not ask allocates nothing and states nothing about
  /// polygons; the topology tests are what ask.
  auto record_polygons(bool on) -> void { _record_polygons = on; }

  /// @brief Also retain what the refinement did with every patch vertex.
  ///
  /// Asked before the build, like @ref record_polygons; a build that did not
  /// ask allocates nothing and the refinement writes nothing. It is what lets a
  /// consumer attribute a vertex's placement to the pass that made it.
  auto record_refit_provenance(bool on) -> void { _record_provenance = on; }

  /// @brief Also retain what each pass of the build cost, in milliseconds.
  ///
  /// Asked before the build, like @ref record_polygons; a build that did not
  /// ask reads no clock. The table is what attributes a total to the passes
  /// that spent it, so a claim about one of them is priced against the rest.
  auto record_pass_times(bool on) -> void { _record_times = on; }

  /// @brief The completed polygons of the last build: one block per dual quad,
  /// empty unless @ref record_polygons was asked.
  auto polygon_offsets() const -> const tf::buffer<Index> & {
    return _poly_offsets;
  }
  auto polygon_indices() const -> const tf::buffer<Index> & {
    return _poly_indices;
  }

  /// @brief The refinement's account of the last build: one record per patch
  /// vertex, empty unless @ref record_refit_provenance was asked.
  auto refit_provenance() const -> const tf::buffer<refit_record> & {
    return _provenance;
  }

  /// @brief What each pass of the last build cost, indexed by @ref dc_pass.
  /// Zero throughout unless @ref record_pass_times was asked.
  auto pass_times() const -> const std::array<double, k_dc_pass_count> & {
    return _pass_times;
  }

  /// @brief Extract the isosurface of @p vol at @p iso as a triangle mesh.
  template <typename Policy>
  auto build(const tf::volume<Policy> &vol, Real iso,
             const tf::isosurface_config &config)
      -> tf::polygons_buffer<Index, Real, 3, 3> {
    dc_grid g;
    g.nx = vol.dims()[0];
    g.ny = vol.dims()[1];
    g.nz = vol.dims()[2];
    g.cx = g.nx - 1;
    g.cy = g.ny - 1;
    g.cz = g.nz - 1;
    for (int i = 0; i < 3; ++i) {
      g.origin[i] = double(vol.origin()[i]);
      g.spacing[i] = double(vol.spacing()[i]);
    }

    tf::polygons_buffer<Index, Real, 3, 3> out;
    if (g.nx < 2 || g.ny < 2 || g.nz < 2)
      return out;

    const auto samples = make_field_samples<Real>(vol);
    const auto cell_rows = static_cast<std::ptrdiff_t>(g.cy) * g.cz;
    const auto rows = static_cast<std::ptrdiff_t>(g.ny) * g.nz;
    dc_pass_clock clock(_record_times, _pass_times.data());

    _x_cases.allocate(static_cast<std::ptrdiff_t>(g.nx - 1) * rows);
    _x_meta.allocate(rows * 4);
    _x_min.allocate(rows);
    _x_max.allocate(rows);
    classify_x_edges(vol, iso, _x_cases.data(), _x_meta.data(), 4,
                     _x_min.data(), _x_max.data());
    clock.mark(dc_pass::classify);

    _row_meta.allocate(cell_rows * 8);
    _active_offsets.allocate(cell_rows + 1);
    _active_offsets[0] = 0;
    count_cell_rows(g, _x_cases.data(), _x_meta.data(), std::size_t(4),
                    _x_min.data(), _x_max.data(), _row_meta.data(),
                    _active_offsets.data());
    clock.mark(dc_pass::count_rows);

    // the row prefix: vertex and quad bases, and each row's block of active
    // records (microseconds at these row counts)
    std::int64_t vertex_count = 0, quad_count = 0;
    for (std::ptrdiff_t row = 0; row < cell_rows; ++row) {
      auto *meta = &_row_meta[row * 8];
      const auto cells_here = meta[0];
      const auto quads_here = meta[1];
      meta[0] = vertex_count;
      meta[1] = quad_count;
      vertex_count += cells_here;
      quad_count += quads_here;
      _active_offsets[row + 1] += _active_offsets[row];
    }
    const auto active_count = _active_offsets[cell_rows];
    clock.mark(dc_pass::row_prefix);

    _active_x.allocate(active_count);
    _active_base.allocate(active_count);
    _active_case.allocate(active_count);
    fill_active_cells(g, _x_cases.data(), _row_meta.data(),
                      _active_offsets.data(), _active_x.data(),
                      _active_base.data(), _active_case.data());
    clock.mark(dc_pass::fill_active);

    dual_contour_cells<Index> cells;
    cells.g = g;
    cells.x_cases = _x_cases.data();
    cells.row_meta = _row_meta.data();
    cells.active_offsets = _active_offsets.data();
    cells.active_x = _active_x.data();
    cells.active_base = _active_base.data();
    cells.active_case = _active_case.data();

    _split_mask.allocate(active_count);
    _split_base.allocate(active_count);
    discover_marked_faces(cells, _split_mask.data(), _row_meta.data(), _marked);
    clock.mark(dc_pass::marked_faces);

    // F identities follow the patch identities; the prefix gives every cell a
    // private range, so no two faces can name the same vertex
    std::int64_t split_vertex_count = 0;
    for (std::ptrdiff_t row = 0; row < cell_rows; ++row) {
      auto *meta = &_row_meta[row * 8];
      const auto n = meta[2];
      meta[2] = split_vertex_count;
      split_vertex_count += n;
    }
    const Index marked_faces = split_vertex_count / 2;
    if (marked_faces > 0)
      tf::parallel_for_each(
          tf::make_sequence_range(cell_rows),
          [&](std::ptrdiff_t row) {
            Index base = _row_meta[row * 8 + 2];
            const auto end = _active_offsets[row + 1];
            for (auto e = _active_offsets[row]; e < end; ++e) {
              _split_base[e] = base;
              base += Index(2 * fe_popcount(_split_mask[e]));
            }
          },
          tf::grain(k_dc_row_grain));

    // Which quads a marked face touches: at most four per face, and the
    // marked set is sparse, so this is grouped directly instead of
    // walking every quad a second time.
    _touched.clear();
    for (std::ptrdiff_t i = 0; i < std::ptrdiff_t(_marked.size()); ++i) {
      const auto &m = _marked[i];
      const int high = m.axis * 2 + 1;
      const unsigned retained = retained_edges(m.x, m.y, m.z, g.cx, g.cy, g.cz);
      for (int o = 0; o < 4; ++o) {
        const int el = k_face_edges[std::size_t(high)][std::size_t(o)];
        if (!((retained >> el) & 1u))
          continue;
        const auto &c0 =
            k_corner_offset[std::size_t(k_cell_edge[std::size_t(el)][0])];
        const auto &c1 =
            k_corner_offset[std::size_t(k_cell_edge[std::size_t(el)][1])];
        int eaxis = 0;
        for (int d = 0; d < 3; ++d)
          if (c0[std::size_t(d)] != c1[std::size_t(d)])
            eaxis = d;
        const int gx = m.x + c0[0], gy = m.y + c0[1], gz = m.z + c0[2];
        _touched.push_back(touched_quad{
            (static_cast<std::ptrdiff_t>(gz) * g.cy + gy), gx * 3 + eaxis});
      }
    }
    std::sort(_touched.begin(), _touched.end(),
              [](const touched_quad &a, const touched_quad &b) {
                return a.row != b.row ? a.row < b.row : a.key < b.key;
              });
    for (std::ptrdiff_t i = 0; i < std::ptrdiff_t(_touched.size());) {
      std::ptrdiff_t j = i;
      while (j < std::ptrdiff_t(_touched.size()) &&
             _touched[j].row == _touched[i].row &&
             _touched[j].key == _touched[i].key)
        ++j;
      auto *meta = &_row_meta[_touched[i].row * 8];
      meta[3] += Index(j - i); // this row's total k
      meta[6] += 1;            // one more touched quad, one more fan centre
      i = j;
    }

    // Triangles: two per ordinary quad, 4 + k around a private fan centre for
    // a touched one. Polygon indices are 4 + k either way.
    std::int64_t triangle_count = 0, fan_count = 0, polygon_index_count = 0;
    for (std::ptrdiff_t row = 0; row < cell_rows; ++row) {
      auto *meta = &_row_meta[row * 8];
      const Index quads_here = (row + 1 < cell_rows)
                                   ? _row_meta[(row + 1) * 8 + 1] - meta[1]
                                   : quad_count - meta[1];
      const Index k_here = meta[3];
      const Index touched_here = meta[6];
      meta[3] = triangle_count;
      meta[6] = fan_count;
      meta[7] = polygon_index_count;
      triangle_count += 2 * quads_here + 2 * touched_here + k_here;
      fan_count += touched_here;
      polygon_index_count += 4 * quads_here + k_here;
    }
    const auto f_origin = Index(vertex_count);
    const auto e_origin = Index(f_origin + split_vertex_count);
    const auto total_vertices = Index(e_origin + fan_count);
    clock.mark(dc_pass::arc_prefix);

    out.points_buffer().allocate(total_vertices);
    out.faces_buffer().data_buffer().allocate(std::ptrdiff_t(triangle_count) *
                                              3);

    if (config.refine)
      _normal_spread.allocate(vertex_count);
    clock.mark(dc_pass::allocate);

    auto points = out.points_buffer().points();
    place_patch_vertices(cells, samples, iso, config.stabilizer, config.refine,
                         points, _normal_spread.data());
    clock.mark(dc_pass::patch_vertices);

    place_arc_vertices(cells, samples, iso, _marked, _split_mask.data(),
                       _split_base.data(), f_origin, points);
    clock.mark(dc_pass::arc_vertices);
    // the emit variant is chosen once, here: a build that did not ask for the
    // polygon surface runs the instantiation that has no notion of it
    if (_record_polygons) {
      _poly_offsets.allocate(std::ptrdiff_t(quad_count) + 1);
      _poly_indices.allocate(polygon_index_count);
      _poly_offsets[quad_count] = Index(polygon_index_count);
      emit_triangles<true>(cells, samples, iso, _split_mask.data(),
                           _split_base.data(), f_origin, e_origin, points,
                           out.faces_buffer().data_buffer().data(),
                           _poly_offsets.data(), _poly_indices.data());
    } else {
      emit_triangles(cells, samples, iso, _split_mask.data(),
                     _split_base.data(), f_origin, e_origin, points,
                     out.faces_buffer().data_buffer().data());
    }
    clock.mark(dc_pass::emit);

    if (config.refine) {
      if (_record_provenance) {
        _provenance.allocate(vertex_count);
        tf::parallel_fill(_provenance, refit_record{});
        refine_feature_vertices<true>(cells, samples, iso, config.stabilizer,
                                      Index(vertex_count),
                                      _normal_spread.data(), _flagged, points,
                                      _provenance.data());
      } else
        refine_feature_vertices(cells, samples, iso, config.stabilizer,
                                Index(vertex_count), _normal_spread.data(),
                                _flagged, points);
    }
    clock.mark(dc_pass::refine);
    return out;
  }

private:
  bool _record_polygons = false;
  bool _record_provenance = false;
  bool _record_times = false;
  std::array<double, k_dc_pass_count> _pass_times{};
  tf::buffer<Index> _poly_offsets;
  tf::buffer<Index> _poly_indices;
  tf::buffer<refit_record> _provenance;
  tf::buffer<unsigned char> _x_cases;
  tf::buffer<std::int64_t> _x_meta;
  tf::buffer<int> _x_min;
  tf::buffer<int> _x_max;
  tf::buffer<Index> _row_meta;
  tf::buffer<int> _active_x;
  tf::buffer<Index> _active_base;
  tf::buffer<unsigned char> _active_case;
  tf::buffer<std::ptrdiff_t> _active_offsets;
  tf::buffer<unsigned char> _split_mask;
  tf::buffer<Index> _split_base;
  tf::buffer<marked_face> _marked;
  tf::buffer<touched_quad> _touched;
  tf::buffer<float> _normal_spread;
  tf::buffer<std::ptrdiff_t> _flagged;
};

} // namespace volume_detail
} // namespace tf
