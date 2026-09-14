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
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/grain.hpp"
#include "../../core/point.hpp"
#include "../../core/points.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../remesh/collapse/quadric.hpp"
#include "./dual_contour_cells.hpp"
#include "./dual_contour_faces.hpp"
#include "./dual_contour_tables.hpp"
#include "./isosurface_classify.hpp"

#include <array>
#include <cmath>
#include <cstddef>

namespace tf {
namespace volume_detail {

// The Hermite crossings of one surface component of one cell: the crossing
// point on each of the component's edges and the field normal there. The sole
// producer of that fact — the vertex pass builds its quadric from it and the
// refinement gathers neighbourhoods of it.
//
// The gradients come from the caller's block, which is what keeps a grid point
// shared by several cells or several components from being differenced once
// for each of them.
template <typename Compute, typename Samples, typename Gradients>
inline auto cell_crossings(const dc_grid &g, const Samples &samples,
                           Compute iso, int x, int y, int z,
                           unsigned char composite, int comp,
                           Gradients &gradients, std::array<double, 3> *pos,
                           std::array<double, 3> *nrm, int cap) -> int {
  const auto &comp_of = components().comp[composite];
  auto corner_gradient = [&](int c) -> const std::array<Compute, 3> & {
    const auto &o = k_corner_offset[c];
    return gradients.at(x + o[0], y + o[1], z + o[2]);
  };
  auto corner_position = [&](int c) -> std::array<double, 3> {
    const auto &o = k_corner_offset[c];
    return {g.origin[0] + (x + o[0]) * g.spacing[0],
            g.origin[1] + (y + o[1]) * g.spacing[1],
            g.origin[2] + (z + o[2]) * g.spacing[2]};
  };
  auto sample = [&](int c) {
    const auto &o = k_corner_offset[c];
    return static_cast<Compute>(
               samples[g.sample_index(x + o[0], y + o[1], z + o[2])]) -
           iso;
  };

  int found = 0;
  for (int ei = 0; ei < 12 && found < cap; ++ei) {
    if (comp_of[ei] != comp)
      continue;
    const auto &e = k_cell_edge[ei];
    auto s0 = sample(e[0]);
    auto s1 = sample(e[1]);
    auto t = s0 / (s0 - s1);
    auto p0 = corner_position(e[0]);
    auto p1 = corner_position(e[1]);
    std::array<double, 3> p;
    for (int d = 0; d < 3; ++d)
      p[d] = p0[d] + t * (p1[d] - p0[d]);
    // the field's normal at the crossing: central differences at the edge's
    // two corners, lerped where the crossing falls
    const auto &g0 = corner_gradient(e[0]);
    const auto &g1 = corner_gradient(e[1]);
    std::array<double, 3> n{};
    for (int d = 0; d < 3; ++d)
      n[std::size_t(d)] =
          double(g0[std::size_t(d)] +
                 Compute(t) * (g1[std::size_t(d)] - g0[std::size_t(d)]));
    const double len = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
    // A vanishing gradient states no plane, but the crossing is still there:
    // dropping it would lose a topological fact to a numerical one. The zero
    // normal marks it as carrying no constraint.
    if (len < 1e-30)
      n = {0.0, 0.0, 0.0};
    else
      for (auto &v : n)
        v /= len;
    pos[found] = p;
    nrm[found] = n;
    ++found;
  }
  return found;
}

/// @brief Place one vertex per surface component of every active cell.
///
/// The vertex minimises the quadric its component's crossings state, pulled
/// toward their centroid by @p stabilizer. A component whose link the domain
/// cut owns one vertex per surviving interval, and every copy takes the same
/// placement; a component the boundary cut away entirely owns none, which is
/// what keeps these writes inside the cell's own range.
///
/// @param normal_spread When refining, receives per vertex the spread of its
///        component's normals, which is the feature gate's whole input.
template <typename Index, typename Points, typename Real, typename Samples>
auto place_patch_vertices(const dual_contour_cells<Index> &cells,
                          const Samples &samples, Real iso, double stabilizer,
                          bool refine, Points points, float *normal_spread)
    -> void {
  using Compute = double;
  const int cy = cells.g.cy, cz = cells.g.cz;
  const auto cell_rows = static_cast<std::ptrdiff_t>(cy) * cz;
  tf::parallel_for_each(
      tf::make_sequence_range(cell_rows),
      [&](std::ptrdiff_t row, row_gradients<Compute, Samples> &gradients) {
        const int z = int(row / cy), y = int(row % cy);
        gradients.open(y, z);
        const auto active_end = cells.active_offsets[row + 1];
        for (auto active = cells.active_offsets[row]; active < active_end;
             ++active) {
          const int x = cells.active_x[active];
          const auto composite = cells.active_case[active];
          const Index id = cells.active_base[active];
          const unsigned retained = cells.retained_at(x, y, z);
          const int n_components = components().count[composite];
          Index vid = id;

          for (int comp = 0; comp < n_components; ++comp) {
            // A component whose link the domain cut entirely owns no output
            // vertex; deciding that first is what keeps every write below
            // inside this component's own range.
            const int copies = component_id_copies(composite, retained, comp);
            if (copies == 0)
              continue;
            std::array<double, 3> pos[12], nrm[12];
            const int n_crossings = cell_crossings<Compute>(
                cells.g, samples, Compute(iso), x, y, z, composite, comp,
                gradients, pos, nrm, 12);
            tf::remesh::quadric q{};
            double mass[3] = {0, 0, 0};
            int n_planes = 0;
            for (int i = 0; i < n_crossings; ++i) {
              const auto &p = pos[i];
              const auto &n = nrm[i];
              // every crossing is in the centroid; only the ones that state a
              // plane are in the quadric
              for (int d = 0; d < 3; ++d)
                mass[d] += p[d];
              if (n[0] == 0.0 && n[1] == 0.0 && n[2] == 0.0)
                continue;
              ++n_planes;
              const double d_plane = -(n[0] * p[0] + n[1] * p[1] + n[2] * p[2]);
              int ai = 0;
              for (int a = 0; a < 3; ++a)
                for (int b = a; b < 3; ++b, ++ai)
                  q.A[ai] += n[a] * n[b];
              q.b[0] += d_plane * n[0];
              q.b[1] += d_plane * n[1];
              q.b[2] += d_plane * n[2];
              q.c += d_plane * d_plane;
            }
            float spread = 0.f;
            if (refine) {
              // the gate's whole input: the spread of this component's normals,
              // which for unit vectors is 1 - |mean|^2
              double ns[3] = {0, 0, 0};
              for (int i = 0; i < n_crossings; ++i)
                for (int d = 0; d < 3; ++d)
                  ns[d] += nrm[i][d];
              const double inv = n_planes > 0 ? 1.0 / n_planes : 0.0;
              for (auto &v : ns)
                v *= inv;
              spread = static_cast<float>(
                  n_planes >= 2
                      ? 1.0 - (ns[0] * ns[0] + ns[1] * ns[1] + ns[2] * ns[2])
                      : 0.0);
            }

            // the geometric centroid of the crossings, which exists whether or
            // not any of them stated a plane
            const double inv_crossings =
                n_crossings > 0 ? 1.0 / n_crossings : 0.0;
            auto center = tf::make_point(Real(mass[0] * inv_crossings),
                                         Real(mass[1] * inv_crossings),
                                         Real(mass[2] * inv_crossings));
            // Tikhonov toward the mass point baked into the quadric so the
            // Cramer fast path solves the regularized system too (near-
            // parallel planes otherwise pass its determinant test and
            // intersect far away)
            double lambda = stabilizer * (q.A[0] + q.A[3] + q.A[5]) / 3.0;
            q.A[0] += lambda;
            q.A[3] += lambda;
            q.A[5] += lambda;
            for (int d = 0; d < 3; ++d)
              q.b[d] -= lambda * double(center[d]);
            auto vertex = center;
            if (auto opt =
                    tf::remesh::solve_optimal_quadric<Real>(q, center, 0)) {
              // one cell of escape keeps corners sharp; mass point beyond
              bool contained = true;
              const double lo[3] = {
                  cells.g.origin[0] + (x - 1) * cells.g.spacing[0],
                  cells.g.origin[1] + (y - 1) * cells.g.spacing[1],
                  cells.g.origin[2] + (z - 1) * cells.g.spacing[2]};
              for (int d = 0; d < 3; ++d)
                contained = contained && double((*opt)[d]) >= lo[d] &&
                            double((*opt)[d]) <= lo[d] + 3 * cells.g.spacing[d];
              if (contained)
                vertex = *opt;
            }
            for (int c = 0; c < copies; ++c) {
              points[vid + c][0] = vertex[0];
              points[vid + c][1] = vertex[1];
              points[vid + c][2] = vertex[2];
              if (refine)
                normal_spread[vid + c] = spread;
            }
            vid += copies;
          } // component
        }
      },
      row_gradients<Compute, Samples>(cells.g, samples),
      tf::grain(k_dc_row_grain));
}

/// @brief Place the two arc vertices of every marked face.
///
/// Each sits at the mean of the two primal crossings its arc joins: on the
/// shared face, and distinct from the other arc's. Placement decides nothing
/// about incidence.
template <typename Index, typename Points, typename Real, typename Samples>
auto place_arc_vertices(const dual_contour_cells<Index> &cells,
                        const Samples &samples, Real iso,
                        const tf::buffer<marked_face> &marked,
                        const unsigned char *split_mask,
                        const Index *split_base, Index f_origin, Points points)
    -> void {
  using Compute = double;
  tf::parallel_for_each(
      tf::make_sequence_range(std::ptrdiff_t(marked.size())),
      [&](std::ptrdiff_t i) {
        const auto &m = marked[i];
        const auto cs = cells.active_case[m.entry];
        const int high = m.axis * 2 + 1;
        const Index base =
            f_origin + split_base[m.entry] +
            Index(2 * fe_popcount(split_mask[m.entry] & ((1u << m.axis) - 1u)));
        for (int arc = 0; arc < 2; ++arc) {
          const auto pr =
              patches().arc_pair[cs][std::size_t(high)][std::size_t(arc)];
          const int o[2] = {pr >> 2, pr & 3};
          std::array<double, 3> mean{};
          for (int j = 0; j < 2; ++j) {
            const int el = k_face_edges[std::size_t(high)][std::size_t(o[j])];
            const auto &c0 =
                k_corner_offset[std::size_t(k_cell_edge[std::size_t(el)][0])];
            const auto &c1 =
                k_corner_offset[std::size_t(k_cell_edge[std::size_t(el)][1])];
            int eaxis = 0;
            for (int d = 0; d < 3; ++d)
              if (c0[std::size_t(d)] != c1[std::size_t(d)])
                eaxis = d;
            const auto p = primal_crossing<Compute>(
                cells.g, samples, Compute(iso), m.x + c0[0], m.y + c0[1],
                m.z + c0[2], eaxis);
            for (int d = 0; d < 3; ++d)
              mean[std::size_t(d)] += 0.5 * p[std::size_t(d)];
          }
          const Index fid = base + Index(arc);
          for (int d = 0; d < 3; ++d)
            points[fid][d] = Real(mean[std::size_t(d)]);
        }
      },
      tf::checked);
}

} // namespace volume_detail
} // namespace tf
