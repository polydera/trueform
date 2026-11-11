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
#include "../../core/algorithm/mask_to_map.hpp"
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/algorithm/sequenced_generate.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/tetrahedra_buffer.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../topology/cdt/dt3/tet_aligned_position.hpp"
#include "../../topology/cdt/dt3/tet_deep_lattice.hpp"
#include "../../topology/cdt/dt3/tet_site_position.hpp"
#include <cmath>
#include <cstddef>

namespace tf::tetmesh {

/// Emit the cells `keep` admits of the published complex `owner` as a
/// tetrahedra buffer, with every site `used` names among its points. The
/// frame stands at `2^frame_bits` of the lattice `converter` maps back, a
/// site below it at `2^(frame_bits + depth)`, its position aligned to its
/// own depth as one integer and rounded once; `point_of` maps each site
/// slot to its output point, `-1` for none.
template <typename RealOut, typename Index, typename Owner, typename Keep,
          typename Converter>
auto emit_tet_mesh(const Owner &owner, const Keep &keep, tf::buffer<char> &used,
                   const Converter &converter, int frame_bits,
                   tf::buffer<Index> &point_of)
    -> tf::tetrahedra_buffer<Index, RealOut> {
  const std::size_t n_finite = owner._n_finite;
  tf::tetrahedra_buffer<Index, RealOut> mesh;
  tf::parallel_for_each(
      tf::make_sequence_range(n_finite),
      [&owner, &keep, &used](std::size_t cell) {
        if (keep(cell))
          for (auto corner : owner._corners[cell])
            used[std::size_t(corner)] = 1;
      },
      tf::checked);
  const std::size_t n_sites = tf::topology::cdt::dt3::tet_site_count(owner);
  point_of.allocate(n_sites);
  // the used sites numbered in site order: a prefix scan
  const Index next = tf::mask_to_map(used, point_of, Index(-1));
  auto &points = mesh.points_buffer();
  points.allocate(std::size_t(next));
  const RealOut unit = std::ldexp(RealOut(1), -frame_bits);
  tf::parallel_for_each(
      tf::make_sequence_range(n_sites),
      [&](std::size_t site) {
        if (point_of[site] < 0)
          return;
        const auto at = tf::topology::cdt::dt3::tet_site_position(
            owner, typename Owner::index_type(site));
        auto out = points[std::size_t(point_of[site])];
        if (!at.depth) {
          for (std::size_t k = 0; k < 3; ++k)
            out[k] =
                RealOut(RealOut(at.floor[k]) * unit / RealOut(converter.scale) +
                        RealOut(converter.offset[k]));
          return;
        }
        using lattice = tf::topology::cdt::dt3::tet_deep_lattice<
            typename Owner::int_type, typename Owner::coord_type>;
        const auto x = tf::topology::cdt::dt3::tet_aligned_position<
            typename lattice::coord_type>(at, at.depth);
        const RealOut below = std::ldexp(RealOut(1), -frame_bits - at.depth);
        for (std::size_t k = 0; k < 3; ++k)
          out[k] = RealOut(RealOut(x[k]) * below / RealOut(converter.scale) +
                           RealOut(converter.offset[k]));
      },
      tf::checked);
  auto &tetras = mesh.tetras_buffer().data_buffer();
  tetras.clear();
  tf::sequenced_generate(
      tf::make_sequence_range(n_finite), tetras,
      [&owner, &keep, &point_of](std::size_t cell, auto &out) {
        if (!keep(cell))
          return;
        for (auto corner : owner._corners[cell])
          out.push_back(point_of[std::size_t(corner)]);
      },
      tf::checked);
  return mesh;
}

} // namespace tf::tetmesh
