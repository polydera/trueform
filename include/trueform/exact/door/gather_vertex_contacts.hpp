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

#include "../../core/algorithm/generate_offset_blocks.hpp"
#include "../../core/buffer.hpp"
#include "../../core/closest_point_on_triangle.hpp"
#include "../../core/distance.hpp"
#include "../../core/frame_of.hpp"
#include "../../core/point.hpp"
#include "../../core/transformed.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../spatial/search.hpp"
#include "../tag_of_flat_vertex.hpp"
#include "../vertex_converter.hpp"
#include "./placement_tables.hpp"

#include <array>
#include <cmath>
#include <cstddef>

namespace tf::exact::door {

/// The faces of the OTHER forms every vertex stands within the band of,
/// one block per flat vertex: the contacts the placement must honour.
///
/// The vertex is the carrier: each one walks every other form's own tree
/// once, pruned by its distance to a node's box, and tests the faces that
/// survive against its lattice point. Every block is the vertex's own, so
/// the table is built in flat order with no sort and no join.
///
/// THE TEST IS CONSERVATIVE AND MAY BE, because admission only adds a
/// constraint the placement then certifies: a face admitted a hair beyond
/// the band either lands its vertex within the band exactly on the face's
/// plane or leaves the vertex where the input put it. So the box prune is
/// widened by the lattice's own rounding, and the face test is taken in
/// double on lattice coordinates with a margin for the rounding of both.
template <typename Index, typename Int, typename RealType, typename ApplyToForm>
auto gather_vertex_contacts(
    placement_tables<Index, Int, RealType> &tables,
    const tf::exact::vertex_converter<Int, RealType, 3> &converter,
    const ApplyToForm &apply_to_form, Index n_tags,
    const tf::buffer<Index> &vertex_offsets,
    const tf::buffer<Index> &face_offsets, Int tolerance) -> void {
  // a lattice point sits half a unit from its source along each axis, so
  // two of them stand at most two units nearer than their sources do
  const auto reach =
      double(converter.coords.deconvert_tolerance(tolerance + Int(4))) *
      (1.0 + 1e-6);
  const auto reach2 = RealType(reach * reach);
  const auto band = double(tolerance);
  const auto &points = tables.points;
  const auto lattice_point = [&points](Index flat) {
    const auto &p = points[std::size_t(flat)];
    return tf::point<double, 3>{double(p[0]), double(p[1]), double(p[2])};
  };

  tf::generate_offset_blocks(
      tf::make_sequence_range(vertex_offsets[std::size_t(n_tags)]),
      tables.contact_offsets, tables.contacts,
      [&](Index flat, tf::buffer<Index> &into) {
        const auto tag = tf::exact::tag_of_flat_vertex(vertex_offsets, flat);
        const auto at = lattice_point(flat);
        const double magnitude = std::fmax(
            std::fabs(at[0]), std::fmax(std::fabs(at[1]), std::fabs(at[2])));
        const double limit = band + 2.0 + magnitude * 0x1p-50;
        const double limit2 = limit * limit;
        apply_to_form(tag, [&](const auto &own) {
          const auto point = tf::transformed(
              own.points()[flat - vertex_offsets[std::size_t(tag)]],
              tf::frame_of(own));
          for (Index other = 0; other < n_tags; ++other) {
            if (other == tag)
              continue;
            const auto base = vertex_offsets[std::size_t(other)];
            const auto face_base = face_offsets[std::size_t(other)];
            apply_to_form(other, [&](const auto &form) {
              const auto faces = form.faces();
              tf::search(
                  form,
                  [&point, reach2](const auto &bv) {
                    return tf::distance2(point, bv) <= reach2;
                  },
                  [&](const auto &primitive) {
                    const auto face = faces[primitive.id()];
                    const auto first = lattice_point(base + Index(face[0]));
                    for (std::size_t k = 1; k + 1 < face.size(); ++k) {
                      const std::array<tf::point<double, 3>, 3> triangle{
                          first, lattice_point(base + Index(face[k])),
                          lattice_point(base + Index(face[k + 1]))};
                      if (tf::distance2(
                              tf::closest_point_on_triangle(triangle, at),
                              at) <= limit2) {
                        into.push_back(face_base + Index(primitive.id()));
                        return;
                      }
                    }
                  });
            });
          }
        });
      });
}

} // namespace tf::exact::door
