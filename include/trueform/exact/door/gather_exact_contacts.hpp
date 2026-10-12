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

#include "../../core/aabb.hpp"
#include "../../core/algorithm/generic_generate.hpp"
#include "../../core/buffer.hpp"
#include "../../core/distance.hpp"
#include "../../core/frame_of.hpp"
#include "../../core/point.hpp"
#include "../../core/segment.hpp"
#include "../../core/small_vector.hpp"
#include "../../core/views/indirect_range.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../spatial/search.hpp"
#include "../tag_of_flat_vertex.hpp"
#include "../vertex_converter.hpp"
#include "./placement_tables.hpp"
#include "./point_on_face.hpp"
#include "./segments_cross.hpp"

#include "tbb/parallel_sort.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>

namespace tf::exact::door {

/// The identifications the input states exactly between two forms, which no
/// placement may undo: a vertex on another form's closed face, and an edge
/// crossing another form's edge. Each is the flat ids of its own features,
/// sorted and unique. Every other meeting of two edges is an end on the
/// other, so the face class states it, once.
template <typename Index> struct exact_contacts {
  tf::buffer<std::array<Index, 2>> on_face;
  tf::buffer<std::array<Index, 4>> edges;
};

/// Every vertex on another form's closed face, exactly. A vertex on a face
/// stands within the band of it, so the discovery admitted the face and the
/// exact test is asked of the admissions alone.
template <typename Index, typename Int, typename RealType>
auto gather_face_contacts(const placement_tables<Index, Int, RealType> &tables,
                          exact_contacts<Index> &out) -> void {
  struct local_t {
    tf::small_vector<tf::point<Int, 3>, 8> corners;
  };
  out.on_face.clear();
  tf::generic_generate(
      tf::make_sequence_range(tables.points.size()), out.on_face,
      [&tables](std::size_t v, tf::buffer<std::array<Index, 2>> &into,
                local_t &local) {
        for (auto k = tables.contact_offsets[v];
             k < tables.contact_offsets[v + 1]; ++k) {
          const auto f = tables.contacts[std::size_t(k)];
          local.corners.clear();
          for (const auto c : tables.face_corners(std::size_t(f)))
            local.corners.push_back(tables.points[std::size_t(c)]);
          if (point_on_face(tables.points[v], local.corners))
            into.push_back({Index(v), f});
        }
      },
      local_t{});
  tbb::parallel_sort(out.on_face.begin(), out.on_face.end());
}

/// Every edge of a vertex in `scan` crossing another form's edge, exactly,
/// added to `out.edges`. The edges a placement can disturb are the edges of
/// the vertices it moved, so only those are walked: each against every other
/// form's tree, pruned by the segment's own distance to a node's box against
/// the lattice's rounding, and
/// the faces that survive offer their edges to the exact test. Two crossing
/// segments stand at distance zero on the lattice, so a pair whose distance
/// taken in double exceeds that evaluation's own error cannot cross and
/// never reaches the exact predicate.
template <typename Index, typename Int, typename RealType, typename ApplyToForm>
auto gather_edge_contacts(
    const placement_tables<Index, Int, RealType> &tables,
    const tf::exact::vertex_converter<Int, RealType, 3> &converter,
    const ApplyToForm &apply_to_form, Index n_tags,
    const tf::buffer<Index> &vertex_offsets,
    const tf::buffer<Index> &face_offsets, const tf::buffer<char> &scan,
    exact_contacts<Index> &out) -> void {
  const auto slack = double(converter.coords.deconvert_tolerance(Int(4)));
  tf::buffer<std::array<Index, 4>> found;
  tf::generic_generate(
      tf::make_sequence_range(tables.points.size()), found,
      [&](std::size_t v, tf::buffer<std::array<Index, 4>> &into) {
        if (!scan[v])
          return;
        const auto tag =
            tf::exact::tag_of_flat_vertex(vertex_offsets, Index(v));
        for (auto k = tables.incidence_offsets[v];
             k < tables.incidence_offsets[v + 1]; ++k) {
          const auto corners = tables.face_corners(
              std::size_t(tables.incidence[std::size_t(k)]));
          const auto n = corners.size();
          for (std::size_t i = 0; i < n; ++i) {
            if (std::size_t(corners[i]) != v)
              continue;
            for (const auto w :
                 {corners[(i + 1) % n], corners[(i + n - 1) % n]}) {
              const auto &a0 = tables.points[v];
              const auto &a1 = tables.points[std::size_t(w)];
              const auto lattice_segment = [](const tf::point<Int, 3> &p,
                                              const tf::point<Int, 3> &q) {
                return tf::make_segment_between_points(
                    tf::point<double, 3>{double(p[0]), double(p[1]),
                                         double(p[2])},
                    tf::point<double, 3>{double(q[0]), double(q[1]),
                                         double(q[2])});
              };
              const auto mine = lattice_segment(a0, a1);
              double magnitude = 0;
              for (std::size_t d = 0; d < 3; ++d)
                magnitude =
                    std::max(magnitude, std::max(std::abs(double(a0[d])),
                                                 std::abs(double(a1[d]))));
              const double screen = 4.0 + magnitude * 0x1p-40;
              const auto p0 = converter.deconvert(a0);
              const auto p1 = converter.deconvert(a1);
              const auto world = tf::make_segment_between_points(
                  tf::point<double, 3>{double(p0[0]), double(p0[1]),
                                       double(p0[2])},
                  tf::point<double, 3>{double(p1[0]), double(p1[1]),
                                       double(p1[2])});
              // the distance is taken in double on world coordinates, so the
              // reach covers that evaluation's own error beside the rounding
              double extent = 0;
              for (std::size_t d = 0; d < 3; ++d)
                extent = std::max(extent, std::max(std::abs(double(p0[d])),
                                                   std::abs(double(p1[d]))));
              const double reach = slack + extent * 0x1p-36;
              const double reach2 = reach * reach;
              for (Index other = 0; other < n_tags; ++other) {
                if (other == tag)
                  continue;
                const auto face_base = face_offsets[std::size_t(other)];
                apply_to_form(other, [&](const auto &form) {
                  tf::search(
                      form,
                      [&world, reach2](const auto &bv) {
                        const auto box = tf::make_aabb(
                            tf::point<double, 3>{double(bv.min[0]),
                                                 double(bv.min[1]),
                                                 double(bv.min[2])},
                            tf::point<double, 3>{double(bv.max[0]),
                                                 double(bv.max[1]),
                                                 double(bv.max[2])});
                        return tf::distance2(box, world) <= reach2;
                      },
                      [&](const auto &primitive) {
                        const auto face = tables.face_corners(
                            std::size_t(face_base + Index(primitive.id())));
                        const auto m = face.size();
                        for (std::size_t j = 0; j < m; ++j) {
                          const auto b0 = face[j], b1 = face[(j + 1) % m];
                          if (tf::distance2(
                                  mine, lattice_segment(
                                            tables.points[std::size_t(b0)],
                                            tables.points[std::size_t(b1)])) >
                              screen * screen)
                            continue;
                          if (!segments_cross<Int>(
                                  a0, a1, tables.points[std::size_t(b0)],
                                  tables.points[std::size_t(b1)]))
                            continue;
                          // one key per crossing pair, whichever side
                          // found it
                          const std::array<Index, 2> mine{
                              std::min(Index(v), w), std::max(Index(v), w)};
                          const std::array<Index, 2> theirs{std::min(b0, b1),
                                                            std::max(b0, b1)};
                          const auto &lo = mine < theirs ? mine : theirs;
                          const auto &hi = mine < theirs ? theirs : mine;
                          into.push_back({lo[0], lo[1], hi[0], hi[1]});
                        }
                      });
                });
              }
            }
          }
        }
      });
  const auto from = out.edges.size();
  out.edges.reallocate(from + found.size());
  std::copy(found.begin(), found.end(), out.edges.begin() + from);
  tbb::parallel_sort(out.edges.begin(), out.edges.end());
  out.edges.erase(std::unique(out.edges.begin(), out.edges.end()),
                  out.edges.end());
}

} // namespace tf::exact::door
