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

#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/frame_of.hpp"
#include "../../core/point.hpp"
#include "../../core/range.hpp"
#include "../../core/transformed.hpp"
#include "../../core/views/drop.hpp"
#include "../../core/views/indirect_range.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../canonical_plane.hpp"
#include "../vertex_converter.hpp"
#include "./make_support_plane.hpp"

#include <cstddef>

namespace tf::exact::door {

/// The flat world the placement reads. It asks one question of every
/// vertex of every form at once, so the facts it asks are stated in one
/// flat space first: the lattice point of every vertex, the exact plane
/// every face stands on, the corners of every face and the faces of every
/// vertex — the two sides of one incidence — the other forms' faces every
/// vertex was admitted against, and the exact identifications every vertex
/// keeps, each a CSR. A tag is then nothing
/// but an offset into those.
///
/// A face that names no plane keeps the zero quadruple, which no plane
/// can take, so a constraint list carries no validity bit beside it.
template <typename Index, typename Int, typename RealType>
struct placement_tables {
  tf::buffer<tf::point<Int, 3>> points;
  tf::buffer<tf::exact::canonical_plane<Int>> planes;
  tf::buffer<Index> corner_offsets;
  tf::buffer<Index> corners;
  tf::buffer<Index> incidence_offsets;
  tf::buffer<Index> incidence;
  tf::buffer<Index> contact_offsets;
  tf::buffer<Index> contacts;
  /// Per vertex the exact identifications it keeps: a face row `f >= 0`
  /// for a face it stands on exactly, `-1 - k` for the plane `k` of
  /// `held_planes` an exact edge contact of its spans.
  tf::buffer<Index> held_offsets;
  tf::buffer<Index> held;
  tf::buffer<tf::exact::canonical_plane<Int>> held_planes;

  /// The flat corners of a flat face.
  auto face_corners(std::size_t face) const {
    return tf::make_range(corners.begin() + corner_offsets[face],
                          corners.begin() + corner_offsets[face + 1]);
  }
  /// Whether every face is a triangle: a face states at least three
  /// corners, so the total says it.
  auto triangles() const -> bool {
    return corners.size() == 3 * (corner_offsets.size() - 1);
  }

  /// `vertex_offsets` and `face_offsets` are the caller's flat prefixes,
  /// so these tables and the placed table they feed share one
  /// addressing. The contacts are the discovery's, stated after.
  template <typename ApplyToForm>
  auto build(const tf::exact::vertex_converter<Int, RealType, 3> &converter,
             const ApplyToForm &apply_to_form, Index n_tags,
             const tf::buffer<Index> &vertex_offsets,
             const tf::buffer<Index> &face_offsets) -> void {
    const auto n_vertices = std::size_t(vertex_offsets[std::size_t(n_tags)]);
    const auto n_faces = std::size_t(face_offsets[std::size_t(n_tags)]);
    points.allocate(n_vertices);
    planes.allocate(n_faces);
    corner_offsets.allocate(n_faces + 1);
    corner_offsets[0] = Index(0);
    incidence_offsets.allocate(n_vertices + 1);
    incidence_offsets[0] = Index(0);

    for (Index tag = 0; tag < n_tags; ++tag)
      apply_to_form(tag, [&, tag](const auto &form) {
        const auto vertex_base = std::size_t(vertex_offsets[std::size_t(tag)]);
        const auto face_base = std::size_t(face_offsets[std::size_t(tag)]);
        const auto frame = tf::frame_of(form);
        const auto form_points = form.points();
        auto &table = points;
        tf::parallel_for_each(
            tf::make_sequence_range(form_points.size()),
            [&table, &converter, form_points, frame,
             vertex_base](std::size_t id) {
              table[vertex_base + id] =
                  converter.convert(tf::transformed(form_points[id], frame));
            },
            tf::checked);

        const auto faces = form.faces();
        auto &plane_table = planes;
        auto corners_of = [this, vertex_base](const auto &face) {
          return tf::make_indirect_range(
              face, tf::drop(tf::make_range(points), vertex_base));
        };
        tf::parallel_for_each(
            tf::make_sequence_range(faces.size()),
            [&plane_table, faces, corners_of, face_base](std::size_t id) {
              auto &plane = plane_table[face_base + id];
              plane = tf::exact::canonical_plane<Int>{};
              make_support_plane<Int>(corners_of(faces[Index(id)]), plane);
            },
            tf::checked);

        auto &arities = corner_offsets;
        tf::parallel_for_each(
            tf::make_sequence_range(faces.size()),
            [&arities, faces, face_base](std::size_t id) {
              arities[face_base + id + 1] = Index(faces[Index(id)].size());
            },
            tf::checked);

        auto &counts = incidence_offsets;
        const auto membership = form.face_membership();
        tf::parallel_for_each(
            tf::make_sequence_range(Index(form_points.size())),
            [&counts, membership, vertex_base](Index id) {
              counts[vertex_base + std::size_t(id) + 1] =
                  Index(membership[id].size());
            },
            tf::checked);
      });

    for (std::size_t f = 1; f <= n_faces; ++f)
      corner_offsets[f] += corner_offsets[f - 1];
    corners.allocate(std::size_t(corner_offsets[n_faces]));
    for (std::size_t v = 1; v <= n_vertices; ++v)
      incidence_offsets[v] += incidence_offsets[v - 1];
    incidence.allocate(std::size_t(incidence_offsets[n_vertices]));

    for (Index tag = 0; tag < n_tags; ++tag)
      apply_to_form(tag, [&, tag](const auto &form) {
        const auto vertex_base = std::size_t(vertex_offsets[std::size_t(tag)]);
        const auto face_base = face_offsets[std::size_t(tag)];
        const auto faces = form.faces();
        auto &corner_at = corner_offsets;
        auto &corner_data = corners;
        tf::parallel_for_each(
            tf::make_sequence_range(faces.size()),
            [&corner_at, &corner_data, faces, face_base,
             vertex_base](std::size_t id) {
              auto at = std::size_t(corner_at[std::size_t(face_base) + id]);
              for (const auto corner : faces[Index(id)])
                corner_data[at++] = Index(vertex_base) + Index(corner);
            },
            tf::checked);

        const auto membership = form.face_membership();
        auto &flat_offsets = incidence_offsets;
        auto &flat_data = incidence;
        tf::parallel_for_each(
            tf::make_sequence_range(Index(form.points().size())),
            [&flat_offsets, &flat_data, membership, vertex_base,
             face_base](Index id) {
              auto at =
                  std::size_t(flat_offsets[vertex_base + std::size_t(id)]);
              for (const auto face : membership[id])
                flat_data[at++] = face_base + Index(face);
            },
            tf::checked);
      });
  }
};

} // namespace tf::exact::door
