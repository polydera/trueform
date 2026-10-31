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

#include "../../core/algorithm/compute_offsets.hpp"
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/algorithm/parallel_iota.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/frame_of.hpp"
#include "../../core/point.hpp"
#include "../../core/range.hpp"
#include "../../core/small_vector.hpp"
#include "../../core/transformed.hpp"
#include "../../core/views/drop.hpp"
#include "../../core/views/indirect_range.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../canonical_plane.hpp"
#include "../vertex_converter.hpp"
#include "./make_support_plane.hpp"

#include "tbb/parallel_sort.h"

#include <algorithm>
#include <cstddef>
#include <iterator>

namespace tf::exact::door {

/// The flat world the placement reads. It asks one question of every
/// vertex of every form at once, so the facts it asks are stated in one
/// flat space first: the lattice point of every vertex, the exact plane
/// every face stands on, the corners of every face and the faces of every
/// vertex — the two sides of one incidence — each a CSR. A tag is then
/// nothing but an offset into those.
///
/// A face that names no plane keeps the zero quadruple, which no plane
/// can take, so a constraint list carries no validity bit beside it.
///
/// ONE POSITION IS ONE VERTEX. Two forms may state one lattice point as a
/// vertex each — a wall two domains share, a surface and the solid it
/// bounds — and the input thereby states that the two coincide. The band
/// must not part what the input joined, so every vertex at one lattice
/// point carries the faces of every vertex at that point: `incidence` is
/// the union over the coincidence class, and every decision the placement
/// takes from a vertex's faces — what it rides, what it keeps, where it
/// stands — is then the same decision for the whole class. `twins` lists,
/// per vertex, the other vertices at its point, for the facts stated per
/// vertex that the class must share.
template <typename Index, typename Int, typename RealType>
struct placement_tables {
  tf::buffer<tf::point<Int, 3>> points;
  tf::buffer<tf::exact::canonical_plane<Int>> planes;
  tf::buffer<Index> corner_offsets;
  tf::buffer<Index> corners;
  tf::buffer<Index> incidence_offsets;
  tf::buffer<Index> incidence;
  tf::buffer<Index> twin_offsets;
  tf::buffer<Index> twins;

  /// The other vertices at a vertex's lattice point.
  auto twins_of(std::size_t vertex) const {
    return tf::make_range(twins.begin() + twin_offsets[vertex],
                          twins.begin() + twin_offsets[vertex + 1]);
  }

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
  /// addressing.
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
    join_coincident_vertices();
  }

private:
  /// The coincidence classes: the vertices sorted by lattice point, the
  /// runs of equal points the classes, each member of a class of more than
  /// one given the union of the class's faces and the class's other
  /// members as twins.
  auto join_coincident_vertices() -> void {
    const auto n_vertices = points.size();
    tf::buffer<Index> order;
    order.allocate(n_vertices);
    tf::parallel_iota(order, Index(0));
    tbb::parallel_sort(order.begin(), order.end(), [this](Index a, Index b) {
      const auto &p = points[std::size_t(a)];
      const auto &q = points[std::size_t(b)];
      return p != q ? p < q : a < b;
    });
    tf::buffer<Index> class_offsets;
    tf::compute_offsets(order, std::back_inserter(class_offsets), 0,
                        [this](Index a, Index b) {
                          return points[std::size_t(a)] ==
                                 points[std::size_t(b)];
                        });
    if (class_offsets.size() == 0)
      class_offsets.push_back(0);
    const auto n_classes = class_offsets.size() - 1;
    tf::buffer<Index> class_of;
    class_of.allocate(n_vertices);
    tf::parallel_for_each(
        tf::make_sequence_range(n_classes),
        [&](std::size_t c) {
          for (auto k = class_offsets[c]; k < class_offsets[c + 1]; ++k)
            class_of[std::size_t(order[std::size_t(k)])] = Index(c);
        },
        tf::checked);
    // counts and one prefix for both tables, then disjoint writes per class
    tf::buffer<Index> joined_offsets;
    joined_offsets.allocate(n_vertices + 1);
    twin_offsets.allocate(n_vertices + 1);
    joined_offsets[0] = Index(0);
    twin_offsets[0] = Index(0);
    struct class_t {
      tf::small_vector<Index, 32> faces;
    };
    tf::buffer<Index> class_size;
    class_size.allocate(n_classes);
    tf::parallel_for_each(
        tf::make_sequence_range(n_classes),
        [&](std::size_t c, class_t &local) {
          const auto members = class_offsets[c + 1] - class_offsets[c];
          local.faces.clear();
          for (auto k = class_offsets[c]; k < class_offsets[c + 1]; ++k) {
            const auto v = std::size_t(order[std::size_t(k)]);
            for (auto f = incidence_offsets[v]; f < incidence_offsets[v + 1];
                 ++f)
              local.faces.push_back(incidence[std::size_t(f)]);
          }
          if (members > 1) {
            std::sort(local.faces.begin(), local.faces.end());
            local.faces.erase(std::unique(local.faces.begin(), local.faces.end()),
                              local.faces.end());
          }
          class_size[c] = Index(local.faces.size());
          for (auto k = class_offsets[c]; k < class_offsets[c + 1]; ++k) {
            const auto v = std::size_t(order[std::size_t(k)]);
            joined_offsets[v + 1] = Index(local.faces.size());
            twin_offsets[v + 1] = members - 1;
          }
        },
        class_t{}, tf::checked);
    for (std::size_t v = 1; v <= n_vertices; ++v) {
      joined_offsets[v] += joined_offsets[v - 1];
      twin_offsets[v] += twin_offsets[v - 1];
    }
    tf::buffer<Index> joined;
    joined.allocate(std::size_t(joined_offsets[n_vertices]));
    twins.allocate(std::size_t(twin_offsets[n_vertices]));
    tf::parallel_for_each(
        tf::make_sequence_range(n_classes),
        [&](std::size_t c, class_t &local) {
          const auto members = class_offsets[c + 1] - class_offsets[c];
          local.faces.clear();
          for (auto k = class_offsets[c]; k < class_offsets[c + 1]; ++k) {
            const auto v = std::size_t(order[std::size_t(k)]);
            for (auto f = incidence_offsets[v]; f < incidence_offsets[v + 1];
                 ++f)
              local.faces.push_back(incidence[std::size_t(f)]);
          }
          if (members > 1) {
            std::sort(local.faces.begin(), local.faces.end());
            local.faces.erase(std::unique(local.faces.begin(), local.faces.end()),
                              local.faces.end());
          }
          for (auto k = class_offsets[c]; k < class_offsets[c + 1]; ++k) {
            const auto v = std::size_t(order[std::size_t(k)]);
            std::copy(local.faces.begin(), local.faces.end(),
                      joined.begin() + joined_offsets[v]);
            auto at = twins.begin() + twin_offsets[v];
            for (auto j = class_offsets[c]; j < class_offsets[c + 1]; ++j)
              if (j != k)
                *at++ = order[std::size_t(j)];
          }
        },
        class_t{}, tf::checked);
    incidence_offsets = std::move(joined_offsets);
    incidence = std::move(joined);
  }
};

} // namespace tf::exact::door
