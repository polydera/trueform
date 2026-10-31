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
#include "../../core/algorithm/parallel_fill.hpp"
#include "../../core/buffer.hpp"
#include "../../core/polygons_buffer.hpp"
#include "../../core/small_vector.hpp"
#include "./piece_corner_point.hpp"
#include "./reseat_pieces.hpp"
#include "./state_reseat_reach.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace tf::arrangement::reseat {

/// What one delta face is in the world: a piece of a carved face, a world
/// face restated whole, or a world face whose contacts only feed the
/// restated ones.
enum class delta_role : std::int8_t { piece, restated, neighbour };

template <typename Index> struct delta_face_origin {
  delta_role role;
  Index tag;
  /// the piece's row in `reseat_pieces::pieces`, or the world source face
  Index id;
};

/// The delta forms: per world tag, the pieces of its carved faces, its
/// restated faces and its neighbours, as they stand, on the world's own
/// lattice points — an integral form, so the delta's converter is the
/// identity and every exact fact it states is the world's.
template <typename Index, typename Int> struct reseat_delta_forms {
  tf::small_vector<tf::polygons_buffer<Index, Int, 3, 3>, 4> meshes;
  /// per tag, per delta vertex: the world corner it is
  tf::small_vector<tf::buffer<piece_corner<Index>>, 4> vertices;
  /// per tag, per delta face: what it is in the world
  tf::small_vector<tf::buffer<delta_face_origin<Index>>, 4> faces;
};

template <typename Index, typename Int, typename ApplyToForm, typename Reader,
          typename Created>
auto make_reseat_delta_forms(const reseat_pieces<Index, Int> &pieces,
                             const reseat_reach<Index> &reach,
                             const standing_pieces<Index> &standing,
                             Index n_tags, const ApplyToForm &apply_to_form,
                             const Reader &reader, const Created &created)
    -> reseat_delta_forms<Index, Int> {
  using corner_t = piece_corner<Index>;
  reseat_delta_forms<Index, Int> out;
  for (Index tag = 0; tag < n_tags; ++tag) {
    out.vertices.emplace_back();
    out.faces.emplace_back();
    out.meshes.emplace_back();
    auto &vertices = out.vertices.back();
    auto &faces = out.faces.back();
    auto &mesh = out.meshes.back();
    tf::buffer<std::array<corner_t, 3>> face_corners;
    for (std::size_t k = 0; k < pieces.pieces.size(); ++k) {
      const auto &piece = pieces.pieces[k];
      if (piece.tag != tag)
        continue;
      face_corners.push_back(piece.corners);
      faces.push_back({delta_role::piece, tag, Index(k)});
    }
    const auto add_world = [&](const tf::buffer<std::array<Index, 2>> &set,
                               delta_role role) {
      apply_to_form(tag, [&](const auto &form) {
        const auto world_faces = form.faces();
        for (const auto &key : set) {
          if (key[0] != tag)
            continue;
          const auto standing_piece = standing.of_key(tag, key[1]);
          if (standing_piece != Index(-1))
            face_corners.push_back(
                standing.pieces[std::size_t(standing_piece)].corners);
          else {
            const auto f = world_faces[key[1]];
            face_corners.push_back({corner_t{std::int16_t(tag), Index(f[0])},
                                    corner_t{std::int16_t(tag), Index(f[1])},
                                    corner_t{std::int16_t(tag), Index(f[2])}});
          }
          faces.push_back({role, tag, key[1]});
        }
      });
    };
    add_world(reach.restated, delta_role::restated);
    add_world(reach.neighbours, delta_role::neighbour);
    // a corner's delta vertex, found on first sight: originals by their own
    // id, seats by their row, created identities by their rank among the
    // ones named, `-1` until named
    tf::buffer<Index> of_original;
    apply_to_form(tag, [&](const auto &form) {
      of_original.allocate(std::size_t(form.points().size()));
    });
    tf::parallel_fill(of_original, Index(-1));
    tf::buffer<Index> of_seat;
    of_seat.allocate(pieces.seats.size());
    tf::parallel_fill(of_seat, Index(-1));
    tf::buffer<Index> named_created;
    for (const auto &face : face_corners)
      for (const auto &c : face)
        if (c.tag == std::int16_t(-1))
          named_created.push_back(c.id);
    std::sort(named_created.begin(), named_created.end());
    named_created.erase_till_end(
        std::unique(named_created.begin(), named_created.end()));
    tf::buffer<Index> of_created;
    of_created.allocate(named_created.size());
    std::fill(of_created.begin(), of_created.end(), Index(-1));
    const auto vertex_of = [&](const corner_t &c) -> Index {
      auto &slot =
          c.tag >= 0 ? of_original[std::size_t(c.id)]
          : c.tag == seat_corner_tag
              ? of_seat[std::size_t(c.id)]
              : of_created[std::size_t(
                    std::lower_bound(named_created.begin(),
                                     named_created.end(), c.id) -
                    named_created.begin())];
      if (slot == Index(-1)) {
        slot = Index(vertices.size());
        vertices.push_back(c);
      }
      return slot;
    };
    tf::buffer<std::array<Index, 3>> corners;
    corners.allocate(face_corners.size());
    for (std::size_t f = 0; f < face_corners.size(); ++f)
      for (std::size_t k = 0; k < 3; ++k)
        corners[f][k] = vertex_of(face_corners[f][k]);
    // the delta's vertices in the world's own order — originals, then the
    // world's created identities, then seats: every rank the delta forms —
    // its vertex keys, its edge classes, the roots a class is born on — is
    // then the world's rank restricted, so a point both name is rounded
    // alike
    {
      tf::buffer<Index> order;
      order.allocate(vertices.size());
      for (std::size_t v = 0; v < vertices.size(); ++v)
        order[v] = Index(v);
      const auto kind = [](const corner_t &c) {
        return c.tag >= 0 ? 0 : c.tag == seat_corner_tag ? 2 : 1;
      };
      std::sort(order.begin(), order.end(), [&](Index x, Index y) {
        const auto &a = vertices[std::size_t(x)];
        const auto &b = vertices[std::size_t(y)];
        if (kind(a) != kind(b))
          return kind(a) < kind(b);
        return a.id < b.id;
      });
      tf::buffer<Index> rank;
      rank.allocate(vertices.size());
      tf::buffer<corner_t> sorted_vertices;
      sorted_vertices.allocate(vertices.size());
      for (std::size_t r = 0; r < order.size(); ++r) {
        rank[std::size_t(order[r])] = Index(r);
        sorted_vertices[r] = vertices[std::size_t(order[r])];
      }
      for (auto &face : corners)
        for (auto &v : face)
          v = rank[std::size_t(v)];
      vertices = std::move(sorted_vertices);
    }
    auto &points = mesh.points_buffer();
    points.allocate(vertices.size());
    for (std::size_t v = 0; v < vertices.size(); ++v)
      points[v] =
          piece_corner_point(vertices[v], reader, pieces.seats, created);
    auto &face_buffer = mesh.faces_buffer();
    face_buffer.allocate(corners.size());
    for (std::size_t f = 0; f < corners.size(); ++f)
      for (std::size_t k = 0; k < 3; ++k)
        face_buffer[f][k] = corners[f][k];
  }
  return out;
}

} // namespace tf::arrangement::reseat
