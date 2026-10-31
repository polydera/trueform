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
#include "../../core/buffer.hpp"
#include "../../core/point.hpp"
#include "../../exact/meta.hpp"
#include "./piece_corner_point.hpp"
#include "./reseat_pieces.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <utility>

namespace tf::arrangement::reseat {

/// The carve, applied to the faces it reaches and to nothing
/// else, round by round: every split edge in every face holding it, its
/// points ordered from its start, then every fanned face in the piece of it
/// that holds its point. A seat is a point, named once per carve.
///
/// A source face enters as it stands: whole, or as the `standing` pieces an
/// earlier carve stated of it. A target naming its source face's own
/// corners names every piece of it, and a target naming a source side names
/// every part of that side its pieces hold.
template <typename Index, typename Int, typename Rounds, typename ApplyToForm,
          typename Reader, typename Created>
auto state_reseat_pieces(const Rounds &rounds,
                         const standing_pieces<Index> &standing, Index n_tags,
                         const ApplyToForm &apply_to_form, const Reader &reader,
                         const Created &created) -> reseat_pieces<Index, Int> {
  using T1 = typename tf::exact::meta<Int>::T1;
  using T2 = typename tf::exact::meta<Int>::T2;
  using T3 = typename tf::exact::meta<Int>::T3;
  using pt = tf::point<Int, 3>;
  using vec1 = std::array<T1, 3>;
  using vec2 = std::array<T2, 3>;
  using corner_t = piece_corner<Index>;
  using piece_t = reseat_piece<Index>;
  using face_key_t = std::array<corner_t, 3>;
  const auto diff = [](const pt &a, const pt &b) {
    return vec1{T1(a[0]) - T1(b[0]), T1(a[1]) - T1(b[1]), T1(a[2]) - T1(b[2])};
  };
  const auto cross = [](const vec1 &a, const vec1 &b) {
    return vec2{T2(a[1]) * T2(b[2]) - T2(a[2]) * T2(b[1]),
                T2(a[2]) * T2(b[0]) - T2(a[0]) * T2(b[2]),
                T2(a[0]) * T2(b[1]) - T2(a[1]) * T2(b[0])};
  };
  const auto sign3 = [](const vec2 &a, const vec2 &b) {
    const T3 d =
        T3(a[0]) * T3(b[0]) + T3(a[1]) * T3(b[1]) + T3(a[2]) * T3(b[2]);
    return (d > T3(0)) - (d < T3(0));
  };
  const auto dot = [](const vec1 &a, const vec1 &b) {
    return T2(a[0]) * T2(b[0]) + T2(a[1]) * T2(b[1]) + T2(a[2]) * T2(b[2]);
  };
  const auto key = [](const pt &p) {
    return std::make_tuple(p[0], p[1], p[2]);
  };
  const auto sorted = [](face_key_t k) {
    std::sort(k.begin(), k.end());
    return k;
  };

  reseat_pieces<Index, Int> out;
  const auto seat_of = [&](const pt &point) -> corner_t {
    for (std::size_t k = 0; k < out.seats.size(); ++k)
      if (out.seats[k] == point)
        return {seat_corner_tag, Index(k)};
    out.seats.push_back(point);
    return {seat_corner_tag, Index(out.seats.size()) - Index(1)};
  };
  const auto position = [&](const corner_t &c) -> pt {
    return piece_corner_point(c, reader, out.seats, created);
  };

  for (Index tag = 0; tag < n_tags; ++tag) {
    tf::buffer<piece_t> pieces;
    tf::buffer<face_key_t> lineage;
    tf::buffer<char> dead;
    tf::buffer<char> touched;
    // per piece, the corners of the source face it lies in
    tf::buffer<std::array<Index, 3>> origin;
    tf::buffer<Index> sources;
    const auto push_piece = [&](const piece_t &piece, const face_key_t &from,
                                char cut, const std::array<Index, 3> &own) {
      pieces.push_back(piece);
      lineage.push_back(from);
      dead.push_back(char(0));
      touched.push_back(cut);
      origin.push_back(own);
    };
    const auto source_corners = [&](Index face) {
      std::array<Index, 3> own{};
      apply_to_form(tag, [&](const auto &form) {
        const auto f = form.faces()[face];
        for (std::size_t k = 0; k < 3; ++k)
          own[k] = Index(f[k]);
      });
      return own;
    };
    // a source face enters the work list as it stands, once
    const auto enter = [&](Index face) {
      if (std::find(sources.begin(), sources.end(), face) != sources.end())
        return;
      sources.push_back(face);
      const auto own = source_corners(face);
      const auto range = standing.of_source(tag, face);
      if (range[0] == range[1]) {
        piece_t piece;
        piece.tag = tag;
        piece.source = face;
        for (std::size_t k = 0; k < 3; ++k) {
          piece.corners[k] = {std::int16_t(tag), own[k]};
          piece.sides[k] = std::int8_t(k);
        }
        push_piece(piece, sorted(piece.corners), char(0), own);
        return;
      }
      for (auto p = range[0]; p < range[1]; ++p)
        push_piece(standing.pieces[p], sorted(standing.pieces[p].corners),
                   char(0), own);
    };
    const auto enter_source_edge = [&](Index a, Index b) {
      apply_to_form(tag, [&](const auto &form) {
        const auto faces = form.faces();
        for (const auto face : form.face_membership()[a]) {
          const auto f = faces[face];
          for (std::size_t k = 0; k < 3; ++k)
            if (Index(f[k]) == b) {
              enter(Index(face));
              break;
            }
        }
      });
    };
    // every face holding an edge: by its own corners when it is a source
    // side, else by the source side a piece holding it lies on
    const auto enter_edge = [&](const corner_t &a, const corner_t &b) {
      if (a.tag >= 0 && b.tag >= 0) {
        enter_source_edge(a.id, b.id);
        return;
      }
      tf::buffer<std::array<Index, 2>> sides;
      for (std::size_t f = 0; f < pieces.size(); ++f) {
        if (dead[f])
          continue;
        for (std::size_t k = 0; k < 3; ++k) {
          const auto x = pieces[f].corners[k];
          const auto y = pieces[f].corners[(k + 1) % 3];
          const auto s = pieces[f].sides[k];
          if (s >= 0 && ((x == a && y == b) || (x == b && y == a)))
            sides.push_back({origin[f][std::size_t(s)],
                             origin[f][(std::size_t(s) + 1) % 3]});
        }
      }
      for (const auto &side : sides)
        enter_source_edge(side[0], side[1]);
    };

    // a side lying on `(a, b)` takes the ids strictly inside it: the whole
    // side, or — on a source side a standing carve split — the part of it
    // a piece holds
    const auto split_edge = [&](const corner_t &a, const corner_t &b,
                                const tf::buffer<corner_t> &ids) {
      const bool source_side = a.tag >= 0 && b.tag >= 0;
      const auto pa = position(a);
      const auto ab = diff(position(b), pa);
      const auto along = [&](const corner_t &c) {
        return dot(diff(position(c), pa), ab);
      };
      tf::buffer<corner_t> inside;
      const auto n = pieces.size();
      for (std::size_t f = 0; f < n; ++f) {
        if (dead[f])
          continue;
        for (std::size_t k = 0; k < 3; ++k) {
          const auto x = pieces[f].corners[k];
          const auto y = pieces[f].corners[(k + 1) % 3];
          const auto along_side = pieces[f].sides[k];
          inside.clear();
          if ((x == a && y == b) || (x == b && y == a)) {
            if (x == a)
              for (std::size_t i = 0; i < ids.size(); ++i)
                inside.push_back(ids[i]);
            else
              for (std::size_t i = ids.size(); i-- > 0;)
                inside.push_back(ids[i]);
          } else if (source_side && along_side >= 0) {
            const auto u = origin[f][std::size_t(along_side)];
            const auto v = origin[f][(std::size_t(along_side) + 1) % 3];
            if (!((u == a.id && v == b.id) || (u == b.id && v == a.id)))
              continue;
            const auto lo = along(x);
            const auto hi = along(y);
            if (lo < hi) {
              for (std::size_t i = 0; i < ids.size(); ++i) {
                const auto t = along(ids[i]);
                if (lo < t && t < hi)
                  inside.push_back(ids[i]);
              }
            } else
              for (std::size_t i = ids.size(); i-- > 0;) {
                const auto t = along(ids[i]);
                if (hi < t && t < lo)
                  inside.push_back(ids[i]);
              }
            if (inside.size() == 0)
              continue;
          } else
            continue;
          const auto z = pieces[f].corners[(k + 2) % 3];
          touched[f] = char(1);
          if (std::find(inside.begin(), inside.end(), z) != inside.end()) {
            dead[f] = char(1);
            break;
          }
          const auto into_x = pieces[f].sides[(k + 2) % 3];
          const auto out_of_y = pieces[f].sides[(k + 1) % 3];
          tf::buffer<corner_t> chain;
          chain.push_back(x);
          for (const auto &id : inside)
            chain.push_back(id);
          chain.push_back(y);
          const auto from = pieces[f].source;
          const auto from_lineage = lineage[f];
          const auto from_origin = origin[f];
          const auto last = chain.size() - 2;
          const auto piece_sides = [&](std::size_t c) {
            return std::array<std::int8_t, 3>{
                along_side, c == last ? out_of_y : std::int8_t(-1),
                c == 0 ? into_x : std::int8_t(-1)};
          };
          pieces[f].corners = {chain[0], chain[1], z};
          pieces[f].sides = piece_sides(0);
          for (std::size_t c = 1; c + 1 < chain.size(); ++c) {
            piece_t piece;
            piece.tag = tag;
            piece.source = from;
            piece.corners = {chain[c], chain[c + 1], z};
            piece.sides = piece_sides(c);
            push_piece(piece, from_lineage, char(1), from_origin);
          }
          break;
        }
      }
    };

    for (const auto &round : rounds) {
      // a face target names a face as the round found it
      for (std::size_t f = 0; f < pieces.size(); ++f)
        lineage[f] = sorted(pieces[f].corners);

      tf::buffer<typename carve_round<Index, Int>::edge_t> edges;
      for (const auto &e : round.edges)
        if (e.tag == tag) {
          auto edge = e;
          if (edge.b < edge.a)
            std::swap(edge.a, edge.b);
          edges.push_back(edge);
        }
      std::stable_sort(edges.begin(), edges.end(),
                       [](const auto &x, const auto &y) {
                         return std::tie(x.a, x.b) < std::tie(y.a, y.b);
                       });
      for (std::size_t i = 0; i < edges.size();) {
        std::size_t j = i;
        while (j < edges.size() && edges[j].a == edges[i].a &&
               edges[j].b == edges[i].b)
          ++j;
        const auto ca = edges[i].a;
        const auto cb = edges[i].b;
        for (std::size_t k = i; k < j; ++k)
          enter(edges[k].source);
        enter_edge(ca, cb);
        const auto pa = position(ca);
        const auto along = diff(position(cb), pa);
        struct along_point_t {
          T2 along;
          pt point;
        };
        tf::buffer<along_point_t> points;
        for (std::size_t k = i; k < j; ++k)
          points.push_back({dot(diff(edges[k].point, pa), along),
                            edges[k].point});
        std::sort(points.begin(), points.end(),
                  [&](const along_point_t &x, const along_point_t &y) {
                    return std::make_pair(x.along, key(x.point)) <
                           std::make_pair(y.along, key(y.point));
                  });
        tf::buffer<corner_t> ids;
        for (const auto &p : points) {
          const auto id = seat_of(p.point);
          if (ids.size() == 0 || !(ids[ids.size() - 1] == id))
            ids.push_back(id);
        }
        split_edge(ca, cb, ids);
        i = j;
      }

      for (const auto &target : round.faces) {
        if (target.tag != tag)
          continue;
        enter(target.source);
        const auto named = sorted(target.corners);
        const auto own = source_corners(target.source);
        const bool names_source =
            named == sorted({corner_t{std::int16_t(tag), own[0]},
                             corner_t{std::int16_t(tag), own[1]},
                             corner_t{std::int16_t(tag), own[2]}});
        bool seated = false;
        for (std::size_t f = 0; f < pieces.size() && !seated; ++f) {
          if (dead[f] ||
              !(lineage[f] == named ||
                (names_source && pieces[f].source == target.source)))
            continue;
          const auto c = pieces[f].corners;
          const std::array<pt, 3> p{position(c[0]), position(c[1]),
                                    position(c[2])};
          if (p[0] == target.point || p[1] == target.point ||
              p[2] == target.point) {
            seated = true;
            continue;
          }
          const auto normal = cross(diff(p[1], p[0]), diff(p[2], p[0]));
          std::array<int, 3> side{};
          for (std::size_t k = 0; k < 3; ++k)
            side[k] = sign3(
                cross(diff(p[(k + 1) % 3], p[k]), diff(target.point, p[k])),
                normal);
          const int zeros =
              int(side[0] == 0) + int(side[1] == 0) + int(side[2] == 0);
          const bool positive = side[0] >= 0 && side[1] >= 0 && side[2] >= 0;
          const bool negative = side[0] <= 0 && side[1] <= 0 && side[2] <= 0;
          if (!(positive || negative) || zeros >= 2)
            continue;
          const auto id = seat_of(target.point);
          if (zeros == 0) {
            const auto e = pieces[f].sides;
            const auto from = pieces[f].source;
            const auto from_lineage = lineage[f];
            const auto from_origin = origin[f];
            touched[f] = char(1);
            pieces[f].corners = {c[0], c[1], id};
            pieces[f].sides = {e[0], -1, -1};
            for (std::size_t k = 1; k < 3; ++k) {
              piece_t piece;
              piece.tag = tag;
              piece.source = from;
              piece.corners = {c[k], c[(k + 1) % 3], id};
              piece.sides = {e[k], -1, -1};
              push_piece(piece, from_lineage, char(1), from_origin);
            }
          } else {
            const std::size_t k = side[0] == 0 ? 0 : side[1] == 0 ? 1 : 2;
            tf::buffer<corner_t> ids;
            ids.push_back(id);
            enter_edge(c[k], c[(k + 1) % 3]);
            split_edge(c[k], c[(k + 1) % 3], ids);
          }
          seated = true;
        }
      }
    }

    // a source face is carved when any piece of it was cut; an untouched
    // work-list face keeps its own shape and is not restated
    const auto carved_begin = std::ptrdiff_t(out.carved.size());
    for (std::size_t f = 0; f < pieces.size(); ++f)
      if (touched[f])
        out.carved.push_back({tag, pieces[f].source});
    std::sort(out.carved.begin() + carved_begin, out.carved.end());
    out.carved.erase_till_end(
        std::unique(out.carved.begin() + carved_begin, out.carved.end()));
    for (std::size_t f = 0; f < pieces.size(); ++f)
      if (!dead[f] && std::binary_search(
                          out.carved.begin() + carved_begin, out.carved.end(),
                          std::array<Index, 2>{tag, pieces[f].source}))
        out.pieces.push_back(pieces[f]);
  }
  std::sort(out.carved.begin(), out.carved.end());
  std::stable_sort(out.pieces.begin(), out.pieces.end(),
                   [](const piece_t &x, const piece_t &y) {
                     return std::tie(x.tag, x.source) <
                            std::tie(y.tag, y.source);
                   });
  return out;
}

} // namespace tf::arrangement::reseat
