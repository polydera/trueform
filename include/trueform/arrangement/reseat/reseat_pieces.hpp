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
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <tuple>

namespace tf::arrangement::reseat {

/// One corner of a carve: an original vertex `(tag, id)` with `tag >= 0`, a
/// seat `(seat_corner_tag, k)` — row `k` of the carve's seat points — or, once
/// stated to a world, the created identity `(-1, id)` the seat resolved to.
inline constexpr std::int16_t seat_corner_tag = -2;

template <typename Index> struct piece_corner {
  std::int16_t tag;
  Index id;

  friend auto operator==(const piece_corner &a, const piece_corner &b) -> bool {
    return a.tag == b.tag && a.id == b.id;
  }
  friend auto operator<(const piece_corner &a, const piece_corner &b) -> bool {
    return std::tie(a.tag, a.id) < std::tie(b.tag, b.id);
  }
};

/// One piece of a carved source face: its corners in the source winding and,
/// per side, the source side it keeps whole (`-1` for a side the carve made).
template <typename Index> struct reseat_piece {
  Index tag;
  Index source;
  std::array<piece_corner<Index>, 3> corners;
  std::array<std::int8_t, 3> sides;
};

/// The carve, in rounds: each round's targets name the faces and edges as
/// they stand after the rounds before it, by their corners, and the source
/// face a target was found on — the carrier its corners enter from.
template <typename Index, typename Int> struct carve_round {
  struct edge_t {
    Index tag;
    Index source;
    piece_corner<Index> a;
    piece_corner<Index> b;
    tf::point<Int, 3> point;
  };
  struct face_t {
    Index tag;
    Index source;
    std::array<piece_corner<Index>, 3> corners;
    tf::point<Int, 3> point;
  };
  tf::buffer<edge_t> edges;
  tf::buffer<face_t> faces;
};

/// The pieces the carves stated into a world that still stand there, in the
/// world's names: per piece its source face, the source side each side lies
/// on and its corners through the world's closed rewrite table, grouped by
/// `(tag, source)` ascending; per piece its world face and its `key`, the
/// face past its tag's own the world's records call it; and per tag the
/// pieces ever stated, so the next carve's names follow them. A source face
/// standing as pieces enters a carve as those pieces, and a carve that cuts
/// any of them restates them all.
template <typename Index> struct standing_pieces {
  tf::buffer<reseat_piece<Index>> pieces;
  tf::buffer<Index> faces;
  tf::buffer<Index> keys;
  /// `{tag, key, piece}` ascending
  tf::buffer<std::array<Index, 3>> by_key;
  tf::buffer<Index> stated;
  /// a corner named outside the standing currency: no carve reads them
  bool foreign = false;

  /// The pieces `source` stands as, `[first, last)`; empty when it stands
  /// whole.
  auto of_source(Index tag, Index source) const -> std::array<std::size_t, 2> {
    const auto less = [](const reseat_piece<Index> &p,
                         const std::array<Index, 2> &k) {
      return std::tie(p.tag, p.source) < std::tie(k[0], k[1]);
    };
    const std::array<Index, 2> key{tag, source};
    const auto first =
        std::lower_bound(pieces.begin(), pieces.end(), key, less);
    auto last = first;
    while (last != pieces.end() && last->tag == tag && last->source == source)
      ++last;
    return {std::size_t(first - pieces.begin()),
            std::size_t(last - pieces.begin())};
  }
  /// The piece a key names, `-1` for a source face.
  auto of_key(Index tag, Index key) const -> Index {
    const auto at = std::lower_bound(
        by_key.begin(), by_key.end(), std::array<Index, 3>{tag, key, Index(-1)});
    if (at == by_key.end() || (*at)[0] != tag || (*at)[1] != key)
      return Index(-1);
    return (*at)[2];
  }
  /// The source face a key lies in: a piece's source, a source face itself.
  auto source_of_key(Index tag, Index key) const -> Index {
    const auto piece = of_key(tag, key);
    return piece == Index(-1) ? key : pieces[std::size_t(piece)].source;
  }
  auto stated_of(Index tag) const -> Index {
    return stated.size() == 0 ? Index(0) : stated[std::size_t(tag)];
  }
};

/// Every piece of every carved source face, grouped by source face, the
/// carved faces as `(tag, face)` ascending, and the seat points.
template <typename Index, typename Int> struct reseat_pieces {
  tf::buffer<reseat_piece<Index>> pieces;
  tf::buffer<std::array<Index, 2>> carved;
  tf::buffer<tf::point<Int, 3>> seats;
};

} // namespace tf::arrangement::reseat
