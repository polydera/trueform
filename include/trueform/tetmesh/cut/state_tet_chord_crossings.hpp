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
#include "../../core/config/verify.hpp"
#include "../../exact/coplanar_edge_edge_point.hpp"
#include "../../exact/meta.hpp"
#include "../../exact/orient2d.hpp"
#include "../../exact/vertex.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <tuple>

namespace tf::tetmesh {

/// A chord of a cut face: the segment between the two points `a < b` its
/// contact with the face `partner` states, on the face `face`; `edge` the
/// edge of the face both its ends stand on, where it cuts nothing and a
/// point splitting it lands on that edge, `-1` otherwise.
template <typename Index> struct tet_cut_chord {
  Index face;
  Index partner;
  Index a;
  Index b;
  Index edge;
};

/// A chord crossing: the point `point` the planes of the input faces `key`,
/// ascending, meet at.
template <typename Index> struct tet_chord_crossing {
  std::array<Index, 3> key;
  Index point;
};

/// A chord crossing a pass found under a key no earlier pass named: the key,
/// where the pass found it — `{first chord, second chord, piece of the
/// first, piece of the second}` — and the point its chords meet at.
template <typename Index, typename Int> struct tet_chord_proposal {
  std::array<Index, 3> key;
  std::array<Index, 4> order;
  tf::exact::pt3<Int> point;
};

/// One piece of a chord in its face's projection: its box, its chord and
/// its place along the chord.
template <typename Index, typename Int> struct tet_chord_piece_box {
  std::array<Int, 4> box;
  Index chord;
  Index piece;
};

/// The scratch @ref state_tet_chord_crossings walks on.
template <typename Index, typename Int> struct tet_chord_crossing_workspace {
  tf::buffer<std::array<Index, 3>> stated;
  tf::buffer<tet_chord_proposal<Index, Int>> pending;
  tf::buffer<std::array<Index, 3>> pending_stated;
  tf::buffer<Index> pending_id;
  tf::buffer<std::array<Index, 2>> winners;
  tf::buffer<Index> run;
  tf::buffer<Index> run_offsets;
  tf::buffer<Index> run_ids;
  tf::buffer<tet_chord_piece_box<Index, Int>> boxes;
  tf::buffer<Index> open;
  tf::buffer<char> dirty;
  tf::buffer<std::array<Index, 2>> split;
};

/// The points along the chord `a`, `b` into `run`: its ends and every point
/// `chord_points` (sorted `{a, b, point}`) puts on it, ordered from `a`.
template <typename Int, typename Index, typename Point, typename Out>
auto state_tet_chord_run(Index a, Index b,
                         const tf::buffer<std::array<Index, 3>> &chord_points,
                         const Point &point, Out &run) -> void {
  using T1 = typename tf::exact::meta<Int>::T1;
  using T2 = typename tf::exact::meta<Int>::T2;
  run.clear();
  run.push_back(a);
  for (auto at = std::lower_bound(chord_points.begin(), chord_points.end(),
                                  std::array<Index, 3>{a, b, Index(-1)});
       at != chord_points.end() && (*at)[0] == a && (*at)[1] == b; ++at)
    run.push_back((*at)[2]);
  run.push_back(b);
  const auto origin = point(a), end = point(b);
  const auto along = [&](Index p) {
    const auto q = point(p);
    T2 sum(0);
    for (std::size_t d = 0; d < 3; ++d)
      sum = sum + T2(T1(q[d]) - T1(origin[d])) * T2(T1(end[d]) - T1(origin[d]));
    return sum;
  };
  std::sort(run.begin() + 1, run.end() - 1, [&](Index x, Index y) {
    const auto ax = along(x), ay = along(y);
    return ax < ay || (!(ay < ax) && x < y);
  });
}

/// State the crossings of the chords of every cut face, `chords` sorted by
/// face, a face's projection `axes_of(face)`: two chords of one face from
/// partners on two other input faces, neither along an edge of the face,
/// that cross properly in the projection meet where the planes of the three
/// input faces `parent_of` names meet, and that point is one identity,
/// keyed by the three faces ascending in `crossings`. A crossing a pass
/// finds under a key no earlier pass named is proposed: the first of each
/// key, in the order chords and their pieces stand, computes the point
/// (@ref tf::exact::coplanar_edge_edge_point, on its first chord) and
/// `propose(p)` names it, in that order, every other crossing of the key
/// reading the name. Every chord holding a crossing is split there, by
/// chord, into `chord_points` (sorted `{a, b, point}`), so the faces sharing
/// a chord split it alike; a point of one of two chords that stands exactly
/// on the other, inside it, splits it there. A pass states each chord's run
/// once and meets the pieces of one face by one sweep along its first
/// projected axis; a later pass tests only pairs a chord of which the last
/// pass split, a pair of unsplit chords stating nothing new, and the passes
/// end when none splits. O(Σ faces' pieces log pieces + the piece pairs
/// whose boxes meet) per pass.
template <typename Index, typename Int, typename Point, typename Axes,
          typename Parent, typename Propose>
auto state_tet_chord_crossings(const tf::buffer<tet_cut_chord<Index>> &chords,
                               const Point &point, const Axes &axes_of,
                               const Parent &parent_of, const Propose &propose,
                               tf::buffer<std::array<Index, 3>> &chord_points,
                               tf::buffer<tet_chord_crossing<Index>> &crossings,
                               tet_chord_crossing_workspace<Index, Int> &work)
    -> void {
  using T1 = typename tf::exact::meta<Int>::T1;
  using T2 = typename tf::exact::meta<Int>::T2;
  chord_points.clear();
  crossings.clear();
  auto &stated = work.stated;
  auto &pending = work.pending;
  auto &pending_stated = work.pending_stated;
  auto &run_offsets = work.run_offsets;
  auto &run_ids = work.run_ids;
  auto &boxes = work.boxes;
  auto &open = work.open;
  auto &dirty = work.dirty;
  dirty.allocate(chords.size());
  std::fill(dirty.begin(), dirty.end(), char(1));
  const auto key_less = [](const tet_chord_crossing<Index> &c,
                           const std::array<Index, 3> &k) { return c.key < k; };
  for (;;) {
    stated.clear();
    pending.clear();
    pending_stated.clear();
    run_offsets.clear();
    run_ids.clear();
    run_offsets.push_back(Index(0));
    for (const auto &chord : chords) {
      if (chord.edge < 0) {
        state_tet_chord_run<Int>(chord.a, chord.b, chord_points, point,
                                 work.run);
        for (auto id : work.run)
          run_ids.push_back(id);
      }
      run_offsets.push_back(Index(run_ids.size()));
    }
    for (std::size_t c = 0; c < chords.size();) {
      std::size_t end = c;
      while (end < chords.size() && chords[end].face == chords[c].face)
        ++end;
      const Index face = chords[c].face;
      const auto plane = axes_of(face);
      const auto flat = [&plane](const tf::exact::pt3<Int> &p) {
        return tf::exact::pt2<Int>{p[plane.first], p[plane.second]};
      };
      boxes.clear();
      for (std::size_t i = c; i < end; ++i)
        for (Index k = run_offsets[i]; k + 1 < run_offsets[i + 1]; ++k) {
          const auto p = flat(point(run_ids[std::size_t(k)]));
          const auto q = flat(point(run_ids[std::size_t(k) + 1]));
          boxes.push_back({{std::min(p[0], q[0]), std::max(p[0], q[0]),
                            std::min(p[1], q[1]), std::max(p[1], q[1])},
                           Index(i),
                           Index(k - run_offsets[i])});
        }
      std::sort(boxes.begin(), boxes.end(), [](const auto &u, const auto &v) {
        return std::tie(u.box[0], u.chord, u.piece) <
               std::tie(v.box[0], v.chord, v.piece);
      });
      open.clear();
      const Index pf = parent_of(face);
      for (std::size_t s = 0; s < boxes.size(); ++s) {
        const auto &e = boxes[s];
        std::size_t w = 0;
        for (std::size_t k = 0; k < open.size(); ++k) {
          const auto &o = boxes[std::size_t(open[k])];
          if (o.box[1] < e.box[0])
            continue;
          open[w++] = open[k];
          if (o.chord == e.chord || o.box[3] < e.box[2] || e.box[3] < o.box[2])
            continue;
          const bool forward = o.chord < e.chord;
          const auto i = std::size_t(forward ? o.chord : e.chord);
          const auto j = std::size_t(forward ? e.chord : o.chord);
          const Index u = forward ? o.piece : e.piece;
          const Index v = forward ? e.piece : o.piece;
          const auto &x = chords[i], &y = chords[j];
          const Index px = parent_of(x.partner), py = parent_of(y.partner);
          if (px == py || px == pf || py == pf || (!dirty[i] && !dirty[j]))
            continue;
          std::array<Index, 3> key{pf, px, py};
          std::sort(key.begin(), key.end());
          const Index a0 = run_ids[std::size_t(run_offsets[i] + u)];
          const Index a1 = run_ids[std::size_t(run_offsets[i] + u) + 1];
          const Index b0 = run_ids[std::size_t(run_offsets[j] + v)];
          const Index b1 = run_ids[std::size_t(run_offsets[j] + v) + 1];
          if (a0 == b0 || a0 == b1 || a1 == b0 || a1 == b1)
            continue;
          const auto pa0 = point(a0), pa1 = point(a1), pb0 = point(b0),
                     pb1 = point(b1);
          const int s0 =
              tf::exact::orient2d_sign(flat(pb0), flat(pb1), flat(pa0));
          const int s1 =
              tf::exact::orient2d_sign(flat(pb0), flat(pb1), flat(pa1));
          const int s2 =
              tf::exact::orient2d_sign(flat(pa0), flat(pa1), flat(pb0));
          const int s3 =
              tf::exact::orient2d_sign(flat(pa0), flat(pa1), flat(pb1));
          if (s0 * s1 >= 0 || s2 * s3 >= 0) {
            const auto within = [&flat](const auto &p, const auto &q0,
                                        const auto &q1) {
              const auto fp = flat(p), f0 = flat(q0), f1 = flat(q1);
              T2 along(0), back(0);
              for (std::size_t d = 0; d < 2; ++d) {
                along = along +
                        T2(T1(fp[d]) - T1(f0[d])) * T2(T1(f1[d]) - T1(f0[d]));
                back = back +
                       T2(T1(fp[d]) - T1(f1[d])) * T2(T1(f0[d]) - T1(f1[d]));
              }
              return along > T2(0) && back > T2(0);
            };
            if (!s0 && within(pa0, pb0, pb1))
              stated.push_back({y.a, y.b, a0});
            if (!s1 && within(pa1, pb0, pb1))
              stated.push_back({y.a, y.b, a1});
            if (!s2 && within(pb0, pa0, pa1))
              stated.push_back({x.a, x.b, b0});
            if (!s3 && within(pb1, pa0, pa1))
              stated.push_back({x.a, x.b, b1});
            continue;
          }
          const auto at = std::lower_bound(crossings.begin(), crossings.end(),
                                           key, key_less);
          if (at != crossings.end() && at->key == key) {
            const Index p = at->point;
            if (p != x.a && p != x.b)
              stated.push_back({x.a, x.b, p});
            if (p != y.a && p != y.b)
              stated.push_back({y.a, y.b, p});
            continue;
          }
          const Index slot = Index(pending.size());
          pending.push_back(tet_chord_proposal<Index, Int>{
              key,
              {Index(i), Index(j), u, v},
              tf::exact::coplanar_edge_edge_point(pa0, pa1, pb0, pb1,
                                                  plane.first, plane.second)});
          pending_stated.push_back({x.a, x.b, slot});
          pending_stated.push_back({y.a, y.b, slot});
        }
        open.erase_till_end(open.begin() + std::ptrdiff_t(w));
        open.push_back(Index(s));
      }
      c = end;
    }
    if (pending.size()) {
      // the first crossing of each key in the order chords and pieces stand
      // computes its point, and the names are proposed in that order
      auto &pending_id = work.pending_id;
      auto &winners = work.winners;
      pending_id.allocate(pending.size());
      winners.clear();
      for (std::size_t k = 0; k < pending.size(); ++k)
        winners.push_back({Index(k), Index(0)});
      std::sort(winners.begin(), winners.end(),
                [&pending](const std::array<Index, 2> &u,
                           const std::array<Index, 2> &v) {
                  const auto &pu = pending[std::size_t(u[0])];
                  const auto &pv = pending[std::size_t(v[0])];
                  return std::tie(pu.key, pu.order) <
                         std::tie(pv.key, pv.order);
                });
      std::size_t n_winners = 0;
      for (std::size_t r = 0; r < winners.size();) {
        std::size_t end = r + 1;
        while (end < winners.size() &&
               pending[std::size_t(winners[end][0])].key ==
                   pending[std::size_t(winners[r][0])].key)
          ++end;
        for (std::size_t k = r; k < end; ++k)
          pending_id[std::size_t(winners[k][0])] = Index(n_winners);
        winners[n_winners] = {winners[r][0], Index(n_winners)};
        ++n_winners;
        r = end;
      }
      winners.erase_till_end(winners.begin() + std::ptrdiff_t(n_winners));
      std::sort(winners.begin(), winners.end(),
                [&pending](const std::array<Index, 2> &u,
                           const std::array<Index, 2> &v) {
                  return pending[std::size_t(u[0])].order <
                         pending[std::size_t(v[0])].order;
                });
      // `run` holds each key's name, by the key's rank
      work.run.allocate(n_winners);
      const std::size_t known = crossings.size();
      for (const auto &winner : winners) {
        const auto &proposal = pending[std::size_t(winner[0])];
        const Index p = propose(proposal.point);
        work.run[std::size_t(winner[1])] = p;
        crossings.push_back(tet_chord_crossing<Index>{proposal.key, p});
      }
      std::sort(
          crossings.begin() + std::ptrdiff_t(known), crossings.end(),
          [](const tet_chord_crossing<Index> &u,
             const tet_chord_crossing<Index> &v) { return u.key < v.key; });
      std::inplace_merge(
          crossings.begin(), crossings.begin() + std::ptrdiff_t(known),
          crossings.end(),
          [](const tet_chord_crossing<Index> &u,
             const tet_chord_crossing<Index> &v) { return u.key < v.key; });
      for (const auto &entry : pending_stated) {
        const Index p =
            work.run[std::size_t(pending_id[std::size_t(entry[2])])];
        if (p != entry[0] && p != entry[1])
          stated.push_back({entry[0], entry[1], p});
      }
    }
    std::sort(stated.begin(), stated.end());
    stated.erase_till_end(std::unique(stated.begin(), stated.end()));
    // the chords this pass split are the ones the next one tests
    auto &split = work.split;
    split.clear();
    for (const auto &entry : stated)
      if (!std::binary_search(chord_points.begin(), chord_points.end(), entry))
        split.push_back({entry[0], entry[1]});
    if (!split.size())
      break;
    split.erase_till_end(std::unique(split.begin(), split.end()));
    const std::size_t known = chord_points.size();
    for (const auto &entry : stated)
      chord_points.push_back(entry);
    std::sort(chord_points.begin() + std::ptrdiff_t(known), chord_points.end());
    std::inplace_merge(chord_points.begin(),
                       chord_points.begin() + std::ptrdiff_t(known),
                       chord_points.end());
    chord_points.erase_till_end(
        std::unique(chord_points.begin(), chord_points.end()));
    for (std::size_t i = 0; i < chords.size(); ++i)
      dirty[i] = char(
          std::binary_search(split.begin(), split.end(),
                             std::array<Index, 2>{chords[i].a, chords[i].b}));
  }
  if constexpr (tf::core::verification)
    for (std::size_t i = 1; i < crossings.size(); ++i)
      tf::core::verify(crossings[i - 1].key < crossings[i].key,
                       "every chord crossing's key is minted once");
}

} // namespace tf::tetmesh
