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
#include "../../core/config/verify.hpp"
#include "../../core/views/drop.hpp"
#include "../../exact/meta.hpp"
#include "../../topology/cdt/dt3/for_each_tet_site_within.hpp"
#include "../../topology/cdt/dt3/recovery/collapse_tet_site_rows.hpp"
#include "../../topology/cdt/dt3/recovery/gather_tet_star.hpp"
#include "../../topology/cdt/dt3/recovery/open_tet_stamps.hpp"
#include "../../topology/cdt/dt3/recovery/open_tet_star_stamps.hpp"
#include "../../topology/cdt/dt3/recovery/tet_crossing_pair.hpp"
#include "../../topology/cdt/dt3/recovery/tet_segment_recovery.hpp"
#include "../../topology/cdt/dt3/recovery/tet_stamps.hpp"
#include "../../topology/cdt/dt3/recovery/tet_star_stamps.hpp"
#include "../../topology/cdt/dt3/tet_owner_complex.hpp"
#include "../../topology/cdt/dt3/tet_site_position.hpp"
#include "../../topology/cdt/dt3/walk_straight_tet.hpp"
#include <algorithm>
#include <array>
#include <cstddef>

namespace tf::tetmesh {

/// The tables the corner conviction walks on: per slot the site a collapse
/// retired it into, `-1` while it stands — the recovery's own sites, which
/// no identity of the drain names, retiring nowhere else; per slot a site's
/// place in the cluster being collapsed (`0` none, `1` found, `2` reached), the
/// sites found, the cluster in the order it is reached from its corner, the
/// edges among it, a star, and the stamps of the reach and of the stars.
/// An edge of the complex inside a corner's cluster, by its exact squared
/// length on the lattice.
template <typename Index, typename Coord> struct tet_corner_census_edge {
  typename tf::exact::meta<Coord>::T2 length2;
  Index a;
  Index b;
};

template <typename Index, typename Coord> struct tet_corner_census_scratch {
  tf::buffer<Index> into;
  tf::buffer<char> place;
  tf::buffer<Index> found;
  tf::buffer<Index> order;
  tf::buffer<tet_corner_census_edge<Index, Coord>> edges;
  tf::buffer<Index> star;
  tf::buffer<Index> flood;
  tf::topology::cdt::dt3::tet_stamps stamps;
  tf::topology::cdt::dt3::tet_star_stamps star_stamps;
};

/// THE CENSUS'S CORNER CONVICTION: a face phase F refused with no
/// constraint to cut, `named` with `crosses` empty, witnesses the sites the
/// recovery minted in the cell of the segment recovery's reach grid one of
/// its corners stands in, an edge from which pierces it. Those sites are that
/// corner
/// (@ref tf::topology::cdt::dt3::tet_shares_reach_cell), and they retire into
/// it edge by edge (@ref tf::topology::cdt::dt3::collapse_tet_site_rows):
/// the cluster is reached from the corner through the sites' stars, and its
/// edges, shortest first, each collapse the two identities standing at
/// their ends into one — the corner where it is one of them, the election's
/// smallest slot otherwise — so a collapse moves a site as little as the
/// cluster allows. A collapse that refuses changes nothing, and the edge's
/// two identities stand apart. Appends to `due` every face a collapse moved,
/// and to `convicted` every face that witnessed; returns the sites collapsed
/// and the collapses refused. O(the cells meeting the witnesses' reach boxes +
/// the stars of their clusters + their edges, sorted, each a collapse).
template <typename Owner, typename Work, typename Faces, typename Coord>
auto collapse_tet_steiner_corners(
    Owner &owner, Work &work, const Faces &faces,
    const tf::buffer<
        tf::topology::cdt::dt3::tet_crossing_pair<typename Owner::index_type>>
        &named,
    tf::topology::cdt::dt3::tet_segment_recovery<typename Owner::index_type,
                                                 Coord> &segment,
    tet_corner_census_scratch<typename Owner::index_type, Coord> &scratch,
    tf::buffer<typename Owner::index_type> &due,
    tf::buffer<typename Owner::index_type> &convicted)
    -> std::array<std::size_t, 2> {
  namespace dt3 = tf::topology::cdt::dt3;
  using Index = typename Owner::index_type;
  std::size_t collapsed = 0, refusals = 0;
  auto &place = scratch.place;
  auto &found = scratch.found;
  auto &order = scratch.order;
  auto &edges = scratch.edges;
  const std::size_t known = place.size();
  if (dt3::tet_site_count(owner) > known) {
    place.reallocate(dt3::tet_site_count(owner));
    tf::parallel_fill(tf::drop(place, known), char(0));
  }
  const auto steiner = [&segment](Index x) {
    return std::size_t(x) < segment.steiner.size() &&
           segment.steiner[std::size_t(x)];
  };
  auto &into = scratch.into;
  if (into.size() < dt3::tet_site_count(owner)) {
    const std::size_t grown = into.size();
    into.reallocate(dt3::tet_site_count(owner));
    tf::parallel_fill(tf::drop(into, grown), Index(-1));
  }
  const auto standing = [&into](Index x) {
    [[maybe_unused]] std::size_t steps = 0;
    while (into[std::size_t(x)] != Index(-1)) {
      x = into[std::size_t(x)];
      if constexpr (tf::core::verification)
        tf::core::verify(++steps <= into.size(),
                         "the census's survivor chase is acyclic");
    }
    return x;
  };
  for (const auto &pair : named) {
    if (pair.face < 0 || pair.crosses.face >= 0 || pair.crosses.segment >= 0)
      continue;
    convicted.push_back(pair.face);
    for (const auto &triangle : faces[std::size_t(pair.face)])
      for (auto corner : triangle) {
        const Index c = Index(corner);
        found.clear();
        order.clear();
        edges.clear();
        {
          const dt3::tet_owner_complex<true, Owner, Work> complex(owner, work);
          dt3::open_tet_stamps(scratch.stamps, complex.tets().size());
          dt3::for_each_tet_site_within(
              complex, dt3::tet_site_position(owner, c),
              dt3::walk_straight_tet(owner, c, owner._hint), segment.reach,
              [&](Index x) {
                if (x != c && steiner(x) && !place[std::size_t(x)]) {
                  place[std::size_t(x)] = 1;
                  found.push_back(x);
                }
              },
              [&scratch](std::size_t k) {
                return dt3::take_tet_stamp(scratch.stamps, k);
              },
              scratch.flood);
          if (!found.size())
            continue;
          order.push_back(c);
          place[std::size_t(c)] = 2;
          dt3::open_tet_star_stamps(complex, scratch.star_stamps);
          for (std::size_t head = 0; head < order.size(); ++head) {
            const Index x = order[head];
            scratch.star.clear();
            dt3::gather_tet_star(complex, x,
                                 dt3::walk_straight_tet(owner, x, owner._hint),
                                 scratch.star_stamps, scratch.star);
            const auto px = dt3::tet_site_position(owner, x);
            for (auto at : scratch.star)
              for (auto y : complex.tets()[std::size_t(at)]) {
                if (!place[std::size_t(y)] || y == x)
                  continue;
                if (place[std::size_t(y)] == 1) {
                  place[std::size_t(y)] = 2;
                  order.push_back(y);
                }
                if (y < x)
                  continue;
                const auto py = dt3::tet_site_position(owner, y);
                using T1 = typename tf::exact::meta<Coord>::T1;
                using T2 = typename tf::exact::meta<Coord>::T2;
                T2 length2(0);
                for (std::size_t k = 0; k < 3; ++k) {
                  const T2 d = T2(T1(py.floor[k]) - T1(px.floor[k]));
                  length2 = length2 + d * d;
                }
                edges.push_back({length2, x, y});
              }
          }
        }
        for (auto x : found)
          place[std::size_t(x)] = 0;
        place[std::size_t(c)] = 0;
        std::sort(edges.begin(), edges.end(), [](const auto &e, const auto &f) {
          if (e.length2 != f.length2)
            return e.length2 < f.length2;
          return e.a != f.a ? e.a < f.a : e.b < f.b;
        });
        edges.erase_till_end(std::unique(edges.begin(), edges.end(),
                                         [](const auto &e, const auto &f) {
                                           return e.a == f.a && e.b == f.b;
                                         }));
        for (const auto &edge : edges) {
          const Index a = standing(edge.a), b = standing(edge.b);
          if (a == b)
            continue;
          const Index survivor = a == c || b == c ? c : std::min(a, b);
          const Index retired = survivor == a ? b : a;
          if (dt3::collapse_tet_site_rows(owner, segment, retired, survivor)) {
            into[std::size_t(retired)] = survivor;
            ++collapsed;
            for (auto g : owner._collapse.faces)
              due.push_back(g);
          } else
            ++refusals;
        }
      }
  }
  return {collapsed, refusals};
}

/// The site the recovery's site `slot` speaks after the census's collapses:
/// its own while it stands, the survivor it retired into otherwise.
template <typename Index, typename Coord>
auto tet_census_standing(const tet_corner_census_scratch<Index, Coord> &scratch,
                         Index slot) -> Index {
  [[maybe_unused]] std::size_t steps = 0;
  while (std::size_t(slot) < scratch.into.size() &&
         scratch.into[std::size_t(slot)] != Index(-1)) {
    slot = scratch.into[std::size_t(slot)];
    if constexpr (tf::core::verification)
      tf::core::verify(++steps <= scratch.into.size(),
                       "the census's survivor chase is acyclic");
  }
  return slot;
}

} // namespace tf::tetmesh
