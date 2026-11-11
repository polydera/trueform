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
#include "./state_tet_cut_contacts.hpp"
#include "./tet_slot_holders.hpp"
#include <algorithm>
#include <array>
#include <cstddef>

namespace tf::tetmesh {

/// The lanes of a drain, every one due: the traced triangles `cut` joined
/// by their coplanar contacts into groups, lanes `0 .. n_groups - 1` in the
/// order of their smallest triangle, and every live triangle beyond them
/// holding an edge a point stands on (`edge_points`) a lane of its own, in
/// the order the edges first reach it. `unit_runs` blocks `unit_members` —
/// positions in `cut` for a group, triangles for one beyond — `unit_of_cut`
/// names each traced triangle's lane and `neighbour_units` each other
/// triangle's (sorted `{triangle, lane}`); `due` lists every lane. Returns
/// `n_groups`. O(cut + contacts + Σ holders of the edges points stand on,
/// each with one log).
template <typename Index, typename Cut, typename Live, typename Holds>
auto state_tet_lanes(const Cut &cut,
                     const tf::buffer<tet_cut_contact<Index>> &contacts,
                     const tf::buffer<std::array<Index, 3>> &edge_points,
                     const tet_slot_holders<Index> &holders, const Live &live,
                     const Holds &holds, tf::buffer<Index> &unit_runs,
                     tf::buffer<Index> &unit_members,
                     tf::buffer<Index> &unit_of_cut,
                     tf::buffer<std::array<Index, 2>> &neighbour_units,
                     tf::buffer<Index> &due, tf::buffer<Index> &group,
                     tf::buffer<std::array<Index, 2>> &grouped,
                     tf::buffer<std::array<Index, 2>> &touched) -> std::size_t {
  group.allocate(cut.size());
  for (std::size_t f = 0; f < cut.size(); ++f)
    group[f] = Index(f);
  const auto root = [&group](Index f) {
    while (group[std::size_t(f)] != f) {
      group[std::size_t(f)] = group[std::size_t(group[std::size_t(f)])];
      f = group[std::size_t(f)];
    }
    return f;
  };
  for (const auto &c : contacts)
    if (c.coplanar) {
      const Index a = root(c.face), b = root(c.partner);
      if (a != b)
        group[std::size_t(std::max(a, b))] = std::min(a, b);
    }
  grouped.clear();
  for (std::size_t f = 0; f < cut.size(); ++f)
    grouped.push_back({root(Index(f)), Index(f)});
  std::sort(grouped.begin(), grouped.end());
  unit_runs.clear();
  unit_members.clear();
  unit_runs.push_back(0);
  unit_of_cut.allocate(cut.size());
  for (std::size_t m = 0; m < grouped.size();) {
    std::size_t m_end = m;
    while (m_end < grouped.size() && grouped[m_end][0] == grouped[m][0])
      ++m_end;
    for (std::size_t i = m; i < m_end; ++i) {
      unit_of_cut[std::size_t(grouped[i][1])] = Index(unit_runs.size() - 1);
      unit_members.push_back(grouped[i][1]);
    }
    unit_runs.push_back(Index(unit_members.size()));
    m = m_end;
  }
  const std::size_t n_groups = unit_runs.size() - 1;
  due.clear();
  for (std::size_t g = 0; g < n_groups; ++g)
    due.push_back(Index(g));
  const auto in_cut = [&cut](Index u) {
    return std::binary_search(cut.begin(), cut.end(), u);
  };
  touched.clear();
  for (std::size_t e = 0; e < edge_points.size();) {
    std::size_t h = e;
    while (h < edge_points.size() && edge_points[h][0] == edge_points[e][0] &&
           edge_points[h][1] == edge_points[e][1])
      ++h;
    const Index b = edge_points[e][1];
    for_each_tet_slot_holder(holders, edge_points[e][0], [&](Index u) {
      if (live(std::size_t(u)) && holds(u, b) && !in_cut(u))
        touched.push_back({u, Index(touched.size())});
    });
    e = h;
  }
  std::sort(touched.begin(), touched.end());
  touched.erase_till_end(
      std::unique(touched.begin(), touched.end(),
                  [](const std::array<Index, 2> &x,
                     const std::array<Index, 2> &y) { return x[0] == y[0]; }));
  std::sort(touched.begin(), touched.end(),
            [](const std::array<Index, 2> &x, const std::array<Index, 2> &y) {
              return x[1] < y[1];
            });
  neighbour_units.clear();
  for (const auto &entry : touched) {
    const Index unit = Index(unit_runs.size() - 1);
    unit_members.push_back(entry[0]);
    unit_runs.push_back(Index(unit_members.size()));
    neighbour_units.push_back({entry[0], unit});
    due.push_back(unit);
  }
  std::sort(neighbour_units.begin(), neighbour_units.end());
  return n_groups;
}

} // namespace tf::tetmesh
