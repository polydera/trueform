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
#include "../../topology/topo_type.hpp"
#include "./reseat_pieces.hpp"
#include "./reseat_targets.hpp"
#include "tbb/parallel_sort.h"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <utility>

namespace tf::arrangement::reseat {

/// The carriers the rewrite `rows` leave off their own identity. Every row
/// fuses a name into a surviving identity standing at one lattice point.
/// Each created name it touches lies, by its own records, on an edge and in
/// a face of the operands; a carrier that does not hold the survivor's
/// point exactly is re-seated through it — the edge split there, the face
/// fanned there. An original vertex names no record, and its own faces hold
/// it; a survivor `created` does not name yet holds nothing to re-seat.
///
/// THE SEAT LAW: a seat within `quantum` of a corner of any carrier it names
/// is not a new feature but one the input cannot tell from that corner, and
/// re-seating through it would plant a sub-quantum piece. Such a seat is
/// refused whole — every carrier it names keeps its own shape. A carrier
/// standing as pieces names their corners too.
template <typename Index, typename Int, typename World, typename Rows,
          typename Created, typename ApplyToForm, typename Reader>
auto state_reseat_targets(const World &world, const Rows &rows,
                          const Created &created,
                          const standing_pieces<Index> &standing,
                          const ApplyToForm &apply_to_form,
                          const Reader &reader, Int quantum)
    -> reseat_targets<Index, Int> {
  using T1 = typename tf::exact::meta<Int>::T1;
  using T2 = typename tf::exact::meta<Int>::T2;
  using vec1 = std::array<T1, 3>;
  using vec2 = std::array<T2, 3>;
  using targets_t = reseat_targets<Index, Int>;
  targets_t targets;
  if (rows.size() == 0)
    return targets;

  const auto records = world.intersections().flat_intersections();
  const Index n_vertex_points = world.n_vertex_points();
  tf::buffer<std::array<Index, 2>> by_name;
  by_name.allocate(records.size());
  for (std::size_t r = 0; r < records.size(); ++r)
    by_name[r] = {Index(records[r].id), Index(r)};
  tbb::parallel_sort(by_name.begin(), by_name.end());

  const auto diff = [](const tf::point<Int, 3> &a, const tf::point<Int, 3> &b) {
    return vec1{T1(a[0]) - T1(b[0]), T1(a[1]) - T1(b[1]), T1(a[2]) - T1(b[2])};
  };
  const auto cross = [](const vec1 &a, const vec1 &b) {
    return vec2{T2(a[1]) * T2(b[2]) - T2(a[2]) * T2(b[1]),
                T2(a[2]) * T2(b[0]) - T2(a[0]) * T2(b[2]),
                T2(a[0]) * T2(b[1]) - T2(a[1]) * T2(b[0])};
  };
  const auto zero = [](const vec2 &v) {
    return v[0] == T2(0) && v[1] == T2(0) && v[2] == T2(0);
  };
  // every carrier a seat names states its corners beside it, for the law
  tf::buffer<std::array<tf::point<Int, 3>, 2>> seat_corners;
  const auto corner_point = [&](const piece_corner<Index> &c) {
    return c.tag >= 0 ? tf::point<Int, 3>(reader(int(c.tag), c.id))
                      : tf::point<Int, 3>(created[std::size_t(c.id)]);
  };
  // the corners of the pieces a carrier stands as: every corner of a face,
  // the corners on its side of an edge
  const auto standing_corners = [&](Index tag, Index face, Index side,
                                    const tf::point<Int, 3> &survivor) {
    const auto range = standing.of_source(tag, face);
    for (auto p = range[0]; p < range[1]; ++p) {
      const auto &piece = standing.pieces[p];
      for (std::size_t k = 0; k < 3; ++k)
        if (side == Index(-1) || piece.sides[k] == std::int8_t(side)) {
          seat_corners.push_back({survivor, corner_point(piece.corners[k])});
          if (side != Index(-1))
            seat_corners.push_back(
                {survivor, corner_point(piece.corners[(k + 1) % 3])});
        }
    }
  };
  const auto carrier = [&](Index tag, Index face, tf::topo_type label,
                           Index local, const tf::point<Int, 3> &survivor) {
    std::array<Index, 3> corners{};
    apply_to_form(tag, [&](const auto &form) {
      const auto f = form.faces()[face];
      for (std::size_t k = 0; k < 3; ++k)
        corners[k] = Index(f[k]);
    });
    if (label == tf::topo_type::edge) {
      const Index a = corners[std::size_t(local)];
      const Index b = corners[(std::size_t(local) + 1) % 3];
      const auto pa = reader(int(tag), a);
      const auto pb = reader(int(tag), b);
      seat_corners.push_back({survivor, pa});
      seat_corners.push_back({survivor, pb});
      standing_corners(tag, face, local, survivor);
      if (survivor == pa || survivor == pb ||
          zero(cross(diff(pb, pa), diff(survivor, pa))))
        return;
      targets.edges.push_back(
          {tag, a < b ? a : b, a < b ? b : a, survivor, face});
    } else if (label == tf::topo_type::face) {
      const auto pa = reader(int(tag), corners[0]);
      for (std::size_t k = 0; k < 3; ++k)
        seat_corners.push_back({survivor, reader(int(tag), corners[k])});
      standing_corners(tag, face, Index(-1), survivor);
      const auto normal = cross(diff(reader(int(tag), corners[1]), pa),
                                diff(reader(int(tag), corners[2]), pa));
      const auto to = diff(survivor, pa);
      if (normal[0] * T2(to[0]) + normal[1] * T2(to[1]) +
              normal[2] * T2(to[2]) ==
          T2(0))
        return;
      targets.faces.push_back({tag, face, survivor});
    }
  };

  for (const auto &row : rows) {
    const Index target = row[2];
    if (target < Index(0) || std::size_t(target) >= created.size())
      continue;
    const tf::point<Int, 3> survivor = created[std::size_t(target)];
    const std::array<Index, 2> names{target,
                                     row[0] == Index(1) ? row[1] : Index(-1)};
    for (const Index name : names) {
      if (name < n_vertex_points)
        continue;
      auto at = std::lower_bound(by_name.begin(), by_name.end(),
                                 std::array<Index, 2>{name, Index(0)});
      for (; at != by_name.end() && (*at)[0] == name; ++at) {
        const auto &record = records[std::size_t((*at)[1])];
        carrier(Index(record.tag), record.object, record.target.label,
                record.target.id, survivor);
        carrier(Index(record.tag_other), record.object_other,
                record.target_other.label, record.target_other.id, survivor);
      }
    }
  }

  if (quantum > Int(0) && targets.size() != 0) {
    const auto point_less = [](const tf::point<Int, 3> &a,
                               const tf::point<Int, 3> &b) {
      return std::tie(a[0], a[1], a[2]) < std::tie(b[0], b[1], b[2]);
    };
    tf::buffer<tf::point<Int, 3>> refused;
    for (const auto &pair : seat_corners) {
      T1 distance(0);
      for (std::size_t k = 0; k < 3; ++k) {
        const T1 d = pair[0][k] < pair[1][k] ? T1(pair[1][k]) - T1(pair[0][k])
                                             : T1(pair[0][k]) - T1(pair[1][k]);
        distance = std::max(distance, d);
      }
      if (distance != T1(0) && distance <= T1(quantum))
        refused.push_back(pair[0]);
    }
    std::sort(refused.begin(), refused.end(), point_less);
    refused.erase_till_end(std::unique(refused.begin(), refused.end()));
    const auto is_refused = [&](const tf::point<Int, 3> &point) {
      return std::binary_search(refused.begin(), refused.end(), point,
                                point_less);
    };
    targets.edges.erase_till_end(
        std::remove_if(targets.edges.begin(), targets.edges.end(),
                       [&](const auto &e) { return is_refused(e.point); }));
    targets.faces.erase_till_end(
        std::remove_if(targets.faces.begin(), targets.faces.end(),
                       [&](const auto &f) { return is_refused(f.point); }));
  }

  const auto edge_key = [](const typename targets_t::edge_t &e) {
    return std::make_tuple(e.tag, e.a, e.b, e.point[0], e.point[1], e.point[2]);
  };
  const auto face_key = [](const typename targets_t::face_t &f) {
    return std::make_tuple(f.tag, f.face, f.point[0], f.point[1], f.point[2]);
  };
  std::sort(targets.edges.begin(), targets.edges.end(),
            [&](const auto &x, const auto &y) {
              return std::make_pair(edge_key(x), x.face) <
                     std::make_pair(edge_key(y), y.face);
            });
  targets.edges.erase_till_end(std::unique(targets.edges.begin(),
                                           targets.edges.end(),
                                           [&](const auto &x, const auto &y) {
                                             return edge_key(x) == edge_key(y);
                                           }));
  std::sort(
      targets.faces.begin(), targets.faces.end(),
      [&](const auto &x, const auto &y) { return face_key(x) < face_key(y); });
  targets.faces.erase_till_end(std::unique(targets.faces.begin(),
                                           targets.faces.end(),
                                           [&](const auto &x, const auto &y) {
                                             return face_key(x) == face_key(y);
                                           }));
  return targets;
}

} // namespace tf::arrangement::reseat
