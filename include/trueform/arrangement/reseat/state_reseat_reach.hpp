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
#include "./reseat_pieces.hpp"
#include <algorithm>
#include <array>
#include <cstddef>

namespace tf::arrangement::reseat {

/// The world faces a carve reaches, as `(tag, face)` ascending: the carved
/// faces, the faces their records name, every member of the planes either
/// stands on (all RESTATED), and the faces the restated ones' records name
/// (NEIGHBOURS, whose contacts complete the restated planes). A face is
/// named as it stands: a source face standing as pieces by its pieces'
/// keys, and none of the pieces of a face the carve cuts.
template <typename Index> struct reseat_reach {
  tf::buffer<std::array<Index, 2>> restated;
  tf::buffer<std::array<Index, 2>> neighbours;
};

/// The world face a face key is, `-1` when the world never named it: a
/// source face of the cut world by binary search over the flat keys, every
/// other face — an entrant, or a piece past its tag's own faces — by its
/// `(tag, object)`.
template <typename Index, typename World> struct world_face_lookup {
  const World *world;
  tf::buffer<std::array<Index, 3>> entrants;

  explicit world_face_lookup(const World &w) : world(&w) {
    for (Index face = w.n_cut_faces(); face < w.n_faces(); ++face) {
      const auto &d = w.descriptor(face);
      entrants.push_back({Index(d.tag), Index(d.object), face});
    }
    std::sort(entrants.begin(), entrants.end());
  }
  auto operator()(Index tag, Index object) const -> Index {
    const auto face_offsets = world->face_offsets();
    const auto n_own = face_offsets[std::size_t(tag) + 1] -
                       face_offsets[std::size_t(tag)];
    if (object < n_own) {
      const auto key = face_offsets[std::size_t(tag)] + object;
      const auto descriptors = world->graph().descriptors();
      const auto at = std::lower_bound(
          descriptors.begin(), descriptors.end(), key,
          [&](const auto &d, Index k) {
            return face_offsets[std::size_t(d.tag)] + Index(d.object) < k;
          });
      if (at != descriptors.end() &&
          face_offsets[std::size_t(at->tag)] + Index(at->object) == key)
        return Index(at - descriptors.begin());
    }
    const auto e =
        std::lower_bound(entrants.begin(), entrants.end(),
                         std::array<Index, 3>{tag, object, Index(-1)});
    if (e != entrants.end() && (*e)[0] == tag && (*e)[1] == object)
      return (*e)[2];
    return Index(-1);
  }
};

template <typename Index, typename World>
auto state_reseat_reach(const World &world,
                        const standing_pieces<Index> &standing,
                        const tf::buffer<std::array<Index, 2>> &carved,
                        const tf::buffer<std::array<Index, 2>> &forced,
                        const tf::buffer<std::array<Index, 2>> &offered)
    -> reseat_reach<Index> {
  using key_t = std::array<Index, 2>;
  const auto blocks = world.intersections().intersections();
  const auto contains = [](const tf::buffer<key_t> &set, const key_t &key) {
    return std::binary_search(set.begin(), set.end(), key);
  };
  const auto close = [](tf::buffer<key_t> &set) {
    std::sort(set.begin(), set.end());
    set.erase_till_end(std::unique(set.begin(), set.end()));
  };
  const world_face_lookup<Index, World> face_of(world);
  // a cut face's records are its own block of the record tier, and a
  // piece's are its source face's
  const auto partners_of = [&](const tf::buffer<key_t> &of,
                               tf::buffer<key_t> &out) {
    for (const auto &key : of) {
      const auto face =
          face_of(key[0], standing.source_of_key(key[0], key[1]));
      if (face == Index(-1) || face >= world.n_cut_faces())
        continue;
      for (const auto &record : blocks[std::size_t(face)])
        out.push_back({Index(record.tag_other), Index(record.object_other)});
    }
  };
  // a source face standing as pieces is named by its pieces
  const auto as_standing = [&](tf::buffer<key_t> &set) {
    const auto n = set.size();
    for (std::size_t k = 0; k < n; ++k) {
      const auto range = standing.of_source(set[k][0], set[k][1]);
      for (auto p = range[0]; p < range[1]; ++p)
        set.push_back({set[k][0], standing.keys[p]});
    }
    set.erase_till_end(
        std::remove_if(set.begin(), set.end(), [&](const key_t &key) {
          const auto range = standing.of_source(key[0], key[1]);
          return range[0] != range[1];
        }));
  };
  // a piece of a face the carve cuts is retired with it
  const auto cut_away = [&](const key_t &key) {
    return contains(carved, key) ||
           (standing.of_key(key[0], key[1]) != Index(-1) &&
            contains(carved,
                     {key[0], standing.source_of_key(key[0], key[1])}));
  };
  const auto members_of = [&](const tf::buffer<key_t> &of,
                              tf::buffer<key_t> &out) {
    tf::buffer<Index> planes;
    for (const auto &key : of) {
      const auto face = face_of(key[0], key[1]);
      if (face != Index(-1))
        planes.push_back(world.plane_of_face(face));
    }
    std::sort(planes.begin(), planes.end());
    planes.erase_till_end(std::unique(planes.begin(), planes.end()));
    for (const auto plane : planes)
      for (Index k = 0; k < world.member_count(plane); ++k) {
        const auto &d = world.descriptor(world.member(plane, k));
        out.push_back({Index(d.tag), Index(d.object)});
      }
  };

  reseat_reach<Index> reach;
  tf::buffer<key_t> ring;
  partners_of(carved, ring);
  for (const auto &key : carved)
    ring.push_back(key);
  for (const auto &key : forced)
    ring.push_back(key);
  close(ring);
  tf::buffer<key_t> standing_ring = ring;
  as_standing(standing_ring);
  members_of(standing_ring, reach.restated);
  for (const auto &key : standing_ring)
    reach.restated.push_back(key);
  close(reach.restated);
  reach.restated.erase_till_end(std::remove_if(
      reach.restated.begin(), reach.restated.end(), cut_away));
  tf::buffer<key_t> inner = reach.restated;
  for (const auto &key : carved)
    inner.push_back(key);
  close(inner);
  partners_of(inner, reach.neighbours);
  // a face a piece's box reaches is offered as a neighbour: the delta's own
  // records say whether a piece meets it
  for (const auto &key : offered)
    reach.neighbours.push_back(key);
  as_standing(reach.neighbours);
  close(reach.neighbours);
  reach.neighbours.erase_till_end(
      std::remove_if(reach.neighbours.begin(), reach.neighbours.end(),
                     [&](const key_t &key) {
                       return contains(inner, key) || cut_away(key);
                     }));
  return reach;
}

} // namespace tf::arrangement::reseat
