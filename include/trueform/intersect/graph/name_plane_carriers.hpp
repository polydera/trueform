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
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/views/enumerate.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../exact/canonical_plane.hpp"
#include "../../exact/meta.hpp"
#include "./face_descriptor.hpp"
#include "./plane_face_support.hpp"
#include "tbb/parallel_sort.h"

#include <cstddef>
#include <cstdint>

namespace tf::intersect::graph {

/// The name of the plane face `object` of form `tag` stands on.
template <typename Index, typename Int, typename ApplyToForm, typename GetPoint>
auto name_plane_face(Index tag, Index object, const ApplyToForm &apply_to_form,
                     const GetPoint &get_point)
    -> tf::exact::canonical_plane<Int> {
  tf::exact::canonical_plane<Int> name{};
  apply_to_form(tag, [&](const auto &form) {
    const auto corners = form.faces()[object];
    name = tf::exact::make_canonical_plane<Int>(
        plane_face_support<Int>(corners, [&](std::size_t corner) {
          return get_point(std::int16_t(tag), Index(corners[corner]));
        }));
  });
  return name;
}

/// A name no plane can take: a face whose corners are collinear.
template <typename Int>
auto is_plane_name_void(const tf::exact::canonical_plane<Int> &name) -> bool {
  using T2 = typename tf::exact::meta<Int>::T2;
  return name[0] == T2(0) && name[1] == T2(0) && name[2] == T2(0);
}

/// THE PLANE IDENTITY: every cut face names the plane it stands on, and
/// equal NAMES are one carrier.
///
/// This is the whole of it. Coplanarity is not closed from pairs here and
/// never was closable from pairs: a pair relation that is not transitive
/// grows classes without bound under a union-find and the answer depends
/// on which pairs were compared. A name is a value, equality of values is
/// transitive, and a plain sort gathers one plane's carriers into one run.
///
/// A face whose corners are collinear has no plane and names the zero
/// quadruple, which no plane can take — so it is not equal to anything,
/// not even to another carrier wearing it, and each such carrier keeps
/// an identity of its own. Sharing one would pool the edge blocks of
/// unrelated collapsed faces into a single constraint set.
///
/// THE NAME IS ONLY A NAME. It settles WHICH carriers are one plane and
/// nothing else: the carrier's frame, projection and orientation are
/// read off a member that stands on the EXACT plane, so no coordinate in
/// the pipeline is ever placed by a quantized descriptor.
///
/// THE NUMBERING IS THE GROUP ORDER: a run is group-ascending, so its
/// first entry is its smallest member, and the planes take their ids in
/// ascending order of that member. The identity is therefore independent
/// of how the names happen to sort.
///
/// `plane_names[p]` is plane `p`'s name, so a face that reaches the cut
/// world later is seated by the same law.
template <typename Index, typename Int, typename ApplyToForm, typename GetPoint>
auto name_plane_carriers(
    const tf::buffer<face_descriptor<Index>> &descriptors,
    const ApplyToForm &apply_to_form, const GetPoint &get_point,
    tf::buffer<Index> &plane_of, Index &n_planes,
    tf::buffer<tf::exact::canonical_plane<Int>> &plane_names) -> void {
  struct entry_t {
    tf::exact::canonical_plane<Int> name;
    Index group;
  };
  tf::buffer<entry_t> entries;
  entries.allocate(descriptors.size());
  tf::parallel_for_each(
      tf::make_sequence_range(Index(descriptors.size())),
      [&](Index g) {
        const auto &d = descriptors[std::size_t(g)];
        entries[std::size_t(g)] = {
            name_plane_face<Index, Int>(Index(d.tag), d.object, apply_to_form,
                                        get_point),
            g};
      },
      // naming a plane is an exact reduction, orders above the threading
      // overhead, so only a handful of faces are worth keeping serial
      tf::checked(64));
  tbb::parallel_sort(entries.begin(), entries.end(),
                     [](const entry_t &a, const entry_t &b) {
                       if (a.name != b.name)
                         return a.name < b.name;
                       return a.group < b.group;
                     });
  tf::buffer<Index> run_offsets;
  run_offsets.reserve(entries.size() + 1);
  tf::compute_offsets(entries, std::back_inserter(run_offsets), Index(0),
                      [](const entry_t &x, const entry_t &y) {
                        return !is_plane_name_void<Int>(x.name) &&
                               x.name == y.name;
                      });
  n_planes = Index(run_offsets.size() - 1);
  struct run_t {
    Index first_group, run;
  };
  tf::buffer<run_t> runs;
  runs.allocate(std::size_t(n_planes));
  tf::parallel_for_each(
      tf::make_sequence_range(n_planes),
      [&](Index run) {
        runs[std::size_t(run)] = {
            entries[std::size_t(run_offsets[std::size_t(run)])].group, run};
      },
      tf::checked);
  tbb::parallel_sort(runs.begin(), runs.end(),
                     [](const run_t &a, const run_t &b) {
                       return a.first_group < b.first_group;
                     });
  plane_of.allocate(descriptors.size());
  plane_names.allocate(std::size_t(n_planes));
  tf::parallel_for_each(
      tf::enumerate(runs),
      [&](auto pair) {
        auto &&[plane, run] = pair;
        const auto begin = run_offsets[std::size_t(run.run)];
        const auto end = run_offsets[std::size_t(run.run) + 1];
        plane_names[std::size_t(plane)] = entries[std::size_t(begin)].name;
        for (auto at = begin; at != end; ++at)
          plane_of[std::size_t(entries[std::size_t(at)].group)] = Index(plane);
      },
      tf::checked);
}

} // namespace tf::intersect::graph
