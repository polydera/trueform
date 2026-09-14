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

#include "../../../core/algorithm/compute_offsets.hpp"
#include "../../../core/algorithm/generic_generate.hpp"
#include "../../../core/algorithm/parallel_fill.hpp"
#include "../../../core/algorithm/parallel_for_each.hpp"
#include "../../../core/buffer.hpp"
#include "../../../core/checked.hpp"
#include "../../../core/offset_block_buffer.hpp"
#include "../../../core/views/sequence_range.hpp"
#include "../../canonical_plane.hpp"

#include "tbb/parallel_sort.h"

#include <cstddef>
#include <iterator>
#include <utility>

namespace tf::exact::door::pool {

/// THE NAME REGISTRY: the distinct exact planes of the world in canonical
/// order, the name every face speaks, and the faces of every name.
///
/// One sort states all three. The faces that name a plane are sorted BY
/// THEIR PLANE, so the runs of that order are the names — the first face of
/// a run carries the name's own quadruple, and the run itself is the block
/// the support point and the support set are read from.
///
/// The record is the face id and the comparator reads its plane, because the
/// sorted ids ARE the product: a name's block of faces is what two later
/// passes walk.
///
/// A name's id is its position in canonical order, so equal normals occupy a
/// contiguous id run and every later canonical comparison is an integer one.
template <typename Index, typename Int>
auto build_name_registry(
    const tf::buffer<tf::exact::canonical_plane<Int>> &face_plane,
    tf::buffer<tf::exact::canonical_plane<Int>> &plane,
    tf::buffer<int> &face_name,
    tf::offset_block_buffer<int, Index> &name_faces) -> void {
  const auto zero = tf::exact::canonical_plane<Int>{};
  const auto n_faces = face_plane.size();

  face_name.allocate(n_faces);
  tf::parallel_fill(face_name, -1);

  tf::buffer<Index> named;
  tf::generic_generate(tf::make_sequence_range(n_faces), named,
                       [&face_plane, &zero](std::size_t f,
                                            tf::buffer<Index> &into) {
                         if (face_plane[f] != zero)
                           into.push_back(Index(f));
                       });

  tbb::parallel_sort(named.begin(), named.end(),
                     [&face_plane](Index a, Index b) {
                       const auto &left = face_plane[std::size_t(a)];
                       const auto &right = face_plane[std::size_t(b)];
                       return left != right ? left < right : a < b;
                     });

  auto &offsets = name_faces.offsets_buffer();
  offsets.clear();
  tf::compute_offsets(named, std::back_inserter(offsets), 0,
                      [&face_plane](Index a, Index b) {
                        return face_plane[std::size_t(a)] ==
                               face_plane[std::size_t(b)];
                      });
  if (offsets.size() == 0)
    offsets.push_back(0);

  const auto n_names = offsets.size() - 1;
  plane.allocate(n_names);
  tf::parallel_for_each(
      tf::make_sequence_range(n_names),
      [&plane, &face_plane, &face_name, &named, &offsets](std::size_t n) {
        plane[n] = face_plane[std::size_t(named[std::size_t(offsets[n])])];
        for (int k = offsets[n]; k < offsets[n + 1]; ++k)
          face_name[std::size_t(named[std::size_t(k)])] = int(n);
      },
      tf::checked);

  name_faces.data_buffer() = std::move(named);
}

} // namespace tf::exact::door::pool
