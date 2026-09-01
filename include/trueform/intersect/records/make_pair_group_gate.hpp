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
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/algorithm/parallel_transform.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/range.hpp"
#include "../../core/views/sequence_range.hpp"
#include "./pair_group_gate.hpp"
#include "tbb/parallel_sort.h"

#include <algorithm>
#include <array>
#include <cstddef>

namespace tf::intersect {

/// The gate's two dense block tables, keyed by the FLAT face — the tag's
/// base plus the object.
///
/// Both are facts a face's records all ask for, so asking either by
/// binary search would make the average record pay for the table's size.
/// Counts plus one prefix turn them into a block per face instead, and
/// the face domain is walked ONCE for both.
///
/// `tickets` are `{flat face, point}` pairs; canonicalizing them here is
/// what makes their point column one table's data, ascending within a
/// face. The coplanar pairs are already ordered, so both halves of the
/// other table scatter ascending.
///
/// A pole where nothing was worth gating states no tickets, and then the
/// delivered table is NOT BUILT — not even its dense spine. One fact
/// decides both ends of that: a face is marked because some fan is a
/// product, and a fan asks the gate only when it is one, so a gate with
/// no delivered blocks is a gate nothing asks.
template <typename Index>
auto make_pair_group_gate(tf::buffer<std::array<Index, 2>> &tickets,
                          const tf::buffer<std::array<Index, 4>> &coplanar_pairs,
                          const tf::buffer<Index> &face_offsets,
                          tf::buffer<Index> &delivered_offsets,
                          tf::buffer<Index> &delivered_points,
                          tf::buffer<Index> &coplanar_offsets,
                          tf::buffer<Index> &coplanar_partners)
    -> pair_group_gate<Index> {
  if (tickets.size() != 0) {
    // A small batch is sorted below the cost of entering the parallel
    // machinery.
    if (tickets.size() < 4096)
      std::sort(tickets.begin(), tickets.end());
    else
      tbb::parallel_sort(tickets.begin(), tickets.end());
    tickets.erase_till_end(std::unique(tickets.begin(), tickets.end()));
  }
  const auto flat = [&face_offsets](Index tag, Index object) {
    return std::size_t(face_offsets[std::size_t(tag)] + object);
  };
  const auto n_faces = std::size_t(face_offsets[face_offsets.size() - 1]);
  const bool delivered = tickets.size() != 0;
  coplanar_offsets.allocate(n_faces + 1);
  std::fill(coplanar_offsets.begin(), coplanar_offsets.end(), Index(0));
  for (const auto &pair : coplanar_pairs) {
    ++coplanar_offsets[flat(pair[0], pair[1]) + 1];
    ++coplanar_offsets[flat(pair[2], pair[3]) + 1];
  }
  if (delivered) {
    delivered_offsets.allocate(n_faces + 1);
    std::fill(delivered_offsets.begin(), delivered_offsets.end(), Index(0));
    // the tickets are sorted, so a run's head states the whole run's
    // length and every head is independent
    tf::parallel_for_each(
        tf::make_sequence_range(tickets.size()),
        [&](std::size_t k) {
          if (k != 0 && tickets[k][0] == tickets[k - 1][0])
            return;
          std::size_t end = k + 1;
          while (end < tickets.size() && tickets[end][0] == tickets[k][0])
            ++end;
          delivered_offsets[std::size_t(tickets[k][0]) + 1] = Index(end - k);
        },
        tf::checked);
    for (std::size_t f = 1; f <= n_faces; ++f) {
      delivered_offsets[f] += delivered_offsets[f - 1];
      coplanar_offsets[f] += coplanar_offsets[f - 1];
    }
    delivered_points.allocate(tickets.size());
    tf::parallel_transform(
        tf::make_range(tickets), delivered_points,
        [](const std::array<Index, 2> &t) { return t[1]; }, tf::checked);
  } else {
    for (std::size_t f = 1; f <= n_faces; ++f)
      coplanar_offsets[f] += coplanar_offsets[f - 1];
  }

  coplanar_partners.allocate(coplanar_pairs.size() * 2);
  tf::buffer<Index> cursor;
  cursor.allocate(n_faces);
  std::copy(coplanar_offsets.begin(), coplanar_offsets.end() - 1,
            cursor.begin());
  for (const auto &pair : coplanar_pairs) {
    const auto a = Index(flat(pair[0], pair[1]));
    const auto b = Index(flat(pair[2], pair[3]));
    coplanar_partners[std::size_t(cursor[std::size_t(a)]++)] = b;
    coplanar_partners[std::size_t(cursor[std::size_t(b)]++)] = a;
  }

  const auto ids = [](const tf::buffer<Index> &b) {
    return tf::make_range(b.begin(), b.end());
  };
  return {ids(face_offsets), ids(delivered_offsets), ids(delivered_points),
          ids(coplanar_offsets), ids(coplanar_partners)};
}

} // namespace tf::intersect
