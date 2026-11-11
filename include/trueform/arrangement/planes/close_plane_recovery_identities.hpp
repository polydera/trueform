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
#include "../../core/algorithm/sequenced_generate.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/config/verify.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../exact/meta.hpp"
#include "../../exact/tag_of_flat_vertex.hpp"
#include "../../exact/vertex.hpp"
#include "../../intersect/graph/elect_identity_components.hpp"
#include "../../intersect/graph/near_identity_rows.hpp"
#include "../../intersect/graph/plane_identity_collapse.hpp"
#include "../../intersect/graph/plane_identity_names.hpp"
#include "./plane_flat_weld.hpp"
#include "./plane_recovery_name.hpp"
#include "./plane_recovery_statement.hpp"
#include "tbb/parallel_sort.h"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <type_traits>
#include <utility>

namespace tf::arrangement {

/// Close one recovery delta's full-3D coincidences and projected CDT
/// equivalences before assigning any new identity.
///
/// The fresh name classes are candidates of THE ELECTION (@ref
/// tf::intersect::graph::elect_identity_components), and full-3D collapse and
/// projection closure both state its relations, so one election closes the
/// classes either joins. `n_existing_created` adds every still-standing
/// identity minted by a prior wave to the position records, so a new name
/// landing on an earlier PA endpoint reuses it instead of minting a
/// coordinate-equal twin. Retired created identities no longer own their old
/// positions. Negative plane-local direct forms in `standing_created_class`
/// are excluded as well: those may weld only through explicit CDT evidence,
/// never through cross-plane coordinate closure. `candidate_points` states the
/// endpoints a new name may land on in the FLAT vertex currency — the one key
/// space a weld also speaks — so a caller dedupes identities, never the edges
/// that mention them. The standing identities are also the prefix a mint
/// follows, so a delta-only caller's output carries fresh class rows alone
/// while its minted ids continue the persistent created space.
///
/// THE LANDING LAW. `class_ends` are the endpoints of the constraints each
/// class's own proposals split, in proposal order, two per proposal, and
/// `class_end_offsets` states each class's run of proposals. A class MINTED
/// on those constraints and no lattice point away from one of their
/// endpoints IS that endpoint: both name one point of the segment they lie
/// on, the lattice cannot state anything between them, and the piece their
/// difference would leave is a stub with no seam and no depth change —
/// crossable to every fence, which floods two sides of one surface into one
/// component. A class that already speaks a standing identity states its own
/// position and never lands.
///
/// `near_rows` receives, when `near_window` asks, the near table of the
/// delta's position sort as one deferred block (@ref
/// tf::intersect::graph::collapse_plane_identities).
///
/// THE WELD WINDOW, when `weld_window` asks, is a grid and not a radius:
/// identities neither of which is an original, whose positions fall in one
/// lattice box of side `weld_window`, are one coincidence of the election —
/// carriers minting one point the input cannot tell from another. The box
/// is the key, so the relation is transitive by construction and never
/// chains; two identities in adjacent boxes stay two. Originals keep the
/// exact law.
template <typename Index, typename Param, typename GetPoint,
          typename ClassPosition, typename VertexOffsets, typename Int>
auto close_plane_recovery_identities(
    const GetPoint &get_point, const ClassPosition &class_position,
    const VertexOffsets &vertex_offsets,
    tf::buffer<plane_recovery_statement<Index, Param>> &statements,
    tf::buffer<Index> &class_offsets, Index n_classes, Index n_points,
    Index n_flat, tf::buffer<Index> &class_id, tf::buffer<Index> &created_class,
    const tf::buffer<plane_flat_weld<Index>> &welds,
    const tf::buffer<std::array<Index, 3>> &standing_merges,
    tf::buffer<std::array<Index, 3>> &merges,
    const tf::buffer<Index> &candidate_points,
    const tf::buffer<Index> &class_end_offsets,
    const tf::buffer<Index> &class_ends,
    const tf::buffer<Index> &standing_created_class, Index n_existing_created,
    Int near_window,
    tf::intersect::graph::near_identity_rows<Index, tf::exact::pt3<Int>>
        &near_rows,
    Int weld_window) -> void {
  using key_t = tf::intersect::graph::identity_key<Index>;
  using relation_t = tf::intersect::graph::identity_relation<Index>;
  using point_t = std::decay_t<decltype(get_point(std::int16_t(-1), Index(0)))>;
  struct position_t {
    point_t point;
    key_t key;
  };

  const auto class_key = [&](Index cls) -> key_t {
    const auto id = class_id[std::size_t(cls)];
    return id == Index(-1)
               ? tf::intersect::graph::fresh_identity_key<Index>(cls)
               : tf::intersect::graph::standing_identity_key<Index>(id);
  };
  const auto key_of_flat = [&](Index flat) -> key_t {
    return flat < n_flat
               ? tf::intersect::graph::original_identity_key<Index>(flat)
               : tf::intersect::graph::standing_identity_key<Index>(
                     Index(flat - n_flat));
  };
  const auto point_of_flat = [&](Index flat) {
    if (flat >= n_flat)
      return get_point(std::int16_t(-1), flat - n_flat);
    const auto tag = tf::exact::tag_of_flat_vertex(vertex_offsets, flat);
    return get_point(std::int16_t(tag),
                     flat - vertex_offsets[std::size_t(tag)]);
  };

  tf::buffer<position_t> positions;
  positions.reserve(std::size_t(n_classes) + std::size_t(n_existing_created) +
                    welds.size() * 2 + candidate_points.size());
  tf::buffer<relation_t> relations;
  relations.reserve(std::size_t(n_classes) + welds.size());
  // one position per class and one per candidate point: both extents are known
  // before the walk, so the blocks are written in place
  positions.allocate(std::size_t(n_classes) + candidate_points.size());
  tf::parallel_for_each(
      tf::make_sequence_range(std::size_t(n_classes)),
      [&](std::size_t cls) {
        positions[cls] = {class_position(Index(cls)), class_key(Index(cls))};
      },
      tf::checked);
  tf::sequenced_generate(
      tf::make_sequence_range(std::size_t(n_classes)), relations,
      [&](std::size_t cls, tf::buffer<relation_t> &out) {
        const auto &name = statements[std::size_t(class_offsets[cls])].name;
        if (name.feature[0] == Index(1))
          out.push_back({class_key(Index(cls)),
                         tf::intersect::graph::original_identity_key<Index>(
                             Index(name.feature[1]))});
      },
      tf::checked);
  tf::parallel_for_each(
      tf::make_sequence_range(candidate_points.size()),
      [&](std::size_t at) {
        const auto flat = candidate_points[at];
        positions[std::size_t(n_classes) + at] = {point_of_flat(flat),
                                                  key_of_flat(flat)};
      },
      tf::checked);
  const auto no_lattice_point_between = [](const point_t &a,
                                           const point_t &b) -> bool {
    using coordinate_t = typename point_t::coordinate_type;
    using wide_t = typename tf::exact::meta<coordinate_t>::T1;
    for (std::size_t k = 0; k < 3; ++k) {
      const auto gap = wide_t(a[k]) - wide_t(b[k]);
      if (gap > wide_t(1) || gap < wide_t(-1))
        return false;
    }
    return true;
  };
  tf::sequenced_generate(
      tf::make_sequence_range(std::size_t(n_classes)), relations,
      [&](std::size_t cls, tf::buffer<relation_t> &out) {
        if (class_id[cls] != Index(-1))
          return;
        const auto from = std::size_t(class_end_offsets[cls]) * 2;
        const auto to = std::size_t(class_end_offsets[cls + 1]) * 2;
        for (auto at = from; at < to; ++at) {
          const auto flat = class_ends[at];
          if (no_lattice_point_between(positions[cls].point,
                                       point_of_flat(flat)))
            out.push_back({class_key(Index(cls)), key_of_flat(flat)});
        }
      },
      tf::checked);
  if constexpr (tf::core::verification)
    tf::core::verify(std::size_t(n_existing_created) <=
                         standing_created_class.size(),
                     "every standing created identity has a class");
  tf::sequenced_generate(
      tf::make_sequence_range(std::size_t(n_existing_created)), positions,
      [&](std::size_t created, tf::buffer<position_t> &out) {
        if (standing_created_class[created] < Index(0))
          return;
        const auto id = n_points + Index(created);
        // a retired created identity no longer owns its old position
        if (tf::intersect::graph::plane_merge_target(standing_merges, Index(1),
                                                     id) != Index(-1))
          return;
        out.push_back({get_point(std::int16_t(-1), id),
                       tf::intersect::graph::standing_identity_key<Index>(id)});
      },
      tf::checked);
  for (const auto &weld : welds) {
    if (weld.flat_0 == weld.flat_1)
      continue;
    const auto key_0 = key_of_flat(weld.flat_0);
    const auto key_1 = key_of_flat(weld.flat_1);
    positions.push_back({point_of_flat(weld.flat_0), key_0});
    positions.push_back({point_of_flat(weld.flat_1), key_1});
    relations.push_back({key_0, key_1});
  }

  tbb::parallel_sort(positions.begin(), positions.end(),
                     [](const position_t &x, const position_t &y) {
                       return std::tie(x.point[0], x.point[1], x.point[2],
                                       x.key) < std::tie(y.point[0], y.point[1],
                                                         y.point[2], y.key);
                     });
  positions.erase_till_end(
      std::unique(positions.begin(), positions.end(),
                  [](const position_t &x, const position_t &y) {
                    return x.point == y.point && x.key == y.key;
                  }));
  // A run of one position is one identity. Stating the run as a chain of
  // adjacent pairs is the same equivalence as a star from its first row — the
  // union closes either way — and it is one parallel adjacency test.
  tf::sequenced_generate(
      tf::make_sequence_range(positions.size()), relations,
      [&positions](std::size_t at, tf::buffer<relation_t> &out) {
        if (at != 0 && positions[at - 1].point == positions[at].point)
          out.push_back({positions[at - 1].key, positions[at].key});
      },
      tf::checked);

  if (weld_window > Int(0)) {
    using coordinate_t = typename point_t::coordinate_type;
    using cell_t = std::array<coordinate_t, 3>;
    const auto cell_of = [weld_window](const point_t &q) {
      cell_t cell{};
      for (std::size_t k = 0; k < 3; ++k) {
        auto c = coordinate_t(q[k] / coordinate_t(weld_window));
        if (q[k] - c * coordinate_t(weld_window) < coordinate_t(0))
          c = coordinate_t(c - coordinate_t(1));
        cell[k] = c;
      }
      return cell;
    };
    struct cell_box {
      cell_t cell;
      Index at;
      auto operator<(const cell_box &other) const -> bool {
        return cell < other.cell || (cell == other.cell && at < other.at);
      }
    };
    tf::buffer<cell_box> boxes;
    tf::sequenced_generate(
        tf::make_sequence_range(positions.size()), boxes,
        [&](std::size_t at, tf::buffer<cell_box> &out) {
          if (positions[at].key[0] !=
              Index(tf::intersect::graph::original_identity_rank))
            out.push_back({cell_of(positions[at].point), Index(at)});
        },
        tf::checked);
    tbb::parallel_sort(boxes.begin(), boxes.end());
    tf::sequenced_generate(
        tf::make_sequence_range(boxes.size()), relations,
        [&](std::size_t at, tf::buffer<relation_t> &out) {
          if (at == 0 || !(boxes[at - 1].cell == boxes[at].cell))
            return;
          const auto &here = positions[std::size_t(boxes[at - 1].at)].key;
          const auto &there = positions[std::size_t(boxes[at].at)].key;
          if (!(here == there))
            out.push_back({here, there});
        },
        tf::checked);
  }

  tf::buffer<key_t> keys;
  keys.reserve(std::size_t(n_classes) + relations.size() * 2);
  for (Index cls = 0; cls < n_classes; ++cls)
    if (class_id[std::size_t(cls)] == Index(-1))
      keys.push_back(class_key(cls));
  tf::buffer<Index> targets;
  created_class.reserve(created_class.size() + std::size_t(n_classes));
  // a coincidence the originals alone state mints a rootless class naming
  // the smallest of them
  const auto mint = [&](const key_t &winner) {
    auto cls = winner[1];
    if (winner[0] == Index(tf::intersect::graph::original_identity_rank)) {
      if (class_offsets.size() == 0)
        class_offsets.push_back(Index(0));
      cls = Index(class_offsets.size()) - Index(1);
      statements.push_back(
          {make_plane_recovery_name<Index, Param>(
               tf::intersect::graph::plane_vertex_name<Index>(winner[1])),
           Index(-1), Index(-1), Index(0)});
      class_offsets.push_back(Index(statements.size()));
    }
    created_class.push_back(cls);
    return n_points + n_existing_created + Index(created_class.size()) -
           Index(1);
  };
  tf::intersect::graph::elect_identity_components(keys, relations, mint,
                                                  targets, merges);

  // The election's answer, read where each kind of class keeps it. An already
  // named class speaks an existing created identity, and what the election did
  // to one of those is exactly what it just published in `merges`. A fresh
  // class keys as `{1, cls}`, so the sorted key array is a permutation of the
  // fresh classes and its back-map is the answer.
  tf::parallel_for_each(
      tf::make_sequence_range(std::size_t(n_classes)),
      [&class_id, &merges](std::size_t cls) {
        const auto id = class_id[cls];
        if (id == Index(-1))
          return;
        const auto target =
            tf::intersect::graph::plane_merge_target(merges, Index(1), id);
        if (target != Index(-1))
          class_id[cls] = target;
      },
      tf::checked);
  tf::parallel_for_each(
      tf::make_sequence_range(keys.size()),
      [&keys, &targets, &class_id](std::size_t at) {
        const auto &key = keys[at];
        if (key[0] == Index(tf::intersect::graph::fresh_identity_rank))
          class_id[std::size_t(key[1])] = targets[at];
      },
      tf::checked);

  // the sort joins the table as one block, each class as the identity the
  // election gave it: the class ids are this wave's, and die with it
  if (near_window > Int(0)) {
    using near_row_t = typename tf::intersect::graph::near_identity_rows<
        Index, tf::exact::pt3<Int>>::row_t;
    tf::buffer<near_row_t> rows;
    rows.allocate(positions.size());
    tf::parallel_transform(
        positions, rows,
        [&class_id](const position_t &position) {
          const auto &key = position.key;
          const bool fresh =
              key[0] == Index(tf::intersect::graph::fresh_identity_rank);
          return near_row_t{
              position.point,
              fresh ? Index(tf::intersect::graph::standing_identity_rank)
                    : key[0],
              fresh ? class_id[std::size_t(key[1])] : key[1]};
        },
        tf::checked);
    near_rows.append(std::move(rows), Index(0));
  }
}

} // namespace tf::arrangement
