/*
 * Copyright (c) 2025 XLAB
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
#include "../core/algorithm/compute_offsets.hpp"
#include "../core/algorithm/parallel_copy.hpp"
#include "../core/algorithm/parallel_fill.hpp"
#include "../core/algorithm/parallel_for_each.hpp"
#include "../core/algorithm/parallel_iota.hpp"
#include "../core/offset_block_buffer.hpp"
#include "../core/polygons.hpp"
#include "../core/polygons_buffer.hpp"
#include "../core/static_size.hpp"
#include "../core/views/enumerate.hpp"
#include "../core/views/indirect_range.hpp"
#include "../core/views/offset_block_range.hpp"
#include "../core/views/take.hpp"
#include "../core/views/zip.hpp"
#include "../topology/domain_labels.hpp"
#include "./return_source_ids.hpp"
#include "tbb/parallel_sort.h"
#include <tuple>

namespace tf {

/// @cond INTERNAL
namespace reindex {

/// Per-side polygon split keyed by @ref tf::domain_labels. The sentinel
/// class (label `dl.n_domains`) sorts last and is trimmed via
/// `tf::take`. Side-0 emissions reverse the stored winding; side-1
/// emissions keep it.
template <typename Index, typename Policy, typename L>
auto split_into_domains_polygons(const tf::polygons<Policy> &polygons,
                                 const tf::domain_labels<L> &dl) {
  using RealT = tf::coordinate_type<Policy>;
  constexpr auto Dims = tf::coordinate_dims_v<Policy>;
  constexpr auto N = tf::static_size_v<decltype(polygons[0])>;
  using out_t = tf::polygons_buffer<Index, RealT, Dims, N>;

  const auto &flat = dl.labels.data_buffer();
  auto n_per_side = static_cast<Index>(flat.size());

  tf::buffer<Index> ids;
  ids.allocate(n_per_side);
  tf::parallel_iota(ids, Index(0));
  // Total order: tie-break equal-label emissions by id so the per-domain
  // emission order (and thus the emitted faces and source ids) is
  // reproducible, not left to parallel_sort's unstable tie handling.
  tbb::parallel_sort(ids.begin(), ids.end(), [&](auto a, auto b) {
    return flat[a] != flat[b] ? flat[a] < flat[b] : a < b;
  });
  tf::buffer<Index> offsets;
  offsets.reserve(dl.n_domains);
  tf::compute_offsets(ids, std::back_inserter(offsets), Index(0),
                      [&](auto a, auto b) { return flat[a] == flat[b]; });
  auto ids_r = tf::take(tf::make_offset_block_range(offsets, ids),
                        std::size_t(dl.n_domains));
  auto n_components = static_cast<Index>(ids_r.size());

  tf::buffer<L> out_labels;
  out_labels.allocate(n_components);

  tf::buffer<Index> point_map;
  point_map.allocate(polygons.points().size());
  tf::parallel_fill(point_map, Index(-1));
  tf::buffer<Index> pt_ids;
  Index point_sentinel = 0;

  std::vector<out_t> out_v(n_components);
  for (auto &&[comp_ids, out, l] : tf::zip(ids_r, out_v, out_labels)) {
    l = flat[comp_ids.front()];
    Index current = point_sentinel;
    pt_ids.clear();
    for (auto elem_id : comp_ids) {
      auto face = polygons.faces()[elem_id / 2];
      for (auto v_id : face) {
        auto &v = point_map[v_id];
        if (v == -1 || v < point_sentinel) {
          pt_ids.push_back(v_id);
          v = current++;
        }
      }
    }

    auto n_faces_out = static_cast<Index>(comp_ids.size());
    if constexpr (N == tf::dynamic_size) {
      auto &offs = out.faces_buffer().offsets_buffer();
      offs.allocate(n_faces_out + 1);
      offs[0] = 0;
      std::size_t total = 0;
      for (auto [i, elem_id] : tf::enumerate(comp_ids)) {
        total += polygons.faces()[elem_id / 2].size();
        offs[i + 1] = static_cast<Index>(total);
      }
      out.faces_buffer().data_buffer().allocate(total);
    } else {
      out.faces_buffer().data_buffer().allocate(N * n_faces_out);
    }

    auto &data = out.faces_buffer().data_buffer();
    if constexpr (N == tf::dynamic_size) {
      auto &offs = out.faces_buffer().offsets_buffer();
      tf::parallel_for_each(tf::enumerate(comp_ids), [&](auto pair) {
        auto &&[i, elem_id] = pair;
        bool reverse = (elem_id & Index(1)) == Index(0);
        auto face = polygons.faces()[elem_id / 2];
        auto n_v = face.size();
        Index write_at = offs[i];
        for (decltype(n_v) k = 0; k < n_v; ++k) {
          auto src_idx = reverse ? (n_v - 1 - k) : k;
          data[write_at + k] = point_map[face[src_idx]] - point_sentinel;
        }
      });
    } else {
      tf::parallel_for_each(tf::enumerate(comp_ids), [&](auto pair) {
        constexpr auto N = tf::static_size_v<decltype(polygons[0])>;
        auto &&[i, elem_id] = pair;
        bool reverse = (elem_id & Index(1)) == Index(0);
        auto face = polygons.faces()[elem_id / 2];
        auto n_v = face.size();
        Index write_at = Index(N) * Index(i);
        for (decltype(n_v) k = 0; k < n_v; ++k) {
          auto src_idx = reverse ? (n_v - 1 - k) : k;
          data[write_at + k] = point_map[face[src_idx]] - point_sentinel;
        }
      });
    }

    out.points_buffer().allocate(pt_ids.size());
    tf::parallel_copy(tf::make_indirect_range(pt_ids, polygons.points()),
                      out.points_buffer());
    point_sentinel = current;
  }

  // Source emissions per kept domain, parallel to out_v. `offsets` carries
  // one block per distinct present label; trim it (and the data tail) down to
  // the n_components kept blocks so the sentinel/garbage block is dropped and
  // no empty trailing block survives. Move, no copy. Guard the empty case:
  // when there are no emissions `offsets` is itself empty, and reallocating it
  // to n_components+1 (== 1) would expose an uninitialized entry.
  if (offsets.size() != 0) {
    offsets.reallocate(static_cast<std::size_t>(n_components) + 1);
    // offsets.back() is now the start of the dropped sentinel block == the
    // number of emissions in the kept blocks.
    ids.reallocate(static_cast<std::size_t>(offsets.back()));
  }
  tf::offset_block_buffer<Index, Index> source;
  source.offsets_buffer() = std::move(offsets);
  source.data_buffer() = std::move(ids);
  return std::make_tuple(std::move(out_v), std::move(out_labels),
                         std::move(source));
}

} // namespace reindex
/// @endcond

/// @ingroup reindex
/// @brief Split polygons by @ref tf::domain_labels.
///
/// Each face contributes one or two emissions, one per side bound to a
/// real domain. Sentinel-bound sides (open-fragment garbage in Mode 1)
/// are dropped. Side-0 emissions reverse the stored winding so each
/// output buffer has outward-of-domain normals; side-1 emissions keep
/// the stored winding.
///
/// @tparam Policy The polygons policy.
/// @tparam L The domain label integer type.
/// @param polygons The input @ref tf::polygons.
/// @param dl Per-face per-side @ref tf::domain_labels.
/// @return Pair of (vector of @ref tf::polygons_buffer, vector of domain ids).
template <typename Policy, typename L>
auto split_into_domains(const tf::polygons<Policy> &polygons,
                        const tf::domain_labels<L> &dl) {
  using Index = std::decay_t<decltype(polygons.faces()[0][0])>;
  auto [out_v, out_labels, source] =
      reindex::split_into_domains_polygons<Index>(polygons, dl);
  (void)source;
  return std::make_pair(std::move(out_v), std::move(out_labels));
}

/// @ingroup reindex
/// @brief Split polygons by @ref tf::domain_labels, additionally returning
///        the source face ids per output domain.
///
/// `source` is an @ref tf::offset_block_buffer whose blocks run parallel to
/// the returned meshes: `source[c][k]` is the original face id of mesh `c`'s
/// face `k`, in the same order the faces are emitted. A face bounds two
/// domains (one per side), so its id appears in both domains' blocks.
///
/// @param polygons The input @ref tf::polygons.
/// @param dl Per-face per-side @ref tf::domain_labels.
/// @return Tuple of (vector of @ref tf::polygons_buffer, vector of domain ids,
///         @ref tf::offset_block_buffer of source face ids per domain).
template <typename Policy, typename L>
auto split_into_domains(const tf::polygons<Policy> &polygons,
                        const tf::domain_labels<L> &dl,
                        tf::return_source_ids_t) {
  using Index = std::decay_t<decltype(polygons.faces()[0][0])>;
  auto [out_v, out_labels, source] =
      reindex::split_into_domains_polygons<Index>(polygons, dl);
  // The impl emits side-tagged ids (2*f + side); halve to the face id.
  tf::parallel_for_each(source.data_buffer(), [](Index &x) { x /= 2; });
  return std::make_tuple(std::move(out_v), std::move(out_labels),
                         std::move(source));
}

} // namespace tf
