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
#include "../core/polygons.hpp"
#include "../core/polygons_buffer.hpp"
#include "../core/static_size.hpp"
#include "../core/views/enumerate.hpp"
#include "../core/views/indirect_range.hpp"
#include "../core/views/offset_block_range.hpp"
#include "../core/views/take.hpp"
#include "../core/views/zip.hpp"
#include "../topology/domain_labels.hpp"
#include "tbb/parallel_sort.h"

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
  tbb::parallel_sort(ids.begin(), ids.end(),
                     [&](auto a, auto b) { return flat[a] < flat[b]; });
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
    tf::parallel_for_each(tf::enumerate(comp_ids), [&](auto pair) {
      auto &&[i, elem_id] = pair;
      bool reverse = (elem_id & Index(1)) == Index(0);
      auto face = polygons.faces()[elem_id / 2];
      auto n_v = face.size();
      Index write_at;
      if constexpr (N == tf::dynamic_size)
        write_at = out.faces_buffer().offsets_buffer()[i];
      else
        write_at = Index(N) * Index(i);
      for (decltype(n_v) k = 0; k < n_v; ++k) {
        auto src_idx = reverse ? (n_v - 1 - k) : k;
        data[write_at + k] = point_map[face[src_idx]] - point_sentinel;
      }
    });

    out.points_buffer().allocate(pt_ids.size());
    tf::parallel_copy(tf::make_indirect_range(pt_ids, polygons.points()),
                      out.points_buffer());
    point_sentinel = current;
  }
  return std::make_pair(std::move(out_v), std::move(out_labels));
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
  return reindex::split_into_domains_polygons<Index>(polygons, dl);
}

} // namespace tf
