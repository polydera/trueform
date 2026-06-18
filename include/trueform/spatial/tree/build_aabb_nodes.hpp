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
#include "../../core/aabb.hpp"
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/algorithm/partition_range_into_parts.hpp"
#include "../../core/buffer.hpp"
#include "../../core/views/enumerate.hpp"
#include "../../core/largest_axis.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../core/views/zip.hpp"
#include "../tree_config.hpp"
#include "./max_nodes_in_tree.hpp"
#include "./tree_node.hpp"

#include <array>
#include <cstring>
#include <type_traits>
namespace tf::spatial {
namespace detail {

// The build partitions one packed stream: an element swaps as one cache
// line touch, where separate id and center arrays would pay two streams
// per swap.
template <typename Index, typename RealT, std::size_t Dims>
struct centroid_entry {
  Index id;
  RealT c[Dims];
};

template <typename Index, typename RealT, std::size_t Dims>
auto centroid_bounds(const centroid_entry<Index, RealT, Dims> *first,
                     const centroid_entry<Index, RealT, Dims> *last)
    -> tf::aabb<RealT, Dims> {
  tf::aabb<RealT, Dims> r;
  for (std::size_t d = 0; d < Dims; ++d)
    r.min[d] = r.max[d] = first->c[d];
  for (auto it = first + 1; it != last; ++it)
    for (std::size_t d = 0; d < Dims; ++d) {
      r.min[d] = std::min(r.min[d], it->c[d]);
      r.max[d] = std::max(r.max[d], it->c[d]);
    }
  return r;
}

// Above this size a node is split by one parallel histogram pass and one
// parallel scatter instead of the serial per-node selection; below it the
// selection wins.
inline constexpr std::size_t parallel_split_threshold = 196608;
inline constexpr int parallel_split_max_parts = 16;

// The ranks partition_range_into_parts realises for `parts` children, in
// ascending order. The parallel split must reproduce them exactly: the
// implicit-heap node addressing (first_child = inner_size * id + 1) sizes
// the node buffer for these subtree sizes, and a child grown past its rank
// can deepen its subtree beyond the allocation.
inline auto partition_part_ranks(std::size_t n, int parts, std::size_t base,
                             std::size_t *out, int &w) -> void {
  if (parts <= 1)
    return;
  int left = parts / 2;
  std::size_t mid = n * std::size_t(left) / std::size_t(parts);
  partition_part_ranks(mid, left, base, out, w);
  out[w++] = base + mid;
  partition_part_ranks(n - mid, parts - left, base + mid, out, w);
}

// Parallel `parts`-way split of n entries by c[axis], scattering into the
// destination array. Boundaries land exactly on the stock ranks: whole
// buckets classify by position against the straddle buckets, and each
// straddle bucket's population is divided by a per-chunk quota. Returns
// false on a degenerate key distribution (caller falls back to the serial
// partitioner).
template <typename Entry, typename RealT>
auto parallel_rank_split(const Entry *first, Entry *dst, std::size_t n,
                         int axis, RealT key_lo, RealT key_hi, int parts,
                         std::size_t *ranks_out) -> bool {
  constexpr int n_buckets = 1024;
  constexpr int max_parts = parallel_split_max_parts;
  // MSVC rejects a function-local constexpr as a std::array size argument from
  // inside a lambda (cannot prove it constant); alias the array types here so
  // the size is fixed at the declaration, not referenced through the lambda.
  using parts_array_t = std::array<std::size_t, max_parts>;
  using bounds_array_t = std::array<std::size_t, max_parts - 1>;
  if (parts < 2 || parts > max_parts)
    return false;
  if (!(key_lo < key_hi))
    return false;

  // Bucket mapping per coordinate type: integer grids shift exactly (no
  // floating point touches a grid coordinate), floats scale in double.
  [[maybe_unused]] double lo = 0, scale = 0;
  [[maybe_unused]] std::make_unsigned_t<
      std::conditional_t<std::is_integral_v<RealT>, RealT, int>>
      u_lo = 0;
  [[maybe_unused]] int shift = 0;
  if constexpr (std::is_integral_v<RealT>) {
    using U = decltype(u_lo);
    u_lo = U(key_lo);
    U span = U(key_hi) - u_lo;
    while ((span >> shift) >= U(n_buckets))
      ++shift;
  } else {
    lo = double(key_lo);
    const double span = double(key_hi) - lo;
    if (!(span > 0))
      return false;
    scale = double(n_buckets) / span;
  }
  auto bucket_of = [&](RealT key) -> int {
    if constexpr (std::is_integral_v<RealT>) {
      using U = decltype(u_lo);
      return int((U(key) - u_lo) >> shift);
    } else {
      int b = int((double(key) - lo) * scale);
      return b < 0 ? 0 : (b >= n_buckets ? n_buckets - 1 : b);
    }
  };

  const std::size_t chunk = 65536;
  const int n_chunks = int((n + chunk - 1) / chunk);
  buffer<int> hist_store;
  hist_store.allocate(std::size_t(n_chunks) * n_buckets);
  int *hists = hist_store.data();
  buffer<std::size_t> off_store, quota_store;
  off_store.allocate(std::size_t(n_chunks) * std::size_t(parts));
  quota_store.allocate(std::size_t(n_chunks) * std::size_t(parts));
  std::size_t *chunk_off = off_store.data();
  std::size_t *chunk_quota = quota_store.data();

  tf::parallel_for_each(tf::make_sequence_range(n_chunks), [&](int c) {
    int *h = hists + std::size_t(c) * n_buckets;
    std::memset(h, 0, n_buckets * sizeof(int));
    std::size_t b = std::size_t(c) * chunk;
    std::size_t e = std::min(n, b + chunk);
    for (std::size_t i = b; i < e; ++i)
      ++h[bucket_of(first[i].c[axis])];
  });

  bounds_array_t want;
  const int n_bounds = parts - 1;
  {
    int w = 0;
    partition_part_ranks(n, parts, 0, want.data(), w);
  }

  std::array<int, n_buckets> totals{};
  for (int c = 0; c < n_chunks; ++c) {
    const int *h = hists + std::size_t(c) * n_buckets;
    for (int b = 0; b < n_buckets; ++b)
      totals[b] += h[b];
  }

  std::array<int, max_parts - 1> straddle;
  bounds_array_t residual;
  {
    std::size_t cum = 0;
    int w = 0;
    for (int b = 0; b < n_buckets && w < n_bounds; ++b) {
      std::size_t next = cum + std::size_t(totals[b]);
      while (w < n_bounds && next >= want[w]) {
        straddle[w] = b;
        residual[w] = want[w] - cum;
        ++w;
      }
      cum = next;
    }
    if (w < n_bounds)
      return false;
  }
  for (int w = 1; w < n_bounds; ++w)
    if (straddle[w] == straddle[w - 1])
      return false;

  auto class_of_bucket = [&](int b) -> int {
    int cls = 0;
    for (int w = 0; w < n_bounds; ++w)
      cls += b > straddle[w];
    return cls;
  };
  auto straddle_of_bucket = [&](int b) -> int {
    for (int w = 0; w < n_bounds; ++w)
      if (b == straddle[w])
        return w;
    return -1;
  };

  {
    parts_array_t run;
    run[0] = 0;
    for (int w = 0; w < n_bounds; ++w)
      run[w + 1] = want[w];
    bounds_array_t strad_run = residual;
    for (int c = 0; c < n_chunks; ++c) {
      const int *h = hists + std::size_t(c) * n_buckets;
      parts_array_t mine{};
      for (int b = 0; b < n_buckets; ++b) {
        if (straddle_of_bucket(b) >= 0)
          continue;
        mine[class_of_bucket(b)] += std::size_t(h[b]);
      }
      for (int w = 0; w < n_bounds; ++w) {
        std::size_t have = std::size_t(h[straddle[w]]);
        std::size_t q = std::min(strad_run[w], have);
        strad_run[w] -= q;
        chunk_quota[std::size_t(c) * n_bounds + w] = q;
        mine[w] += q;
        mine[w + 1] += have - q;
      }
      for (int cls = 0; cls < parts; ++cls) {
        chunk_off[std::size_t(c) * parts + cls] = run[cls];
        run[cls] += mine[cls];
      }
    }
  }

  tf::parallel_for_each(tf::make_sequence_range(n_chunks), [&](int c) {
    parts_array_t off;
    for (int cls = 0; cls < parts; ++cls)
      off[cls] = chunk_off[std::size_t(c) * parts + cls];
    bounds_array_t quota;
    for (int w = 0; w < n_bounds; ++w)
      quota[w] = chunk_quota[std::size_t(c) * n_bounds + w];
    std::size_t b = std::size_t(c) * chunk;
    std::size_t e = std::min(n, b + chunk);
    for (std::size_t i = b; i < e; ++i) {
      int bk = bucket_of(first[i].c[axis]);
      int w = straddle_of_bucket(bk);
      int cls = w >= 0 ? (quota[w] ? (--quota[w], w) : w + 1)
                       : class_of_bucket(bk);
      dst[off[cls]++] = first[i];
    }
  });

  for (int w = 0; w < n_bounds; ++w)
    ranks_out[w] = want[w];
  return true;
}

// Recursive helper over the packed centroid entries: the partitioner
// reorders them in place by c[max_axis]. The outer entry point already
// wrote entry ids (0..n-1 or the user's ids).
template <typename Partitioner, typename Index, typename RealT,
          std::size_t Dims, typename Range1>
auto build_tree_nodes(buffer<tree_node<Index, tf::aabb<RealT, Dims>>> &nodes,
                      const Range1 &aabbs,
                      centroid_entry<Index, RealT, Dims> *first,
                      centroid_entry<Index, RealT, Dims> *last,
                      Index node_id, Index offset,
                      const tf::tree_config &config) -> void {
  Index n_ids = Index(last - first);

  // Leaf: union the primitive AABBs of this leaf's prims.
  if (n_ids <= config.leaf_size) {
    nodes[node_id].bv = aabbs[first->id];
    for (auto it = first + 1; it != last; ++it)
      aabb_union_inplace(nodes[node_id].bv, aabbs[it->id]);
    nodes[node_id].set_data(offset, n_ids);
    nodes[node_id].set_as_leaf();
    return;
  }

  // Inner: split-axis from the bbox of centroids.
  auto cbb = centroid_bounds(first, last);
  int max_axis = tf::largest_axis(cbb.diagonal());
  nodes[node_id].axis = max_axis;

  Index first_child = config.inner_size * node_id + 1;

  Index n_children = tf::partition_range_into_parts(
      tf::make_range(first, last), config.inner_size,
      [&](auto begin, auto mid, auto end) {
        Partitioner::partition(begin, mid, end,
            [max_axis](const auto &a, const auto &b) {
              return a.c[max_axis] < b.c[max_axis];
            });
      },
      [&](auto &&range, Index this_node_id) {
        Index this_offset = Index(range.begin() - first);
        build_tree_nodes<Partitioner>(nodes, aabbs, range.begin(), range.end(),
                                      this_node_id, offset + this_offset,
                                      config);
      },
      first_child);

  // Compose this node's BV from the children's already-written BVs.
  nodes[node_id].bv = nodes[first_child].bv;
  for (Index k = 1; k < n_children; ++k)
    aabb_union_inplace(nodes[node_id].bv, nodes[first_child + k].bv);

  nodes[node_id].set_data(first_child, n_children);
}

// Large-node recursion: the parallel rank split scatters into the OTHER
// entry buffer and recursion continues there (ping-pong, no copy-back). A
// subtree finishing anywhere writes its slice of the final ids directly,
// so neither buffer is ever copied wholesale.
template <typename Partitioner, typename Index, typename RealT,
          std::size_t Dims, typename Range1>
auto build_tree_nodes_split(
    buffer<tree_node<Index, tf::aabb<RealT, Dims>>> &nodes,
    const Range1 &aabbs, centroid_entry<Index, RealT, Dims> *first,
    centroid_entry<Index, RealT, Dims> *other, std::size_t n_ids,
    Index node_id, Index offset, buffer<Index> &ids_out,
    const tf::tree_config &config) -> void {
  auto serial_here = [&] {
    build_tree_nodes<Partitioner>(nodes, aabbs, first, first + n_ids, node_id,
                                  offset, config);
    Index *out = ids_out.data() + offset;
    tf::parallel_for_each(
        tf::zip(tf::make_range(first, first + n_ids),
                tf::make_range(out, out + n_ids)),
        [](auto pair) {
          auto &&[en, id] = pair;
          id = en.id;
        },
        tf::checked);
  };

  if (n_ids < parallel_split_threshold) {
    serial_here();
    return;
  }

  auto cbb = centroid_bounds(first, first + n_ids);
  int max_axis = tf::largest_axis(cbb.diagonal());

  const int parts = config.inner_size;
  std::array<std::size_t, parallel_split_max_parts - 1> ranks;
  if (!parallel_rank_split(first, other, n_ids, max_axis,
                                              cbb.min[max_axis],
                                              cbb.max[max_axis], parts,
                                              ranks.data())) {
    serial_here();
    return;
  }

  nodes[node_id].axis = max_axis;
  Index first_child = config.inner_size * node_id + 1;

  std::array<std::size_t, parallel_split_max_parts + 1> splits;
  splits[0] = 0;
  for (int w = 0; w < parts - 1; ++w)
    splits[w + 1] = ranks[w];
  splits[parts] = n_ids;
  tf::parallel_for_each(tf::make_sequence_range(parts), [&](int c) {
    build_tree_nodes_split<Partitioner>(
        nodes, aabbs, other + splits[c], first + splits[c],
        splits[c + 1] - splits[c], first_child + Index(c),
        offset + Index(splits[c]), ids_out, config);
  });

  nodes[node_id].bv = nodes[first_child].bv;
  for (Index k = 1; k < Index(parts); ++k)
    aabb_union_inplace(nodes[node_id].bv, nodes[first_child + k].bv);
  nodes[node_id].set_data(first_child, Index(parts));
}

} // namespace detail

template <typename Partitioner, typename Index, typename RealT,
          std::size_t Dims, typename Range0, typename Range1>
auto build_tree_nodes(buffer<tree_node<Index, tf::aabb<RealT, Dims>>> &nodes,
                      buffer<Index> &ids, const Range0 &primitives,
                      const Range1 &aabbs, tf::tree_config config,
                      bool use_ids = false) -> void {
  (void)primitives;
  nodes.clear();
  if (!aabbs.size()) {
    ids.clear();
    return;
  }
  Index n_ids = use_ids ? Index(ids.size()) : Index(aabbs.size());
  nodes.allocate(max_nodes_in_tree(n_ids, config.inner_size, config.leaf_size));
  tf::parallel_for_each(nodes, [](auto &x) { x.set_as_empty(); }, tf::checked);
  if (!use_ids)
    ids.allocate(aabbs.size());

  using entry_t = detail::centroid_entry<Index, RealT, Dims>;
  buffer<entry_t> entries;
  entries.allocate(std::size_t(n_ids));
  if (!use_ids) {
    tf::parallel_for_each(tf::enumerate(aabbs), [&](auto pair) {
      auto &&[i, bb] = pair;
      auto &en = entries[i];
      en.id = Index(i);
      for (std::size_t d = 0; d < Dims; ++d) en.c[d] = bb.min[d] + bb.max[d];
    }, tf::checked);
  } else {
    tf::parallel_for_each(tf::enumerate(ids), [&](auto pair) {
      auto &&[i, id] = pair;
      auto const &bb = aabbs[id];
      auto &en = entries[i];
      en.id = id;
      for (std::size_t d = 0; d < Dims; ++d) en.c[d] = bb.min[d] + bb.max[d];
    }, tf::checked);
  }

  if (std::size_t(n_ids) >= detail::parallel_split_threshold &&
      config.inner_size >= 2 &&
      config.inner_size <= detail::parallel_split_max_parts) {
    buffer<entry_t> entries_scratch;
    entries_scratch.allocate(std::size_t(n_ids));
    detail::build_tree_nodes_split<Partitioner>(nodes, aabbs, entries.data(),
                                        entries_scratch.data(),
                                        std::size_t(n_ids), Index(0), Index(0),
                                        ids, config);
    return;
  }

  detail::build_tree_nodes<Partitioner>(nodes, aabbs, entries.data(),
                                        entries.data() + n_ids, Index(0),
                                        Index(0), config);
  tf::parallel_for_each(tf::enumerate(ids), [&](auto pair) {
    auto &&[i, id] = pair;
    id = entries[i].id;
  }, tf::checked);
}
} // namespace tf::spatial
