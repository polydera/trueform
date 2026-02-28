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
#include "../loop/vertex.hpp"
#include "../../core/buffer.hpp"
#include "../../core/hash_map.hpp"
#include "../../core/hash_set.hpp"
#include <iostream>

namespace tf::cut::debug {

template <typename Index>
struct verify_result {
  int n_ok = 0;
  int n_no_rev = 0;
  int n_multi = 0;
  bool passed() const { return n_no_rev == 0 && n_multi == 0; }
};

template <typename Index>
auto verify_cut_face_edges(
    const tf::buffer<tf::loop::vertex<Index>> &base_loop,
    const tf::buffer<tf::loop::vertex<Index>> &loop_vertices,
    const tf::buffer<Index> &loop_offsets,
    Index loop_start_offset,
    Index offsets_start,
    Index n_loops,
    Index object = -1) -> verify_result<Index> {

  verify_result<Index> result;
  if (base_loop.size() == 0)
    return result;

  struct DirEdge {
    Index a_id, b_id;
    int a_src, b_src;
    bool operator==(const DirEdge &o) const {
      return a_id == o.a_id && b_id == o.b_id &&
             a_src == o.a_src && b_src == o.b_src;
    }
  };
  struct DirEdgeHash {
    std::size_t operator()(const DirEdge &e) const {
      return std::hash<long long>()(
          ((long long)e.a_id << 32) | (long long)e.b_id) ^
          std::hash<int>()(e.a_src * 3 + e.b_src);
    }
  };

  tf::hash_set<DirEdge, DirEdgeHash> base_edges;
  {
    auto bsz = base_loop.size();
    auto bprev = bsz - 1;
    for (decltype(bsz) bi = 0; bi < bsz; bprev = bi++)
      base_edges.insert({base_loop[bprev].id, base_loop[bi].id,
                         (int)base_loop[bprev].source, (int)base_loop[bi].source});
  }

  tf::hash_map<DirEdge, int, DirEdgeHash> non_base_count;
  for (Index li = 0; li < n_loops; li++) {
    auto lo = loop_offsets[offsets_start + li] - loop_start_offset;
    auto hi = (li + 1 < n_loops)
        ? loop_offsets[offsets_start + li + 1] - loop_start_offset
        : loop_vertices.size() - loop_start_offset;
    auto sz = hi - lo;
    auto pprev = sz - 1;
    for (decltype(sz) pi = 0; pi < sz; pprev = pi++) {
      auto va = loop_vertices[loop_start_offset + lo + pprev];
      auto vb = loop_vertices[loop_start_offset + lo + pi];
      DirEdge de{(Index)va.id, (Index)vb.id, (int)va.source, (int)vb.source};
      if (base_edges.count(de)) continue;
      non_base_count[de]++;
    }
  }

  for (auto &[de, cnt] : non_base_count) {
    DirEdge rev{de.b_id, de.a_id, de.b_src, de.a_src};
    auto rev_it = non_base_count.find(rev);
    if (cnt == 1 && rev_it != non_base_count.end() &&
        rev_it->second == 1) {
      result.n_ok++;
    } else if (cnt > 1) {
      result.n_multi++;
    } else if (rev_it == non_base_count.end()) {
      result.n_no_rev++;
    }
  }

  if (!result.passed()) {
    std::cerr << "  [CUT INVARIANT FAIL] polygon=" << object
              << " n_loops=" << n_loops
              << " base_loop=" << base_loop.size()
              << " non_base_edges=" << non_base_count.size()
              << " paired=" << result.n_ok
              << " no_reverse=" << result.n_no_rev
              << " multi=" << result.n_multi << "\n";
  }

  return result;
}

} // namespace tf::cut::debug
