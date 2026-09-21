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
#include "../../core/local_value.hpp"
#include "../classify/intersection_payload.hpp"
#include "./face_pair_search.hpp"
#include "./face_self_kernels.hpp"

#include <atomic>

namespace tf::intersect {

/// Whether the form states a self record at all: @ref
/// tf::polygon_intersections' one-form discovery, stopped at the first
/// record that survives the election.
///
/// The verdict is written one way only — a pair that emits stores true and
/// nothing ever stores false — so `false` is what stands when every pair
/// has been seen, whatever order the threads walked in.
template <typename Index, typename Int, typename Form, typename Lattice>
auto find_self_intersection(const Form &form, const Lattice &lattice) -> bool {
  using workspace_t =
      face_pair_workspace<Index, Int, tf::exact::edge_fractions<Int, Index>>;
  auto &&mel = form.manifold_edge_link();
  auto &&fm = form.face_membership();

  tf::local_value<workspace_t> ws;
  std::atomic_bool found{false};
  auto abort = [&found] { return found.load(); };
  search_face_pairs_self(
      form, 0, lattice, ws,
      [&](workspace_t &w, bool is_self) {
        w.intersections.clear();
        w.payloads.clear();
        w.coplanar_pairs.clear();
        self_process(w, is_self, form, 0, mel, fm);
        if (w.intersections.size() != 0)
          found.store(true);
      },
      abort);
  return found.load();
}

} // namespace tf::intersect
