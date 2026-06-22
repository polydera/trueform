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
#include <tbb/parallel_invoke.h>
#include <tbb/task_group.h>
#include <utility>
#include <vector>

namespace tf::py {

template <typename Wrapper> auto compute_intersect_structures(Wrapper &w) {
  w.build_tree();
  auto fm = w.compute_face_membership_if_missing();
  auto mel = w.compute_manifold_edge_link_if_missing(fm);
  return std::make_pair(std::move(fm), std::move(mel));
}

template <typename Wrapper, typename Computed>
auto commit_intersect_structures(Wrapper &w, Computed &c) -> void {
  if (c.first)
    w.commit_face_membership(std::move(*c.first));
  if (c.second)
    w.commit_manifold_edge_link(std::move(*c.second));
}

template <typename Wrapper>
auto build_intersect_structures(Wrapper &w) -> void {
  auto c = compute_intersect_structures(w);
  commit_intersect_structures(w, c);
}

template <typename Wrapper0, typename Wrapper1>
auto build_intersect_structures(Wrapper0 &a, Wrapper1 &b) -> void {
  decltype(compute_intersect_structures(a)) ca;
  decltype(compute_intersect_structures(b)) cb;
  tbb::parallel_invoke([&] { ca = compute_intersect_structures(a); },
                       [&] { cb = compute_intersect_structures(b); });
  commit_intersect_structures(a, ca);
  commit_intersect_structures(b, cb);
}

template <typename Wrappers>
auto build_intersect_structures_all(Wrappers &wrappers) -> void {
  using W = typename Wrappers::value_type;
  std::size_t n = wrappers.size();
  std::vector<decltype(compute_intersect_structures(std::declval<W &>()))> cs(n);
  tbb::task_group tg;
  for (std::size_t i = 0; i < n; ++i)
    tg.run([&, i] { cs[i] = compute_intersect_structures(wrappers[i]); });
  tg.wait();
  for (std::size_t i = 0; i < n; ++i)
    commit_intersect_structures(wrappers[i], cs[i]);
}

} // namespace tf::py
