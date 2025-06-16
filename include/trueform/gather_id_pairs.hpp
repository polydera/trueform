/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

#include "./local_vector.hpp"
#include "./search.hpp"

namespace tf {
template <std::size_t Dims, typename Policy0, typename Policy1, typename F,
          typename Iterator>
auto gather_id_pairs(const tf::form<Dims, Policy0> &form0,
                     const tf::form<Dims, Policy1> &form1, const F &predicate,
                     Iterator out) -> Iterator {
  using index_t0 = typename Policy0::index_t;
  using index_t1 = typename Policy1::index_t;
  struct holder_t {
    index_t0 id0;
    index_t1 id1;
  };
  tf::local_vector<holder_t> l_ids;
  tf::search(form0, form1, predicate, [&](const auto &obj0, const auto &obj1) {
    if (predicate(obj0, obj1))
      l_ids.push_back(holder_t{obj0.id(), obj1.id()});
  });
  for (const auto &v : l_ids.vectors())
    for (const auto &e : v)
      *out++ = {e.id0, e.id1};
  return out;
}
} // namespace tf
