/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

#include "../core/local_vector.hpp"
#include "./search.hpp"

namespace tf {
template <std::size_t Dims, typename Policy0, typename Policy1, typename F,
          typename Iterator>
auto gather_ids(const tf::form<Dims, Policy0> &form0,
                const tf::form<Dims, Policy1> &form1, const F &predicate,
                Iterator out) -> Iterator {
  auto get_index_t = [](auto form) {
    if constexpr (tf::has_id_policy<decltype(form[0])>)
      return form[0].id();
    else
      return typename decltype(form)::index_t(0);
  };
  using index_t0 = std::decay_t<decltype(get_index_t(form0))>;
  using index_t1 = std::decay_t<decltype(get_index_t(form1))>;
  struct holder_t {
    index_t0 id0;
    index_t1 id1;

    operator std::pair<index_t0, index_t1>() const { return {id0, id1}; }
  };
  tf::local_vector<holder_t> l_ids;
  tf::search(form0, form1, predicate, [&](const auto &obj0, const auto &obj1) {
    if (predicate(obj0, obj1))
      l_ids.push_back(holder_t{obj0.id(), obj1.id()});
  });
  return l_ids.to_iterator(out);
}
template <std::size_t Dims, typename Policy, typename F, typename Iterator>
auto gather_ids(const tf::form<Dims, Policy> &form, const F &predicate,
                Iterator out) -> Iterator {
  tf::search(form, predicate, [&](const auto &obj) {
    if (predicate(obj))
      *out++ = obj.id();
  });
  return out;
}
} // namespace tf
