/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

#include "../core/local_vector.hpp"
#include "./search_self.hpp"

namespace tf {
template <std::size_t Dims, typename Policy, typename F0, typename F1,
          typename Iterator>
auto gather_self_ids(const tf::form<Dims, Policy> &form,
                     const F0 &aabbs_predicate, const F1 &primitives_predicate,
                     Iterator out) -> Iterator {
  auto get_index_t = [](auto form) {
    if constexpr (tf::has_id_policy<decltype(form[0])>)
      return form[0].id();
    else
      return typename decltype(form)::index_t(0);
  };
  using index_t = std::decay_t<decltype(get_index_t(form))>;
  struct holder_t {
    index_t id0;
    index_t id1;

    operator std::pair<index_t, index_t>() const { return {id0, id1}; }
    operator std::array<index_t, 2>() const { return {id0, id1}; }
  };
  tf::local_vector<holder_t> l_ids;
  tf::search_self(form, aabbs_predicate,
                  [&](const auto &obj0, const auto &obj1) {
                    if (primitives_predicate(obj0, obj1))
                      l_ids.push_back(holder_t{obj0.id(), obj1.id()});
                  });
  return l_ids.to_iterator(out);
}

template <std::size_t Dims, typename Policy, typename F, typename Iterator>
auto gather_self_ids(const tf::form<Dims, Policy> &form, const F &predicate,
                     Iterator out) -> Iterator {
  return gather_self_ids(form, predicate, predicate, out);
}
} // namespace tf
