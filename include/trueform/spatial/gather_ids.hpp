/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

#include "../core/local_vector.hpp"
#include "./search.hpp"

namespace tf {
template <std::size_t Dims, typename Policy0, typename Policy1, typename F0,
          typename F1, typename Iterator>
auto gather_ids(const tf::form<Dims, Policy0> &form0,
                const tf::form<Dims, Policy1> &form1, const F0 &aabbs_predicate,
                const F1 &primitives_predicate, Iterator out) -> Iterator {
  auto get_index_t = [](auto form) {
    if constexpr (tf::has_id_policy<decltype(form[0])>)
      return form[0].id();
    else
      return typename decltype(form)::index_type(0);
  };
  using index_t0 = std::decay_t<decltype(get_index_t(form0))>;
  using index_t1 = std::decay_t<decltype(get_index_t(form1))>;
  struct holder_t {
    index_t0 id0;
    index_t1 id1;
  };
  tf::local_vector<holder_t> l_ids;
  tf::search(form0, form1, aabbs_predicate,
             [&](const auto &obj0, const auto &obj1) {
               if (primitives_predicate(obj0, obj1))
                 l_ids.push_back(holder_t{obj0.id(), obj1.id()});
             });
  for (const auto &v : l_ids.vectors())
    for (const auto &e : v)
      *out++ = std::make_pair(e.id0, e.id1);
  return out;
}

template <std::size_t Dims, typename Policy0, typename Policy1, typename F,
          typename Iterator>
auto gather_ids(const tf::form<Dims, Policy0> &form0,
                const tf::form<Dims, Policy1> &form1, const F &predicate,
                Iterator out) -> Iterator {
  return gather_ids(form0, form1, predicate, predicate, out);
}

template <std::size_t Dims, typename Policy, typename F0, typename F1,
          typename Iterator>
auto gather_ids(const tf::form<Dims, Policy> &form, const F0 &aabb_predicate,
                const F1 &primitive_predicate, Iterator out) -> Iterator {
  tf::search(form, aabb_predicate, [&](const auto &obj) {
    if (primitive_predicate(obj))
      *out++ = obj.id();
  });
  return out;
}

template <std::size_t Dims, typename Policy, typename F, typename Iterator>
auto gather_ids(const tf::form<Dims, Policy> &form, const F &predicate,
                Iterator out) -> Iterator {
  return gather_ids(form, predicate, predicate, out);
}
} // namespace tf
