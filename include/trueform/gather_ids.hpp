/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

#include "./form.hpp"
#include "./local_vector.hpp"
#include "./search_broad.hpp"
#include "./transformed.hpp"

namespace tf {
template <std::size_t Dims, typename Policy0, typename Policy1, typename F,
          typename Iterator>
auto gather_ids(const tf::form<Dims, Policy0> &form0,
                const tf::form<Dims, Policy1> &form1, const F &predicate,
                Iterator out) -> Iterator {
  using index_t = typename Policy0::index_t;
  auto predicate_f = [&](const auto &obj0, const auto &obj1) {
    return predicate(tf::transformed(obj0, form0.transformation()),
                     tf::transformed(obj1, form1.transformation()));
  };
  tf::local_vector<index_t> l_ids;
  tf::search_broad(
      form0.tree(), form1.tree(), predicate_f,
      [&](const auto &ids0, const auto &ids1) {
        for (auto id0 : ids0) {
          auto obj0 = tf::inject_id(
              id0, tf::transformed(form0[id0], form0.transformation()));
          for (auto id1 : ids1) {
            if (predicate_f(form0.tree().primitive_aabbs()[id0],
                            form1.tree().primitive_aabbs()[id1]) &&
                predicate_f(
                    obj0, tf::inject_id(
                              id1, tf::transformed(form1[id1],
                                                   form1.transformation())))) {
              l_ids.push_back(id0);
              return false;
            }
          }
        }
        return false;
      },[]{return false;});
  return l_ids.to_iterator(out);
}
} // namespace tf
