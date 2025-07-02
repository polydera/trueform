/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "./search/self_search_dispatch.hpp"
namespace tf {

template <typename Index, typename RealT, std::size_t N, typename F0,
          typename F1>
auto search_self(const tf::tree<Index, RealT, N> &tree, const F0 &check_aabbs,
                 const F1 &primitive_apply, int paralelism_depth = 6) -> bool {
  return spatial::search_self_dispatch<Index>(
      tree, check_aabbs, primitive_apply, paralelism_depth);
}

template <std::size_t N, typename Policy, typename F0, typename F1>
auto search_self(const tf::form<N, Policy> &form, const F0 &check_aabbs,
                 const F1 &primitive_apply, int paralelism_depth = 6) -> bool {
  return spatial::search_self_form_dispatch<typename Policy::index_t>(
      form, check_aabbs, primitive_apply, paralelism_depth);
}
} // namespace tf
