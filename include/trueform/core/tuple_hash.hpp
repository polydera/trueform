/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "./join_hashes.hpp"
#include "./zip_apply.hpp"
#include <functional>
#include <tuple>
namespace tf {
template <typename... Ts> class tuple_hash {
public:
  template <typename T> auto operator()(const T &tuple_like) const {
    return tf::zip_apply(
        [](auto &&...pairs) {
          using std::get;
          return tf::core::join_hashes(get<1>(pairs)(get<0>(pairs))...);
        },
        tuple_like, _hashes);
  }

private:
  std::tuple<std::hash<Ts>...> _hashes;
};
} // namespace tf
