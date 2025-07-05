/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "./join_hashes.hpp"
#include <functional>
#include <tuple>
namespace tf {
template <typename T, std::size_t Size> class array_hash {
public:
  auto operator()(const std::array<T, Size> &array) const {
    return std::apply(
        [this](auto &&...ts) { return tf::core::join_hashes(_hash(ts)...); },
        array);
  }

private:
  std::hash<T> _hash;
};
} // namespace tf
