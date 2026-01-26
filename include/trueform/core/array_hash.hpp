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
#include "./join_hashes.hpp"
#include <functional>
#include <tuple>
namespace tf {

/// @ingroup core
/// @brief Hash functor for std::array.
///
/// Combines element hashes using join_hashes.
///
/// @tparam T The array element type.
/// @tparam Size The array size.
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
