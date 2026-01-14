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
#include "./zip_apply.hpp"
#include <functional>
#include <tuple>
namespace tf {

/// @ingroup core
/// @brief Hash functor for tuple-like types.
///
/// Combines element hashes using join_hashes.
///
/// @tparam Ts The tuple element types.
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
