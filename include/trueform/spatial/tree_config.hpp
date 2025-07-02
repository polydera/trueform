/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../core/aabb_from.hpp"
#include <type_traits>
#include <utility>

namespace tf {

struct tree_node_config {
  int inner_size;
  int leaf_size;
};


/// @ingroup spatial_structures
/// @brief Configuration object for controlling tree construction.
///
/// `tree_config` defines the parameters and behavior used during tree construction,
/// including maximum node sizes and the function for generating primitive AABBs.
///
/// Users are encouraged to create this object using the @ref tf::config_tree helper
/// functions, which provide type deduction and convenient defaults.
///
/// @tparam F A callable type used to generate an AABB from a primitive.
///           Signature: `(const T&) -> tf::aabb<RealT, N>`
template <typename F> struct tree_config : private F {
  tree_config(F f, int inner_size, int leaf_size)
      : F(std::move(f)), node_config{inner_size, leaf_size} {}

  template <typename T> auto make_aabb(T &&t) const -> decltype(auto) {
    return F::operator()(static_cast<T &&>(t));
  }

  tree_node_config node_config;
};


/// @ingroup spatial_structures
/// @brief Create a tree configuration using a user-provided AABB construction function.
///
/// Returns a `tree_config` object that defines how the tree will be built,
/// including inner and leaf node sizes and a callable for computing primitive AABBs.
///
/// @tparam F A callable type that returns an AABB from a primitive.
/// @param inner_size The maximum number of children per internal node.
/// @param leaf_size The maximum number of primitives per leaf node.
/// @param make_aabb_f A function that computes an AABB for a given primitive.
///                    Signature: `(const T&) -> tf::aabb<RealT, N>`
///
/// @return A `tree_config<F>` instance for use with `tf::tree::build(...)`.
template <typename F>
auto config_tree(int inner_size, int leaf_size, F &&make_aabb_f) {
  return tree_config<std::decay_t<F>>{static_cast<F &&>(make_aabb_f),
                                      inner_size, leaf_size};
}


/// @ingroup spatial_structures
/// @brief Create a default tree configuration using `tf::aabb_from(...)`.
///
/// Returns a `tree_config` object with the provided node sizes and a default
/// AABB constructor that delegates to `aabb_from(...)`. This is suitable
/// for primitive types that specialize or support `aabb_from(...)`.
///
/// @param inner_size The maximum number of children per internal node.
/// @param leaf_size The maximum number of primitives per leaf node.
///
/// @return A `tree_config` instance for use with `tf::tree::build(...)`.
inline auto config_tree(int inner_size, int leaf_size) {
  auto make_aabb_f = [](const auto &x) {
    using tf::aabb_from;
    return aabb_from(x);
  };
  return tree_config<decltype(make_aabb_f)>{std::move(make_aabb_f), inner_size,
                                            leaf_size};
}

} // namespace tf
