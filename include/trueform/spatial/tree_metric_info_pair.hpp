/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include <utility>

namespace tf {

/// @brief Result of a nearest-point query between two spatial trees.
///
/// Represents the closest pair of primitives found between two trees, along
/// with their closest points and the associated distance metric. This structure
/// is returned by @ref tf::nearness_search.
///
/// Includes utility accessors for checking validity and retrieving the distance
/// metric.
///
/// @tparam Index The type used for primitive identifiers (typically an
/// integer).
/// @tparam RealT The scalar coordinate type (e.g., float or double).
/// @tparam Dims The spatial dimension (typically 2 or 3).
template <typename Index, typename InfoT> struct tree_metric_info_pair {
  using index_t = Index;
  using element_t = std::pair<Index, Index>;
  using info_t = InfoT;
  using real_t = decltype(std::declval<info_t>().metric);
  //
  static constexpr Index no_id = -1;
  /// @brief A pair of primitive ids
  std::pair<Index, Index> elements{no_id, no_id};
  /// @brief A @ref tf::closest_point_pair
  info_t info;

  /// @brief Converts to bool, signaling validity
  operator bool() const { return elements.first != no_id; }
  auto metric() const { return info.metric; }
  auto metric(real_t val) { info.metric = val; }
};
} // namespace tf
