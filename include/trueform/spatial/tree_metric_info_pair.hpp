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
#include <utility>

namespace tf {

/// @ingroup spatial_results
/// @brief Result of a nearest-neighbor query between two spatial trees.
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
template <typename Index0, typename Index1, typename InfoT> struct tree_metric_info_pair {
  using element_t = std::pair<Index0, Index1>;
  using info_t = InfoT;
  using real_t = decltype(std::declval<info_t>().metric);
  //
  static constexpr Index0 no_id = -1;
  /// @brief A pair of primitive ids
  element_t elements{no_id, no_id};
  /// @brief A @ref tf::closest_point_pair
  info_t info;

  /// @brief Converts to bool, signaling validity
  operator bool() const { return elements.first != no_id; }
  auto metric() const { return info.metric; }
  auto metric(real_t val) { info.metric = val; }
};
} // namespace tf
