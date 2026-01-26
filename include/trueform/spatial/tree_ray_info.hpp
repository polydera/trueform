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
namespace tf {

/// @ingroup spatial_results
/// @brief Stores intersection metadata for ray queries in spatial trees.
///
/// This structure is used to represent the result of a ray query against a spatial tree.
/// It associates a hit element index with additional user-defined
/// information about the intersection (e.g., hit position, normal, distance).
///
/// The `InfoT` type is typically either @ref ray_hit_info or @ref ray_cast_info,
/// depending on the use case.
///
/// @tparam Index The integer type used to identify elements (e.g., `std::size_t`, `int32_t`).
/// @tparam InfoT A user-defined structure containing additional ray hit information.
///               Common options include @ref ray_hit_info and @ref ray_cast_info.
template <typename Index, typename InfoT>
struct tree_ray_info {
  /// @brief Type alias for the element index.
  using index_t = Index;

  /// @brief Type alias for the element type (same as index).
  using element_t = Index;

  /// @brief Type alias for the associated hit information.
  using info_t = InfoT;
  /// @brief Type alias for the associated real type.
  using real_t = typename InfoT::real_t;

  /// @brief Sentinel value indicating no intersection occurred.
  static constexpr Index no_id = -1;

  /// @brief The index of the intersected element, or `no_id` if no hit occurred.
  Index element{no_id};

  /// @brief Additional information about the intersection.
  info_t info{};

  /// @brief Returns `true` if a valid hit was recorded (i.e., `element != no_id`).
  operator bool() const { return element != no_id; }
};

}
