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

#include <array>
#include "../core/direction.hpp"

namespace tf {

/// @ingroup cut_types
/// @brief Classification of face positions in boolean operations.
///
/// Describes where a face lies relative to another mesh:
/// inside, outside, or on a boundary (aligned or opposing normals).
/// Used by @ref tf::make_mesh_arrangements to classify regions.
enum class arrangement_class : unsigned {
  none = 0,                ///< No classification.
  inside = 1,              ///< Face is inside the other mesh.
  outside = 2,             ///< Face is outside the other mesh.
  aligned_boundary = 4,    ///< Coplanar faces with aligned normals.
  opposing_boundary = 8,   ///< Coplanar faces with opposing normals.
  on_boundary = aligned_boundary | opposing_boundary  ///< Any boundary type.
};

/// @ingroup cut_types
/// @brief Combine arrangement classes with bitwise OR.
/// @param a First @ref tf::arrangement_class.
/// @param b Second @ref tf::arrangement_class.
/// @return Combined classification.
inline constexpr auto operator|(arrangement_class a, arrangement_class b)
    -> arrangement_class {
  return static_cast<arrangement_class>(static_cast<unsigned>(a) |
                                        static_cast<unsigned>(b));
}

/// @ingroup cut_types
/// @brief Test if arrangement class contains a flag.
/// @param a The @ref tf::arrangement_class to test.
/// @param b The flag to check for.
/// @return True if the flag is set.
inline constexpr auto operator&(arrangement_class a, arrangement_class b)
    -> bool {
  return (static_cast<unsigned>(a) & static_cast<unsigned>(b)) != 0;
}

/// @ingroup cut_types
/// @brief Compute face directions for boolean operations.
///
/// Returns face directions based on arrangement classes.
/// Reverses faces only for difference operations: when one mesh contributes
/// "inside" faces (the carved region) while the other contributes "outside".
/// For intersection (both inside) or union (both outside), no reversal needed.
///
/// @param c0 The @ref tf::arrangement_class for the first mesh.
/// @param c1 The @ref tf::arrangement_class for the second mesh.
/// @return Array of @ref tf::direction for each mesh.
inline constexpr auto make_directions(arrangement_class c0, arrangement_class c1)
    -> std::array<direction, 2> {
  return {(c0 & arrangement_class::inside) && (c1 & arrangement_class::outside)
              ? direction::reverse
              : direction::forward,
          (c1 & arrangement_class::inside) && (c0 & arrangement_class::outside)
              ? direction::reverse
              : direction::forward};
}
} // namespace tf
