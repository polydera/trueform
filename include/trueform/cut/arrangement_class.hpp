/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via info@polydera.com.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

#include <array>
#include "../core/direction.hpp"

namespace tf {
enum class arrangement_class : unsigned {
  none = 0,
  inside = 1,
  outside = 2,
  aligned_boundary = 4,
  opposing_boundary = 8,
  on_boundary = aligned_boundary | opposing_boundary
};

inline constexpr auto operator|(arrangement_class a, arrangement_class b)
    -> arrangement_class {
  return static_cast<arrangement_class>(static_cast<unsigned>(a) |
                                        static_cast<unsigned>(b));
}

inline constexpr auto operator&(arrangement_class a, arrangement_class b)
    -> bool {
  return (static_cast<unsigned>(a) & static_cast<unsigned>(b)) != 0;
}

// Returns face directions for boolean operations based on arrangement classes.
// Reverses faces only for difference operations: when one mesh contributes
// "inside" faces (the carved region) while the other contributes "outside".
// For intersection (both inside) or union (both outside), no reversal needed.
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
