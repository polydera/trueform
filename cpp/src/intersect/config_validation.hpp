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

#include "trueform/intersect/intersect_config.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace tf::cpp::intersect_detail {

template <typename Real>
auto require_config(tf::intersect_config config, const char *operation,
                    bool allow_self_intersections = false) -> void {
  if (!std::isfinite(config.tolerance))
    throw std::invalid_argument(std::string(operation) +
                                ": tolerance must be finite");
  if (config.tolerance < 0.0)
    throw std::invalid_argument(std::string(operation) +
                                ": tolerance must be non-negative");
  if (!std::isfinite(static_cast<Real>(config.tolerance)))
    throw std::invalid_argument(
        std::string(operation) +
        ": tolerance must be representable in the coordinate type");

  constexpr auto base_mask = static_cast<int>(tf::intersect_mode::sos) |
                             static_cast<int>(tf::intersect_mode::primitives);
  constexpr auto self_mask =
      static_cast<int>(tf::intersect_mode::self_intersections);
  constexpr auto known_mask =
      base_mask |
      static_cast<int>(tf::intersect_mode::resolve_crossing_contours) |
      static_cast<int>(tf::intersect_mode::resolve_self_crossing_contours) |
      self_mask;
  const auto mode = static_cast<int>(config.mode);
  if ((mode & ~known_mask) != 0)
    throw std::invalid_argument(std::string(operation) +
                                ": mode contains unknown flags");
  const auto base_mode = mode & base_mask;
  if (base_mode != static_cast<int>(tf::intersect_mode::sos) &&
      base_mode != static_cast<int>(tf::intersect_mode::primitives))
    throw std::invalid_argument(std::string(operation) +
                                ": mode must select exactly one base mode");
  if (!allow_self_intersections && (mode & self_mask) != 0)
    throw std::invalid_argument(
        std::string(operation) +
        ": self-intersection flags are not applicable to curve extraction");
}

} // namespace tf::cpp::intersect_detail
