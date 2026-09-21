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
#include "trueform/intersect/intersect_mode.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace tf::cpp::intersect_detail {

/// The facade refuses what the compiled entry cannot answer: a tolerance its
/// coordinate type cannot hold, a mode naming an invalid classifier or within
/// combination, and — where `allow_within` is false — a `within` the entry's
/// own arity states.
template <typename Real>
auto require_config(tf::intersect_config config, const char *operation,
                    bool allow_within = false) -> void {
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

  const auto mode = static_cast<int>(config.mode);
  const auto classifier =
      mode & (static_cast<int>(tf::intersect_mode::sos) |
              static_cast<int>(tf::intersect_mode::primitives));
  const bool within = config.mode & tf::intersect_mode::within;
  if ((classifier != static_cast<int>(tf::intersect_mode::sos) &&
       classifier != static_cast<int>(tf::intersect_mode::primitives)) ||
      (mode != classifier &&
       mode != (classifier | static_cast<int>(tf::intersect_mode::within))))
    throw std::invalid_argument(std::string(operation) +
                                ": mode must be sos or primitives, optionally "
                                "with within");
  if (!allow_within && within)
    throw std::invalid_argument(std::string(operation) +
                                ": within is stated by the entry's own arity");
}

} // namespace tf::cpp::intersect_detail
