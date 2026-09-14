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

#include "trueform/topology/triangulation_type.hpp"

#include <stdexcept>
#include <string>

namespace tf::cpp::arrangement_detail {

/// @brief The cut surface a caller asks for, refused where it is not one.
///
/// The one producer of that fact for every entry that builds an arrangement,
/// whichever tier it enters from; the caller states the door it is refused at.
inline auto require_triangulation(tf::triangulation_type triangulation,
                                  const char *operation) -> void {
  switch (triangulation) {
  case tf::triangulation_type::cdt:
  case tf::triangulation_type::refined_cdt:
    return;
  }
  throw std::invalid_argument(std::string(operation) +
                              ": invalid triangulation type");
}

} // namespace tf::cpp::arrangement_detail
