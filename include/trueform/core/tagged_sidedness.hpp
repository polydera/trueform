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
#include "./sidedness.hpp"

namespace tf {

/// @ingroup core
/// @brief A sidedness value paired with an operand tag id.
///
/// @tparam Index The integer type used for the operand tag id.
template <typename Index> struct tagged_sidedness {
  Index tag;
  tf::sidedness side;
};

} // namespace tf
