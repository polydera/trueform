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
#include <cstdint>

namespace tf {

/// @ingroup topology
/// @brief One recorded constraint split: the input constraint edge and the
/// dyadic parameter numerator/2^depth along it (numerator odd, canonical).
template <typename Index> struct cdt_constraint_split {
  Index edge;
  std::uint8_t numerator;
  std::uint8_t depth;
};

} // namespace tf
