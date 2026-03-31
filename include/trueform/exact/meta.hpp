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

#include "./int32.hpp"
#include "./int64.hpp"
#include "./int128.hpp"
#include "./int256.hpp"

namespace tf::exact {

template <typename T> struct meta;

template <> struct meta<int32> {
  using T0 = int32;
  using T1 = int64;
  using unsigned_T1 = std::uint64_t;
  using T2 = int128;
};

template <> struct meta<int64> {
  using T0 = int64;
  using T1 = int128;
  using unsigned_T1 = uint128;
  using T2 = int256;
};

} // namespace tf::exact
