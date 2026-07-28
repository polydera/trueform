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
#include <limits>

namespace tf::exact {

template <typename T> struct meta;

template <> struct meta<int32> {
  using T0 = int32;
  using T1 = int64;
  using unsigned_T1 = std::uint64_t;
  using T2 = int128;
  /// Value bits of a lattice coordinate and of the two rungs above it.
  /// The ladder doubles at every step, so both follow from the lattice.
  static constexpr int coordinate_bits = std::numeric_limits<T0>::digits;
  static constexpr int t1_bits = 2 * (coordinate_bits + 1) - 1;
  static constexpr int t2_bits = 4 * (coordinate_bits + 1) - 1;

  /// Dyadic parameter along an edge, as fine as the exact arithmetic
  /// permits rather than a chosen width: it is carried in T1, and
  /// @ref tf::exact::dyadic_ratio multiplies a squared edge length by
  /// 2^param_bits in T2. That length is a THREE-dimensional one, so it
  /// occupies 2*coordinate_bits + 4 bits — a coordinate difference spans
  /// coordinate_bits + 1, its square twice that, and three of them add two
  /// more. Budgeting for a plane instead leaves a lattice built straight
  /// from integer input, which the converter does not rescale, overflowing
  /// T2 on its longest edge.
  using param_type = T1;
  static constexpr int param_bits =
      (t1_bits - 1) < (t2_bits - 2 * coordinate_bits - 4)
          ? (t1_bits - 1)
          : (t2_bits - 2 * coordinate_bits - 4);

  /// The scale a split's position is stated on, and with it the definition
  /// of split identity: two splits on one edge are the same split exactly
  /// when they occupy the same value. It is the parameter's own resolution,
  /// so a position is carried as finely as it is computed; how far apart two
  /// positions must be to be two splits is decided where the split is
  /// placed, by @ref tf::exact::snap_parameter_to_edge, in lattice units.
  static constexpr int split_grid_bits = param_bits;
  static_assert(split_grid_bits <= param_bits,
                "a split parameter is carried in a parameter, so its scale "
                "cannot exceed the parameter's own");

  /// The real type this lattice is resolved from by
  /// @ref tf::exact::resolve_int_type, and the bits of the lattice that
  /// type cannot express. A coordinate is finer than its source by this
  /// many bits, so nothing inside that gap came from the model.
  using real_type = float;
  static constexpr int conversion_ulp_bits =
      coordinate_bits - std::numeric_limits<real_type>::digits;
};

template <> struct meta<int64> {
  using T0 = int64;
  using T1 = int128;
  using unsigned_T1 = uint128;
  using T2 = int256;
  /// Value bits of a lattice coordinate and of the two rungs above it.
  /// The ladder doubles at every step, so both follow from the lattice.
  static constexpr int coordinate_bits = std::numeric_limits<T0>::digits;
  static constexpr int t1_bits = 2 * (coordinate_bits + 1) - 1;
  static constexpr int t2_bits = 4 * (coordinate_bits + 1) - 1;

  /// Dyadic parameter along an edge, as fine as the exact arithmetic
  /// permits rather than a chosen width: it is carried in T1, and
  /// @ref tf::exact::dyadic_ratio multiplies a squared edge length by
  /// 2^param_bits in T2. That length is a THREE-dimensional one, so it
  /// occupies 2*coordinate_bits + 4 bits — a coordinate difference spans
  /// coordinate_bits + 1, its square twice that, and three of them add two
  /// more. Budgeting for a plane instead leaves a lattice built straight
  /// from integer input, which the converter does not rescale, overflowing
  /// T2 on its longest edge.
  using param_type = T1;
  static constexpr int param_bits =
      (t1_bits - 1) < (t2_bits - 2 * coordinate_bits - 4)
          ? (t1_bits - 1)
          : (t2_bits - 2 * coordinate_bits - 4);

  /// The scale a split's position is stated on, and with it the definition
  /// of split identity: two splits on one edge are the same split exactly
  /// when they occupy the same value. It is the parameter's own resolution,
  /// so a position is carried as finely as it is computed; how far apart two
  /// positions must be to be two splits is decided where the split is
  /// placed, by @ref tf::exact::snap_parameter_to_edge, in lattice units.
  static constexpr int split_grid_bits = param_bits;
  static_assert(split_grid_bits <= param_bits,
                "a split parameter is carried in a parameter, so its scale "
                "cannot exceed the parameter's own");

  /// The real type this lattice is resolved from by
  /// @ref tf::exact::resolve_int_type, and the bits of the lattice that
  /// type cannot express. A coordinate is finer than its source by this
  /// many bits, so nothing inside that gap came from the model.
  using real_type = double;
  static constexpr int conversion_ulp_bits =
      coordinate_bits - std::numeric_limits<real_type>::digits;
};

} // namespace tf::exact
