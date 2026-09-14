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

#include "../util/make_numpy_array.hpp"
#include <array>
#include <cstddef>
#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <trueform/volume/volume_buffer.hpp>
#include <utility>

namespace tf::py {

/// A natively produced field crosses the boundary as an ownership transfer:
/// the samples move into NumPy through make_numpy_array, and the facade wraps
/// them back into a borrowing volume_wrapper — so the produced buffer is the
/// one allocation and Python is its owner.
template <typename T, typename Coord>
auto to_python_volume(tf::volume_buffer<T, Coord, 3> &&vol) {
  const auto dims = vol.dims();
  const auto &sp = vol.spacing();
  const auto &og = vol.origin();
  const std::array<double, 3> spacing{static_cast<double>(sp[0]),
                                      static_cast<double>(sp[1]),
                                      static_cast<double>(sp[2])};
  const std::array<double, 3> origin{static_cast<double>(og[0]),
                                     static_cast<double>(og[1]),
                                     static_cast<double>(og[2])};
  auto samples = make_numpy_array(std::move(vol.samples_buffer()));
  return nanobind::make_tuple(samples, dims, spacing, origin);
}

} // namespace tf::py
