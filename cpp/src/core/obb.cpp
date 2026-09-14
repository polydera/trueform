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
#include "trueform/cpp/core/obb.hpp"

#include "trueform/core/obb_from.hpp"

#include <cstddef>
#include <stdexcept>
#include <utility>

namespace tf::cpp {
namespace {

template <typename Real, typename OBB>
auto to_result(const OBB &box) -> obb_result<Real> {
  tf::buffer<Real> origin_buffer;
  origin_buffer.allocate(3);
  for (int index = 0; index < 3; ++index)
    origin_buffer[static_cast<std::size_t>(index)] =
        static_cast<Real>(box.origin[index]);

  tf::buffer<Real> axes_buffer;
  axes_buffer.allocate(9);
  for (int axis = 0; axis < 3; ++axis)
    for (int coordinate = 0; coordinate < 3; ++coordinate)
      axes_buffer[static_cast<std::size_t>(axis * 3 + coordinate)] =
          static_cast<Real>(box.axes[axis][coordinate]);

  tf::buffer<Real> extent_buffer;
  extent_buffer.allocate(3);
  for (int index = 0; index < 3; ++index)
    extent_buffer[static_cast<std::size_t>(index)] =
        static_cast<Real>(box.extent[index]);

  return {
      nd_array<Real>::from_buffer(std::move(origin_buffer), {3}),
      nd_array<Real>::from_buffer(std::move(axes_buffer), {3, 3}),
      nd_array<Real>::from_buffer(std::move(extent_buffer), {3}),
  };
}

} // namespace

template <typename Real>
auto obb_from(const obb_options<Real> &options) -> obb_result<Real> {
  if (!options.points.is_valid() || options.points.ndim() != 2 ||
      options.points.shape_at(1) != 3)
    throw std::invalid_argument("obb_from: points must have shape [N, 3]");
  return to_result<Real>(
      tf::obb_from(tf::make_points<3>(options.points.make_range())));
}

#define TF_CPP_INSTANTIATE_OBB(Real)                                           \
  template auto obb_from(const obb_options<Real> &) -> obb_result<Real>

TF_CPP_MATRIX_FOR_EACH_REAL(TF_CPP_INSTANTIATE_OBB)

#undef TF_CPP_INSTANTIATE_OBB

} // namespace tf::cpp
