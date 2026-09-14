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

#include "distance_impl.hpp"
#include "primitive_dispatch.hpp"
#include "trueform/cpp/spatial/signed_distance.hpp"

#include "trueform/core/algorithm/parallel_for_each.hpp"
#include "trueform/core/buffer.hpp"
#include "trueform/core/checked.hpp"
#include "trueform/core/policy/frame.hpp"
#include "trueform/core/views/sequence_range.hpp"
#include "trueform/cpp/core/mesh.hpp"
#include "trueform/cpp/core/nd_array.hpp"
#include "trueform/spatial/policy/tree.hpp"
#include "trueform/spatial/policy/winding.hpp"
#include "trueform/spatial/signed_distance.hpp"

#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace tf::cpp {
namespace detail {

/// A signed distance is a distance, so the door, the empty batch and the result
/// are the distance family's. This is the one refusal that is its own: the sign
/// is a winding number, and a winding number is of a point.
template <typename Real, std::size_t Dims>
auto require_point_query(const primitive<Real, Dims> &query) -> void {
  if (query.kind() != primitive_kind::point)
    throw std::invalid_argument("signed_distance: query must be a point");
}

template <typename Index, typename Real, std::size_t Ngon>
auto winding_form_of(const mesh<Index, Real, 3, Ngon> &form) {
  return form.polygons() | tf::tag(form.tree()) |
         tf::tag(form.winding_moments()) | tf::tag(form.frame());
}

template <typename Index, typename Real, std::size_t Ngon>
auto signed_distance_compute(const mesh<Index, Real, 3, Ngon> &form,
                             const primitive<Real, 3> &query)
    -> distance_result<Real> {
  const auto values = query.data();
  const auto *data = values.raw_data();
  const auto native_form = winding_form_of(form);
  require_nonempty_form(native_form, "signed_distance");
  if (!query.is_batch())
    return distance_result<Real>(
        static_cast<Real>(tf::signed_distance(native_form, as_point<3>(data))));

  const auto count = query.count();
  const auto stride = query.element_stride();
  tf::buffer<Real> output_buffer;
  output_buffer.allocate(count);
  auto *output = output_buffer.data();
  const auto compute = [&](int index) {
    output[index] = static_cast<Real>(tf::signed_distance(
        native_form,
        as_point<3>(data + static_cast<std::size_t>(index) * stride)));
  };
  tf::parallel_for_each(tf::make_sequence_range(count), compute,
                        tf::checked(distance_form_parallel_threshold));

  return distance_result<Real>(
      nd_array<Real>::from_buffer(std::move(output_buffer), {count}));
}

} // namespace detail

template <typename Index, typename FormReal, std::size_t Dims, std::size_t Ngon,
          typename PrimitiveReal, std::enable_if_t<Dims == 3, int>>
auto signed_distance(const mesh<Index, FormReal, Dims, Ngon> &form,
                     const primitive<PrimitiveReal, Dims> &query)
    -> distance_result<FormReal> {
  detail::require_point_query(query);
  if (detail::empty_batch_distance(query))
    return detail::empty_distance_batch<FormReal>();
  return detail::signed_distance_compute(
      form, detail::cast_primitive<FormReal>(query));
}

} // namespace tf::cpp
