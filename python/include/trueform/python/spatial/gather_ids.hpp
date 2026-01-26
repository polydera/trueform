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

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <trueform/core/buffer.hpp>
#include <trueform/core/form.hpp>
#include <trueform/python/util/make_numpy_array.hpp>
#include <trueform/spatial/policy/tree.hpp>
#include <trueform/spatial/gather_ids.hpp>

namespace tf::py {

template <typename RealT, std::size_t Dims, typename FormWrapper,
          typename F0, typename F1>
auto gather_ids(FormWrapper &form_wrapper, const F0 &aabb_predicate,
                const F1 &primitive_predicate) {
  using Index = typename std::decay_t<decltype(form_wrapper.tree())>::index_type;

  tf::buffer<Index> buffer;

  if (form_wrapper.has_transformation()) {
    tf::gather_ids(
        form_wrapper.make_primitive_range() | tf::tag(form_wrapper.tree()) |
            tf::tag(tf::make_frame(form_wrapper.transformation_view())),
        aabb_predicate, primitive_predicate, std::back_inserter(buffer));
  } else {
    tf::gather_ids(form_wrapper.make_primitive_range() | tf::tag(form_wrapper.tree()),
                   aabb_predicate, primitive_predicate, std::back_inserter(buffer));
  }

  // Get size and release ownership
  size_t n = buffer.size();
  Index *data = buffer.release();

  // Create ndarray with proper empty array handling
  return make_numpy_array(data, {n});
}

} // namespace tf::py
