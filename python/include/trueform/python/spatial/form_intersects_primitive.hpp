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

#include <trueform/core/frame.hpp>
#include <trueform/core/form.hpp>
#include <trueform/spatial/policy/tree.hpp>
#include <trueform/spatial/intersects.hpp>

namespace tf::py {
template <typename FormWrapper, typename Primitive>
auto form_intersects_primitive(FormWrapper &form_wrapper,
                               const Primitive &primitive) {
  if (form_wrapper.has_transformation()) {
    return tf::intersects(
        form_wrapper.make_primitive_range() | tf::tag(form_wrapper.tree()) |
            tf::tag(tf::make_frame(form_wrapper.transformation_view())),
        primitive);
  } else {
    return tf::intersects(
        form_wrapper.make_primitive_range() | tf::tag(form_wrapper.tree()),
        primitive);
  }
}
} // namespace tf::py
