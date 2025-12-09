/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

#include <trueform/core/frame.hpp>
#include <trueform/core/policy/frame.hpp>
#include <trueform/spatial/form.hpp>
#include <trueform/spatial/intersects.hpp>

namespace tf::py {
template <typename FormWrapper0, typename FormWrapper1>
auto form_intersects_form(FormWrapper0 &form_wrapper0,
                          FormWrapper1 &form_wrapper1) {
  bool has0 = form_wrapper0.has_transformation();
  bool has1 = form_wrapper1.has_transformation();
  auto form0 =
      tf::make_form(form_wrapper0.tree(), form_wrapper0.make_primitive_range());
  auto form1 =
      tf::make_form(form_wrapper1.tree(), form_wrapper1.make_primitive_range());
  if (has0 && has1)
    return tf::intersects(
        form0 | tf::tag(tf::make_frame(form_wrapper0.transformation_view())),
        form1 | tf::tag(tf::make_frame(form_wrapper1.transformation_view())));
  else if (has0 && !has1)
    return tf::intersects(
        form0 | tf::tag(tf::make_frame(form_wrapper0.transformation_view())),
        form1);
  else if (!has0 && has1)
    return tf::intersects(
        form0,
        form1 | tf::tag(tf::make_frame(form_wrapper1.transformation_view())));
  else
    return tf::intersects(form0, form1);
}
} // namespace tf::py
